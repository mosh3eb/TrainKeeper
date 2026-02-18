
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from trainkeeper import (
    run_reproducible,
    distributed_training,
    wrap_model_ddp,
    CheckpointManager,
    GPUProfiler,
    ModelMonitor
)

@run_reproducible()
def main(run_ctx=None):
    # 1. Distributed Context
    with distributed_training() as dist_config:
        
        # 2. Model & Tokenizer
        model_name = "gpt2" # Using small model for example
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = wrap_model_ddp(model, dist_config) # One-line DDP!
        
        # 3. Dataset (Streaming for large scale!)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        
        def tokenize(element):
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=512,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == 512:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        
        # 4. DataLoader
        train_dataloader = DataLoader(tokenized_datasets, batch_size=4)
        
        # 5. Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # 6. TrainKeeper Tools
        if dist_config.is_main_process:
            monitor = ModelMonitor(window_size=100)
            ckpt_manager = CheckpointManager(keep_best=1, metric="loss", mode="min")
            profiler = GPUProfiler(check_interval=50)
            profiler.start()
            
        # 7. Training Loop
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.train()
        
        step = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            if dist_config.is_main_process:
                monitor.log_request(latency_ms=0, error=False) # Track throughput
                profiler.step("train_step")
                
                if step % 100 == 0:
                    print(f"Step {step}: Loss {loss.item()}")
                    ckpt_manager.save(model, optimizer, epoch=0, step=step, metrics={"loss": loss.item()})
            
            step += 1
            if step >= 100: # Short run for example
                break
                
        if dist_config.is_main_process:
            print(profiler.stop().summary())

if __name__ == "__main__":
    main()
