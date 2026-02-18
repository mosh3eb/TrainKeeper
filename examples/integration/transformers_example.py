
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset
import evaluate
from trainkeeper import (
    run_reproducible,
    distributed_training,
    wrap_model_ddp,
    CheckpointManager,
    GPUProfiler
)

@run_reproducible()
def main(run_ctx=None):
    # 1. Distributed Setup
    with distributed_training() as dist_config:
        print(f"Running on rank {dist_config.rank} of {dist_config.world_size}")
        
        # 2. Data Preparation
        dataset = load_dataset("glue", "mrpc")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        def tokenize_function(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        
        train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=8
        )
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"], batch_size=8
        )
        
        # 3. Model Setup
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model = wrap_model_ddp(model, dist_config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        num_epochs = 3
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        
        # 4. Checkpoint Manager & Profiler
        if dist_config.is_main_process:
            ckpt_manager = CheckpointManager(keep_best=2, metric="accuracy")
            profiler = GPUProfiler()
            profiler.start()
        
        # 5. Training Loop
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        
        for epoch in range(num_epochs):
            model.train()
            for batch in train_dataloader:
                if dist_config.is_main_process:
                     profiler.step("train_step")
                     
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 6. Evaluation
            model.eval()
            metric = evaluate.load("accuracy")
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            
            eval_metric = metric.compute()
            print(f"Epoch {epoch}: {eval_metric}")
            
            # 7. Save Checkpoint
            if dist_config.is_main_process:
                ckpt_manager.save(
                    model, 
                    optimizer, 
                    epoch, 
                    metrics=eval_metric
                )

        if dist_config.is_main_process:
            profile_report = profiler.stop()
            print(profile_report.summary())

if __name__ == "__main__":
    main()
