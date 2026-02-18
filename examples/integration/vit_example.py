
import torch
import torchvision
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from datasets import load_dataset
from trainkeeper import (
    run_reproducible,
    distributed_training,
    wrap_model_ddp,
    CheckpointManager,
    GPUProfiler
)

@run_reproducible()
def main(run_ctx=None):
    with distributed_training() as dist_config:
        
        # 1. Dataset (CIFAR-10)
        # Using huggingface datasets for streaming
        dataset = load_dataset("cifar10")
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        def transform(examples):
            # Take a batch of images
            inputs = feature_extractor([x for x in examples["img"]], return_tensors="pt")
            inputs["labels"] = examples["label"]
            return inputs

        dataset = dataset.with_transform(transform)
        
        train_dataloader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
        
        # 2. Model
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=10,
            id2label={i: str(i) for i in range(10)},
            label2id={str(i): i for i in range(10)}
        )
        
        # 3. Distributed Wrap
        model = wrap_model_ddp(model, dist_config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # 4. Profiling
        if dist_config.is_main_process:
            profiler = GPUProfiler()
            profiler.start()
            ckpt_mgr = CheckpointManager(keep_best=1, metric="loss", mode="min")
            
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.train()
        
        for epoch in range(3):
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if dist_config.is_main_process:
                    profiler.step()
                    if step % 100 == 0:
                        print(f"Loss: {loss.item()}")
                        ckpt_mgr.save(model, optimizer, epoch, step=step, metrics={"loss": loss.item()})
        
        if dist_config.is_main_process:
            print(profiler.stop().summary())

if __name__ == "__main__":
    main()
