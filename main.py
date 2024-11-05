import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import optuna
import json
import random

@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_identifier: str = field(default="t5-base", metadata={"help": "Pre-trained model identifier from Hugging Face."})
    chat_template: Optional[str] = field(default="none", metadata={"help": "Format for chat template. Options: chatml, zephyr, none."})
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_rank: Optional[int] = field(default=64)
    lora_target_layers: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj", metadata={"help": "Target layers for LoRA."})
    nested_quant: Optional[bool] = field(default=False, metadata={"help": "Enable nested quantization for 4-bit base models."})
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16", metadata={"help": "Compute data type for 4-bit base models."})
    bnb_4bit_quant_storage_dtype: Optional[str] = field(default="uint8", metadata={"help": "Quantization storage data type for 4-bit base models."})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "Quantization type for 4-bit base models. Options: fp4, nf4."})
    use_flash_attention: Optional[bool] = field(default=False, metadata={"help": "Enable flash attention for training."})
    use_peft_lora: Optional[bool] = field(default=False, metadata={"help": "Enable PEFT LoRA for training."})
    use_8bit_quantization: Optional[bool] = field(default=False, metadata={"help": "Enable 8-bit quantization."})
    use_4bit_quantization: Optional[bool] = field(default=False, metadata={"help": "Enable 4-bit quantization."})
    use_reentrant: Optional[bool] = field(default=False, metadata={"help": "Enable reentrant gradient checkpointing."})
    use_unsloth: Optional[bool] = field(default=False, metadata={"help": "Enable UnSloth for training."})
    use_triplet_loss_trainer: Optional[bool] = field(default=False, metadata={"help": "Use TripletLossTrainer for training."})

@dataclass
class TrainingDataConfig:
    """Configuration for training data."""
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco", metadata={"help": "Dataset name."})
    append_concat_token: Optional[bool] = field(default=False, metadata={"help": "Append EOS token to each sample."})
    add_special_tokens: Optional[bool] = field(default=False, metadata={"help": "Add special tokens to each sample."})
    splits: Optional[str] = field(default="train,test", metadata={"help": "Comma-separated list of dataset splits."})
    tokenized_dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to tokenized dataset."})

@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = field(default="./results", metadata={"help": "Output directory for training results."})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Batch size per device for training."})
    per_device_eval_batch_size: int = field(default=64, metadata={"help": "Batch size per device for evaluation."})
    warmup_steps: int = field(default=500, metadata={"help": "Number of warmup steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    logging_dir: str = field(default="./logs", metadata={"help": "TensorBoard log directory."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X steps."})
    save_total_limit: int = field(default=2, metadata={"help": "Total number of checkpoints to save."})
    no_cuda: bool = field(default=False, metadata={"help": "Disable CUDA."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Resume training from checkpoint."})

class Dataset(torch.utils.data.Dataset):
    """Custom dataset class."""
    def __init__(self, dataset, epoch, batch_size):
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        random.shuffle(self.indices)
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        return batch

class TripletDataset(torch.utils.data.Dataset):
    """Custom triplet dataset class."""
    def __init__(self, dataset, epoch, batch_size):
        self.dataset = dataset
        self.epoch = epoch
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        random.shuffle(self.indices)
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        positive_samples = [sample for sample in batch if sample["label"] == 1]
        negative_samples = [sample for sample in batch if sample["label"] == 0]
        return positive_samples, negative_samples

class ModelTrainer:
    """Model trainer class."""
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.accelerator = Accelerator()

    def prepare_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_args.model_identifier)

    def create_datasets(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_identifier)
        datasets = torch.utils.data.ConcatDataset([torch.load(f"{self.data_args.dataset_name}/train.json"), torch.load(f"{self.data_args.dataset_name}/test.json")])
        apply_chat_template = self.data_args.chat_template != "none"

        def process_data(examples):
            if apply_chat_template:
                inputs = []
                labels = []
                for example in examples:
                    inputs.append(f"{example['input']} {tokenizer.sep_token}")
                    labels.append(f"{example['output']} {tokenizer.sep_token}")
                examples["input_ids"] = tokenizer(inputs, return_tensors="pt", truncation=True, padding="max_length").input_ids
                examples["labels"] = tokenizer(labels, return_tensors="pt", truncation=True, padding="max_length").input_ids
            else:
                examples["input_ids"] = tokenizer(examples["input"], return_tensors="pt", truncation=True, padding="max_length").input_ids
                examples["labels"] = tokenizer(examples["output"], return_tensors="pt", truncation=True, padding="max_length").input_ids

            return examples

        datasets = torch.utils.data.ConcatDataset([process_data(datasets[i]) for i in range(len(datasets))])
        return datasets

    def get_trainer(self, model, train_dataset, eval_dataset):
        if self.model_args.use_triplet_loss_trainer:
            return TripletLossTrainer(model=model, args=self.training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, layer_index=-1)
        else:
            return SFTTrainer(model=model, tokenizer=model.tokenizer, args=self.training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)

    def run_pipeline(self):
        model = self.prepare_model()
        datasets = self.create_datasets()
        train_dataset = Dataset(datasets, 0, self.training_args.per_device_train_batch_size)
        eval_dataset = datasets
        trainer = self.get_trainer(model, train_dataset, eval_dataset)
        trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        trainer.save_model()

class SFTTrainer:
    """SFT trainer class."""
    def __init__(self, model, tokenizer, args, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.accelerator = Accelerator()

    def train(self, resume_from_checkpoint=None):
        self.model, self.train_dataset, self.eval_dataset = self.accelerator.prepare(self.model, self.train_dataset, self.eval_dataset)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_dataset:
                input_ids = batch["input_ids"].to(self.accelerator.device)
                attention_mask = batch["attention_mask"].to(self.accelerator.device)
                labels = batch["labels"].to(self.accelerator.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                self.accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss / len(self.train_dataset)}")

    def save_model(self):
        self.accelerator.save(self.model.state_dict(), f"{self.args.output_dir}/model.pth")

class TripletLossTrainer(SFTTrainer):
    """Triplet loss trainer class."""
    def __init__(self, model, args, train_dataset, eval_dataset, layer_index):
        super().__init__(model, model.tokenizer, args, train_dataset, eval_dataset)
        self.layer_index = layer_index

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none")
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)

    pipeline = ModelTrainer(model_args, data_args, training_args)
    pipeline.run_pipeline()