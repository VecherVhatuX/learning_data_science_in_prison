import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import Tensor
import json
import random
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Data classes
@dataclass
class ModelConfig:
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
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco", metadata={"help": "Dataset name."})
    append_concat_token: Optional[bool] = field(default=False, metadata={"help": "Append EOS token to each sample."})
    add_special_tokens: Optional[bool] = field(default=False, metadata={"help": "Add special tokens to each sample."})
    splits: Optional[str] = field(default="train,test", metadata={"help": "Comma-separated list of dataset splits."})
    tokenized_dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to tokenized dataset."})

@dataclass
class TrainingConfig:
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
    def __init__(self, data, batch_size, num_negative_samples):
        self.data = data
        self.batch_size = batch_size
        self.num_negative_samples = num_negative_samples
        self.indices = list(range(len(data)))
        self.epoch = 0

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.data[i] for i in batch_indices]
        positive_samples = [sample for sample in batch if sample["label"] == 1]
        negative_samples = random.sample([sample for sample in batch if sample["label"] == 0], self.num_negative_samples)
        return positive_samples, negative_samples

    def epoch_shuffle(self):
        random.shuffle(self.indices)
        self.epoch += 1

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

class ModelManager:
    def __init__(self, model_args, training_args):
        self.model_args = model_args
        self.training_args = training_args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not training_args.no_cuda else "cpu")

    def create_model(self):
        return Transformer().to(self.device)

    def create_optimizer(self, model):
        return Adam(model.parameters(), lr=0.001)

    def save_model(self, model):
        torch.save(model.state_dict(), self.training_args.output_dir)

    def load_model(self, path):
        model = Transformer()
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        return model

class DatasetManager:
    def __init__(self, model_args, data_args):
        self.model_args = model_args
        self.data_args = data_args

    def load_data(self, file_name):
        return json.load(open(file_name))

    def process_data(self, examples):
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        apply_chat_template = self.model_args.chat_template != "none"
        if apply_chat_template:
            inputs = []
            labels = []
            for example in examples:
                inputs.append(f"{example['input']} {tokenizer.sep_token}")
                labels.append(f"{example['output']} {tokenizer.sep_token}")
            examples["input_ids"] = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids
            examples["labels"] = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids
        else:
            examples["input_ids"] = tokenizer([example["input"] for example in examples], return_tensors="pt", padding=True, truncation=True).input_ids
            examples["labels"] = tokenizer([example["output"] for example in examples], return_tensors="pt", padding=True, truncation=True).input_ids
        return examples

    def prepare_datasets(self):
        train_data = self.load_data("train.json")
        test_data = self.load_data("test.json")
        train_data = self.process_data(train_data)
        test_data = self.process_data(test_data)
        return train_data, test_data

    def create_data_loaders(self):
        train_data, test_data = self.prepare_datasets()
        train_dataset = Dataset(train_data, self.data_args.per_device_train_batch_size, 5)
        test_dataset = Dataset(test_data, self.data_args.per_device_eval_batch_size, 5)
        return train_dataset, test_dataset

class Trainer:
    def __init__(self, model, device, train_dataset, optimizer):
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.optimizer = optimizer

    def train_step(self, batch):
        positive_samples, negative_samples = batch
        input_ids = torch.tensor([sample["input_ids"][0] for sample in positive_samples + negative_samples], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([sample["attention_mask"][0] for sample in positive_samples + negative_samples], dtype=torch.long).to(self.device)
        labels = torch.tensor([sample["labels"][0] for sample in positive_samples + negative_samples], dtype=torch.long).to(self.device)
        self.model.zero_grad()
        loss = self.model(input_ids, attention_mask, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_dataset.epoch_shuffle()
            for batch in DataLoader(self.train_dataset, batch_size=1, shuffle=False):
                loss = self.train_step(batch)
            print(f"Epoch {epoch+1}, Loss: {loss}")

def run_pipeline(model_args, data_args, training_args):
    model_manager = ModelManager(model_args, training_args)
    dataset_manager = DatasetManager(model_args, data_args)
    model = model_manager.create_model()
    optimizer = model_manager.create_optimizer(model)
    train_dataset, _ = dataset_manager.create_data_loaders()
    trainer = Trainer(model, model_manager.device, train_dataset, optimizer)
    trainer.train(training_args.num_train_epochs)
    model_manager.save_model(model)

def resume_pipeline(model_args, data_args, training_args, checkpoint_path):
    model_manager = ModelManager(model_args, training_args)
    dataset_manager = DatasetManager(model_args, data_args)
    model = model_manager.load_model(checkpoint_path)
    optimizer = model_manager.create_optimizer(model)
    train_dataset, _ = dataset_manager.create_data_loaders()
    trainer = Trainer(model, model_manager.device, train_dataset, optimizer)
    trainer.train(training_args.num_train_epochs)

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none")
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)

    #run_pipeline(model_args, data_args, training_args)
    resume_pipeline(model_args, data_args, training_args, "./results/model.pth")