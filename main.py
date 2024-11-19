import os
import json
from dataclasses import make_dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

Config = make_dataclass(
    "Config",
    [
        ("model_id", str, "t5-base"),
        ("chat_format", str, "none"),
        ("lora_alpha", int, 16),
        ("lora_dropout", float, 0.1),
        ("lora_rank", int, 64),
        ("target_layers", str, "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"),
        ("nested_quantization", bool, False),
        ("bit4_compute_type", str, "float16"),
        ("bit4_quant_storage_type", str, "uint8"),
        ("bit4_quant_type", str, "nf4"),
        ("flash_attention", bool, False),
        ("peft_lora", bool, False),
        ("bit8_quantization", bool, False),
        ("bit4_quantization", bool, False),
        ("reentrant", bool, False),
        ("unsloth", bool, False),
        ("triplet_loss_training", bool, False),
        ("dataset_name", str, "timdettmers/openassistant-guanaco"),
        ("append_token", bool, False),
        ("add_special_tokens", bool, False),
        ("data_splits", str, "train,test"),
        ("tokenized_data_path", str, None),
        ("output_path", str, "./results"),
        ("num_epochs", int, 3),
        ("train_batch_size", int, 16),
        ("eval_batch_size", int, 64),
        ("warmup_steps", int, 500),
        ("weight_decay", float, 0.01),
        ("log_path", str, "./logs"),
        ("save_steps", int, 500),
        ("max_checkpoints", int, 2),
        ("random_seed", int, 42),
        ("resume_checkpoint", str, None),
    ],
)

def create_model():
    return nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1000)
    )

def prepare_dataset(chat_format, data):
    return {
        "input_ids": [f"{chat_format} {example['input']}" for example in data],
        "labels": [f"{chat_format} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }

def load_dataset(chat_format):
    with open("train.json", 'r') as f:
        train_data = json.load(f)
    with open("test.json", 'r') as f:
        test_data = json.load(f)
    return prepare_dataset(chat_format, train_data), prepare_dataset(chat_format, test_data)

def training_step(model, device, criterion, optimizer, batch):
    inputs, labels = batch
    inputs, labels = torch.tensor(inputs, dtype=torch.float32).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def training_epoch(model, device, criterion, optimizer, dataset, batch_size):
    batches = np.array_split(dataset, len(dataset) // batch_size)
    total_loss = 0
    for batch in batches:
        loss = training_step(model, device, criterion, optimizer, (np.array(batch[:, 0], dtype=object), np.array(batch[:, 1], dtype=object)))
        total_loss += loss
    return total_loss / len(batches)

def train(config):
    model = create_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    train_data, _ = load_dataset(config.chat_format)
    dataset = np.array(list(zip(train_data["input_ids"], train_data["labels"])))
    for _ in range(config.num_epochs):
        loss = training_epoch(model, device, criterion, optimizer, dataset, config.train_batch_size)
        print(f"Epoch {_+1}, Loss: {loss}")
    return model

def main():
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    model = train(config)

if __name__ == "__main__":
    main()