import os
import json
from dataclasses import dataclass, field
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import random

@dataclass
class Config:
    model_base: str = "t5-base"
    conversation_format: str = "none"
    low_rank_alpha: int = 16
    low_rank_dropout: float = 0.1
    low_rank_rank: int = 64
    target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quantization: bool = False
    four_bit_dtype: str = "float16"
    four_bit_storage_dtype: str = "uint8"
    four_bit_quantization: str = "nf4"
    flash_attention: bool = False
    peft_low_rank: bool = False
    eight_bit_quantization: bool = False
    four_bit_quantization_enabled: bool = False
    reentrant_training: bool = False
    unsloth_training: bool = False
    triplet_loss_training: bool = True
    dataset: str = "timdettmers/openassistant-guanaco"
    append_special_token: bool = False
    add_special_tokens: bool = False
    dataset_splits: str = "train,test"
    tokenized_data_path: str = None
    output_dir: str = "./results"
    num_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    log_dir: str = "./logs"
    save_steps: int = 500
    max_checkpoints: int = 2
    random_seed: int = 42
    resume_checkpoint: str = None
    negative_samples: int = 5

class TripletModel(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.dense_layer = nn.Linear(embedding_dim, embedding_dim)
        self.output_dense_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        x, _ = self.lstm_layer(x)
        x = self.dense_layer(x[:, -1, :])
        x = torch.relu(x)
        x = self.output_dense_layer(x)
        return x

    def compute_triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return F.relu(F.pairwise_distance(anchor, positive) - F.pairwise_distance(anchor, negative) + 2.0).mean()

class TripletDataset(Dataset):
    def __init__(self, data: Dict, config: Config, tokenizer: AutoTokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.indices = list(range(len(data)))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        example = self.data[self.indices[index]]
        input_ids = self.tokenizer.encode(example['input'], return_tensors='pt')
        labels = self.tokenizer.encode(example['output'], return_tensors='pt')
        negative_examples = []
        for _ in range(self.config.negative_samples):
            negative_index = random.randint(0, len(self.data) - 1)
            if negative_index == index:
                negative_index = (negative_index + 1) % len(self.data)
            negative_example = self.tokenizer.encode(self.data[negative_index]['input'], return_tensors='pt')
            negative_examples.append(negative_example)
        negative_examples = torch.cat(negative_examples)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'negative_examples': negative_examples
        }

    def on_epoch_end(self) -> None:
        random.seed(self.config.random_seed)
        random.seed(random.randint(0, 2**32))
        self.indices = random.sample(range(len(self.data)), len(self.data))

def load_data(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None

def train_model(model: TripletModel, optimizer: AdamW, scheduler, config: Config, train_dataset: TripletDataset, test_dataset: TripletDataset) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    def train_step(batch: Dict) -> float:
        anchor = batch['input_ids'].squeeze(1).to(device)
        positive = batch['labels'].squeeze(1).to(device)
        negative = batch['negative_examples'].to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            anchor_outputs = model(anchor)
            positive_outputs = model(positive)
            negative_outputs = model(negative)
            loss = model.compute_triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item()

    def test_step(batch: Dict) -> float:
        input_ids = batch['input_ids'].squeeze(1).to(device)
        labels = batch['labels'].squeeze(1).to(device)
        outputs = model(input_ids)
        loss = F.cross_entropy(outputs, labels)
        return loss.item()

    for epoch in range(config.num_epochs):
        train_dataset.on_epoch_end()
        model.train()
        total_loss = sum(train_step(batch) for batch in train_data) / len(train_data)
        print(f"Epoch {epoch+1}, Loss: {total_loss}")
        model.eval()
        test_loss = sum(test_step(batch) for batch in test_data) / len(test_data)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss}")
        torch.save(model.state_dict(), f"triplet_model_epoch_{epoch+1}.pth")

def main() -> None:
    config = Config()
    train_data = load_data("train.json")
    test_data = load_data("test.json")
    tokenizer = AutoTokenizer.from_pretrained(config.model_base)
    train_dataset = TripletDataset(train_data, config, tokenizer)
    test_dataset = TripletDataset(test_data, config, tokenizer)
    model = TripletModel(128, len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=0.001)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.num_epochs * len(train_dataset))
    train_model(model, optimizer, scheduler, config, train_dataset, test_dataset)

if __name__ == "__main__":
    main()