import os
import json
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

@dataclass
class ModelConfig:
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
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.dense_layer = nn.Linear(embedding_dim, embedding_dim)
        self.output_dense_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        x, _ = self.lstm_layer(x)
        x = self.dense_layer(x[:, -1, :])
        x = self.output_dense_layer(x)
        return x

class TripletDataset(Dataset):
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        input_ids = self.tokenizer.encode(example['input'], return_tensors='pt')
        labels = self.tokenizer.encode(example['output'], return_tensors='pt')
        negative_examples = []
        for _ in range(self.config.negative_samples):
            negative_example = self.tokenizer.encode(tf.strings.reduce_join(tf.random.shuffle(self.tokenizer.encode(example['input'], return_tensors='pt'))).numpy(), return_tensors='pt')
            negative_examples.append(negative_example)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'negative_examples': torch.cat(negative_examples)
        }

def load_json_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None

def create_triplet_dataset(data, config, tokenizer):
    return TripletDataset(data, config, tokenizer)

def train_triplet_model(model, optimizer, config, train_dataset, test_dataset):
    train_data = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    for epoch in range(config.num_epochs):
        total_loss = 0
        for batch in train_data:
            anchor = batch['input_ids'].squeeze(1)
            positive = batch['labels'].squeeze(1)
            negative = batch['negative_examples']
            optimizer.zero_grad()
            anchor_outputs = model(anchor)
            positive_outputs = model(positive)
            negative_outputs = model(negative)
            loss = F.relu(F.pairwise_distance(anchor_outputs, positive_outputs) - F.pairwise_distance(anchor_outputs, negative_outputs) + 2.0).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_data)}")
        test_loss = 0
        with torch.no_grad():
            for batch in test_data:
                input_ids = batch['input_ids'].squeeze(1)
                labels = batch['labels'].squeeze(1)
                outputs = model(input_ids)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()
        print(f"Epoch {epoch+1}, Test Loss: {test_loss / len(test_data)}")
        torch.save(model.state_dict(), "triplet_model.pth")

def main():
    config = ModelConfig()
    train_data = load_json_data("train.json")
    test_data = load_json_data("test.json")
    tokenizer = AutoTokenizer.from_pretrained(config.model_base)
    train_dataset = create_triplet_dataset(train_data, config, tokenizer)
    test_dataset = create_triplet_dataset(test_data, config, tokenizer)
    model = TripletModel(128, len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=0.001)
    train_triplet_model(model, optimizer, config, train_dataset, test_dataset)

if __name__ == "__main__":
    main()