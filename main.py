import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class Config:
    def __init__(self):
        self.model_name = "t5-base"
        self.conversation_style = "none"
        self.alpha = 16
        self.dropout_rate = 0.1
        self.decomposition_rank = 64
        self.layers_to_modify = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
        self.enable_nested_quantization = False
        self.dtype_for_four_bit = "float16"
        self.storage_dtype = "uint8"
        self.quantization_method = "nf4"
        self.use_flash_attention = False
        self.low_rank_peft = False
        self.enable_eight_bit_quantization = False
        self.enable_four_bit_quantization = False
        self.allow_reentrant_training = False
        self.allow_unsloth_training = False
        self.use_triplet_loss = True
        self.data_source = "timdettmers/openassistant-guanaco"
        self.special_token_flag = False
        self.special_tokens_inclusion = False
        self.data_splits = ["train", "test"]
        self.tokenized_data_file = None
        self.results_directory = "./results"
        self.num_epochs = 3
        self.batch_size_train = 16
        self.batch_size_eval = 64
        self.warmup_steps_count = 500
        self.weight_decay_rate = 0.01
        self.logging_directory = "./logs"
        self.model_save_frequency = 500
        self.max_checkpoints_to_keep = 2
        self.seed = 42
        self.checkpoint_resume_path = None
        self.negative_samples_per_batch = 5

class TripletNet(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(TripletNet, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, batch_first=True)
        self.fc_layer = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_tensor):
        embedded = self.embedding_layer(input_tensor)
        lstm_out, _ = self.lstm_layer(embedded)
        dense_out = self.fc_layer(lstm_out[:, -1, :])
        output = self.output_layer(dense_out)
        return output

    def calculate_triplet_loss(self, anchor, positive, negative):
        return torch.mean(torch.clamp(torch.mean((anchor - positive) ** 2) - torch.mean((anchor - negative) ** 2) + 2.0, min=0.0))

class TripletData(Dataset):
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.indices = list(range(len(data)))
        self.batch_size = config.batch_size_train

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        input_ids, labels, negative_exs = [], [], []
        for i in range(self.batch_size):
            idx = self.indices[index * self.batch_size + i]
            example = self.data[idx]
            input_ids.append(self.tokenizer.encode(example['input'], max_length=512, padding='max_length', truncation=True))
            labels.append(self.tokenizer.encode(example['output'], max_length=512, padding='max_length', truncation=True))
            for _ in range(self.config.negative_samples_per_batch):
                neg_idx = random.choice([j for j in range(len(self.data)) if j != idx])
                negative_ex = self.tokenizer.encode(self.data[neg_idx]['input'], max_length=512, padding='max_length', truncation=True)
                negative_exs.append(negative_ex)
        return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(negative_exs))

    def shuffle_indices(self):
        random.seed(self.config.seed)
        random.shuffle(self.indices)

def read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def train_triplet_model(model, config, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(config.num_epochs):
        for input_ids, labels, negative_exs in train_loader:
            optimizer.zero_grad()
            anchor_output = model(input_ids)
            positive_output = model(labels)
            negative_output = model(negative_exs)
            loss = model.calculate_triplet_loss(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()
        print(f"Completed Epoch: {epoch + 1}/{config.num_epochs}")

def evaluate_triplet_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels, negative_exs in test_loader:
            anchor_output = model(input_ids)
            positive_output = model(labels)
            negative_output = model(negative_exs)
            loss = model.calculate_triplet_loss(anchor_output, positive_output, negative_output)
            total_loss += loss.item()
    average_loss = total_loss / len(test_loader)
    print(f"Test Loss Average: {average_loss:.4f}")

def initialize_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

def main():
    config = Config()
    train_data = read_json("train.json")
    test_data = read_json("test.json")
    tokenizer = initialize_tokenizer()
    train_dataset = TripletData(train_data, config, tokenizer)
    test_dataset = TripletData(test_data, config, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size_eval, shuffle=False)
    model = TripletNet(128, 30522)
    train_triplet_model(model, config, train_loader)
    evaluate_triplet_model(model, test_loader)

if __name__ == "__main__":
    main()