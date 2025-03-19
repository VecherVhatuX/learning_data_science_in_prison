import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

class ModelConfig:
    def __init__(self):
        self.settings = {
            "model_name": "t5-base", "alpha": 16, "dropout_rate": 0.1, "decomposition_rank": 64,
            "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            "quantization_config": {"nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
                                   "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
                                   "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
                                   "unsloth_training": False},
            "use_triplet_loss": True, "data_source": "timdettmers/openassistant-guanaco",
            "token_flags": {"use_special_token": False, "include_special_tokens": False}, "data_splits": ["train", "test"],
            "tokenized_file": None, "results_dir": "./results", "epochs": 3, "batch_sizes": {"train": 16, "eval": 64},
            "warmup_steps": 500, "weight_decay": 0.01, "logging_dir": "./logs", "model_save_interval": 500,
            "max_checkpoints": 2, "seed": 42, "checkpoint_path": None, "negative_samples_per_batch": 5
        }

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.fc2(x)

class DataHandler:
    @staticmethod
    def load_data(file):
        if os.path.exists(file):
            return json.load(open(file, 'r'))
        print(f"File not found: {file}")
        return None

    @staticmethod
    def tokenize_data(data, tokenizer, max_len=512):
        return torch.tensor(tokenizer.encode(data, max_length=max_len, padding='max_length', truncation=True))

class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, neg_samples=5):
        self.data = data
        self.tokenizer = tokenizer
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = DataHandler.tokenize_data(self.data[idx]['input'], self.tokenizer)
        labels = DataHandler.tokenize_data(self.data[idx]['output'], self.tokenizer)
        neg_samples = torch.stack([DataHandler.tokenize_data(self.data[random.choice([j for j in range(len(self.data)) if j != idx])]['input'], self.tokenizer) for _ in range(self.neg_samples)])
        return input_ids, labels, neg_samples

class TrainingManager:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_loss = float('inf')
        self.counter = 0

    def train(self, data_loader, epochs, patience=3):
        self.model.train()
        for epoch in range(epochs):
            for input_ids, labels, neg_samples in data_loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
                loss.backward()
                self.optimizer.step()
            if self.early_stop(loss.item(), patience):
                print("Early stopping triggered.")
                break

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        for input_ids, labels, neg_samples in data_loader:
            total_loss += self.loss_fn(self.model(input_ids), labels, neg_samples).item()
        print(f"Mean Evaluation Loss: {total_loss / len(data_loader):.4f}")

    def early_stop(self, current_loss, patience):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= patience

class ModelSaver:
    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def save_history(history, path):
        json.dump(history, open(path, 'w'))

def initialize_components():
    config = ModelConfig().settings
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = TextEncoder(30522, 128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return config, tokenizer, model, optimizer

def execute_training():
    config, tokenizer, model, optimizer = initialize_components()
    train_data = DataHandler.load_data("train.json")
    test_data = DataHandler.load_data("test.json")
    train_dataset = TripletDataset(train_data, tokenizer, config["negative_samples_per_batch"])
    test_dataset = TripletDataset(test_data, tokenizer, config["negative_samples_per_batch"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_sizes"]['train'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_sizes"]['eval'], shuffle=False)
    trainer = TrainingManager(model, optimizer, triplet_loss)
    trainer.train(train_loader, config["epochs"])
    trainer.evaluate(test_loader)
    ModelSaver.save_model(model, os.path.join(config["results_dir"], "triplet_model.pth"))

if __name__ == "__main__":
    execute_training()