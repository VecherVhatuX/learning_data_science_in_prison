import os
import json
import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

CONFIG = {
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

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def load_json_data(file_path):
    return json.load(open(file_path, 'r')) if os.path.exists(file_path) else None

def tokenize_text(data, tokenizer, max_len=512):
    return torch.tensor(tokenizer.encode(data, max_length=max_len, padding='max_length', truncation=True))

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, neg_samples=5):
        self.data = data
        self.tokenizer = tokenizer
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = tokenize_text(self.data[idx]['input'], self.tokenizer)
        labels = tokenize_text(self.data[idx]['output'], self.tokenizer)
        neg_samples = torch.stack([tokenize_text(self.data[random.choice([j for j in range(len(self.data)) if j != idx])]['input'], self.tokenizer) for _ in range(self.neg_samples)])
        return input_ids, labels, neg_samples

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_loss = float('inf')
        self.counter = 0

    def train_model(self, data_loader, epochs, patience=3):
        for epoch in range(epochs):
            for input_ids, labels, neg_samples in data_loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
                loss.backward()
                self.optimizer.step()
                if self.check_early_stopping(loss.item(), patience):
                    print("Training halted early.")
                    return

    def evaluate_model(self, data_loader):
        total_loss = 0
        for input_ids, labels, neg_samples in data_loader:
            loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
            total_loss += loss.item()
        print(f"Average Loss on Evaluation: {total_loss / len(data_loader):.4f}")

    def check_early_stopping(self, current_loss, patience):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= patience

def save_model_weights(model, path):
    torch.save(model.state_dict(), path)

def save_training_history(history, path):
    json.dump(history, open(path, 'w'))

def setup_environment():
    config = CONFIG
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = LanguageModel(30522, 128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return config, tokenizer, model, optimizer

def run_pipeline():
    config, tokenizer, model, optimizer = setup_environment()
    train_data, test_data = load_json_data("train.json"), load_json_data("test.json")
    train_dataset = TextDataset(train_data, tokenizer, config["negative_samples_per_batch"])
    test_dataset = TextDataset(test_data, tokenizer, config["negative_samples_per_batch"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_sizes"]["train"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_sizes"]["eval"], shuffle=False)
    trainer = Trainer(model, optimizer, nn.TripletMarginLoss())
    trainer.train_model(train_loader, config["epochs"])
    trainer.evaluate_model(test_loader)
    save_model_weights(model, os.path.join(config["results_dir"], "triplet_model.pth"))

def add_learning_rate_scheduler(optimizer, initial_lr, decay_steps, decay_rate):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

def add_model_checkpoint(model, optimizer, checkpoint_dir, max_to_keep=2):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_{len(os.listdir(checkpoint_dir))}.pt"))

if __name__ == "__main__":
    run_pipeline()