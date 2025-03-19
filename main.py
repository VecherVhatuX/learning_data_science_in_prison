import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

def get_model_config():
    return {
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

class SentenceEmbedder(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(SentenceEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear1(x)
        return self.linear2(x)

def triplet_loss(anchor, positive, negative):
    return torch.mean(
        torch.maximum(torch.mean(torch.square(anchor - positive), dim=-1) - 
        torch.mean(torch.square(anchor - negative), dim=-1) + 2.0, torch.tensor(0.0))
    )

class SentenceDataset(Dataset):
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokenizer.encode(self.data[idx]['input'], max_length=512, padding='max_length', truncation=True))
        labels = torch.tensor(self.tokenizer.encode(self.data[idx]['output'], max_length=512, padding='max_length', truncation=True))
        neg_samples = torch.tensor([self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != idx])]['input'],
                                   max_length=512, padding='max_length', truncation=True) for _ in range(self.config["negative_samples_per_batch"])])
        return input_ids, labels, neg_samples

def load_json_data(file_path):
    if os.path.exists(file_path):
        return json.load(open(file_path, 'r'))
    else:
        print(f"File not found: {file_path}")
        return None

def get_t5_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)

def setup_training_components(model, optimizer):
    return model, optimizer, triplet_loss

def train_model(model, config, data_loader, optimizer, loss_fn):
    model.train()
    for epoch in range(config["epochs"]):
        for input_ids, labels, neg_samples in data_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(input_ids), labels, neg_samples)
            loss.backward()
            optimizer.step()
        if early_stopping(loss.item()):
            print("Early stopping triggered.")
            break

def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0
    for input_ids, labels, neg_samples in data_loader:
        total_loss += loss_fn(model(input_ids), labels, neg_samples).item()
    print(f"Mean Evaluation Loss: {total_loss / len(data_loader):.4f}")

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def save_history(history, file_path):
    json.dump(history, open(file_path, 'w'))

def add_scheduler(optimizer, config):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def early_stopping(patience=3):
    best_loss = float('inf')
    counter = 0
    def check(current_loss):
        nonlocal best_loss, counter
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
        else:
            counter += 1
        return counter >= patience
    return check

def main():
    config = get_model_config()
    train_data = load_json_data("train.json")
    test_data = load_json_data("test.json")
    tokenizer = get_t5_tokenizer()
    train_dataset = SentenceDataset(train_data, config, tokenizer)
    test_dataset = SentenceDataset(test_data, config, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_sizes"]['train'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_sizes"]['eval'], shuffle=False)
    model = SentenceEmbedder(128, 30522)
    optimizer = get_optimizer(model)
    model, optimizer, loss_fn = setup_training_components(model, optimizer)
    scheduler = add_scheduler(optimizer, config)
    train_model(model, config, train_loader, optimizer, loss_fn)
    evaluate_model(model, test_loader, loss_fn)
    save_model(model, os.path.join(config["results_dir"], "triplet_model.pth"))

if __name__ == "__main__":
    main()