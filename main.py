import os
import json
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

@dataclass
class Hyperparameters:
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

class TripletDataset(Dataset):
    def __init__(self, data, conversation_format, negative_samples):
        self.data = data
        self.conversation_format = conversation_format
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_ids = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format} {example['input']}"] + [1], dtype=torch.long)
        labels = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format} {example['output']}"] + [1], dtype=torch.long)
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.long)

        negative_examples = []
        for _ in range(self.negative_samples):
            negative_idx = torch.randint(0, len(self.data), (1,)).item()
            while negative_idx == idx:
                negative_idx = torch.randint(0, len(self.data), (1,)).item()
            negative_example = self.data[negative_idx]
            negative_input_ids = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format} {negative_example['input']}"] + [1], dtype=torch.long)
            negative_labels = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format} {negative_example['output']}"] + [1], dtype=torch.long)
            negative_attention_mask = torch.tensor([1] * len(negative_input_ids), dtype=torch.long)
            negative_examples.append({"input_ids": negative_input_ids, "labels": negative_labels, "attention_mask": negative_attention_mask})

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "negative_examples": negative_examples}

class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1000)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

class T5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('t5', 't5-base', use_cuda=torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids)
        return outputs.last_hidden_state

class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class ModelTrainer:
    def __init__(self, model, hyperparameters):
        self.model = model
        self.hyperparameters = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_step(self, batch):
        self.model.train()
        anchor_input_ids = batch["input_ids"]
        positive_input_ids = batch["labels"]
        negative_input_ids = batch["negative_examples"][0]["input_ids"]
        anchor_input_ids, positive_input_ids, negative_input_ids = anchor_input_ids.to(self.device), positive_input_ids.to(self.device), negative_input_ids.to(self.device)
        anchor_outputs = self.model(anchor_input_ids)
        positive_outputs = self.model(positive_input_ids)
        negative_outputs = self.model(negative_input_ids)
        triplet_loss = TripletLoss()
        loss = triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, dataset):
        for epoch in range(self.hyperparameters.num_epochs):
            total_loss = 0
            for batch in dataset:
                loss = self.train_step(batch)
                total_loss += loss
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

    def evaluate(self, dataset):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataset:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                input_ids, labels, attention_mask = input_ids.to(self.device), labels.to(self.device), attention_mask.to(self.device)
                outputs = self.model(input_ids)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(dataset)}")

def load_data(file_name):
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{file_name} not found.")
        return None

def create_data_loader(data, conversation_format, batch_size, negative_samples):
    dataset = TripletDataset(data, conversation_format, negative_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    hyperparameters = Hyperparameters(model_base="t5-base", conversation_format="none", triplet_loss_training=True)
    model = T5Model()
    training_data, testing_data = load_data("train.json"), load_data("test.json")
    if training_data is not None and testing_data is not None:
        train_data_loader = create_data_loader(training_data, hyperparameters.conversation_format, hyperparameters.train_batch_size, hyperparameters.negative_samples)
        test_data_loader = create_data_loader(testing_data, hyperparameters.conversation_format, hyperparameters.eval_batch_size, hyperparameters.negative_samples)
        trainer = ModelTrainer(model, hyperparameters)
        trainer.train(train_data_loader)
        trainer.evaluate(test_data_loader)

if __name__ == "__main__":
    main()