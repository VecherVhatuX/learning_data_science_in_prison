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
    base_model_identifier: str = "t5-base"
    conversation_format_identifier: str = "none"
    low_rank_approximation_alpha: int = 16
    low_rank_approximation_dropout_rate: float = 0.1
    low_rank_approximation_rank: int = 64
    target_model_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quantization_enabled: bool = False
    four_bit_computation_data_type: str = "float16"
    four_bit_quantization_storage_data_type: str = "uint8"
    four_bit_quantization_type: str = "nf4"
    flash_attention_enabled: bool = False
    peft_low_rank_approximation_enabled: bool = False
    eight_bit_quantization_enabled: bool = False
    four_bit_quantization_enabled: bool = False
    reentrant_training_enabled: bool = False
    unsloth_training_enabled: bool = False
    triplet_loss_training_enabled: bool = True
    dataset_identifier: str = "timdettmers/openassistant-guanaco"
    append_special_token: bool = False
    add_special_tokens: bool = False
    dataset_splits: str = "train,test"
    tokenized_data_path: str = None
    output_directory_path: str = "./results"
    number_of_epochs: int = 3
    training_batch_size: int = 16
    evaluation_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay_rate: float = 0.01
    log_directory_path: str = "./logs"
    save_steps: int = 500
    maximum_checkpoints: int = 2
    random_seed_value: int = 42
    resume_checkpoint_path: str = None
    negative_samples_per_positive_sample: int = 5

class TripletDataset(Dataset):
    def __init__(self, data, conversation_format_identifier, negative_samples_per_positive_sample):
        self.data = data
        self.conversation_format_identifier = conversation_format_identifier
        self.negative_samples_per_positive_sample = negative_samples_per_positive_sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_ids = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format_identifier} {example['input']}"] + [1], dtype=torch.long)
        labels = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format_identifier} {example['output']}"] + [1], dtype=torch.long)
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.long)

        # Sample negative examples
        negative_examples = []
        for _ in range(self.negative_samples_per_positive_sample):
            negative_idx = torch.randint(0, len(self.data), (1,)).item()
            while negative_idx == idx:
                negative_idx = torch.randint(0, len(self.data), (1,)).item()
            negative_example = self.data[negative_idx]
            negative_input_ids = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format_identifier} {negative_example['input']}"] + [1], dtype=torch.long)
            negative_labels = torch.tensor([0] + [ord(c) for c in f"{self.conversation_format_identifier} {negative_example['output']}"] + [1], dtype=torch.long)
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

class DataLoaderFactory:
    @staticmethod
    def create_data_loader(data, conversation_format_identifier, batch_size, negative_samples_per_positive_sample):
        dataset = TripletDataset(data, conversation_format_identifier, negative_samples_per_positive_sample)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class DataUtil:
    @staticmethod
    def load_json_data(file_name):
        try:
            with open(file_name, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"{file_name} not found.")
            return None

class HyperparameterUtil:
    @staticmethod
    def load_hyperparameters(base_model_identifier, conversation_format_identifier, triplet_loss_training_enabled):
        return Hyperparameters(base_model_identifier=base_model_identifier, conversation_format_identifier=conversation_format_identifier, triplet_loss_training_enabled=triplet_loss_training_enabled)

class ModelUtil:
    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    @staticmethod
    def train_model(model, dataset, hyperparameters):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(hyperparameters.number_of_epochs):
            total_loss = 0
            for batch in dataset:
                loss = ModelUtil.train_step(model, optimizer, batch)
                total_loss += loss
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

    @staticmethod
    def evaluate_model(model, dataset):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataset:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                outputs = model(input_ids)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(dataset)}")

class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TripletModelUtil:
    @staticmethod
    def train_step(model, triplet_loss, optimizer, batch):
        model.train()
        anchor_input_ids = batch["input_ids"]
        positive_input_ids = batch["labels"]
        negative_input_ids = batch["negative_examples"][0]["input_ids"]
        anchor_outputs = model(anchor_input_ids)
        positive_outputs = model(positive_input_ids)
        negative_outputs = model(negative_input_ids)
        loss = triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @staticmethod
    def train_model(model, triplet_loss, dataset, hyperparameters):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(hyperparameters.number_of_epochs):
            total_loss = 0
            for batch in dataset:
                loss = TripletModelUtil.train_step(model, triplet_loss, optimizer, batch)
                total_loss += loss
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

class Main:
    def run(self):
        hyperparameters = HyperparameterUtil.load_hyperparameters("t5-base", "none", True)
        model = T5Model()
        triplet_loss = TripletLoss()
        training_data, testing_data = DataUtil.load_json_data("train.json"), DataUtil.load_json_data("test.json")
        if training_data is not None and testing_data is not None:
            train_data_loader = DataLoaderFactory.create_data_loader(training_data, hyperparameters.conversation_format_identifier, hyperparameters.training_batch_size, hyperparameters.negative_samples_per_positive_sample)
            test_data_loader = DataLoaderFactory.create_data_loader(testing_data, hyperparameters.conversation_format_identifier, hyperparameters.evaluation_batch_size, hyperparameters.negative_samples_per_positive_sample)
            TripletModelUtil.train_model(model, triplet_loss, train_data_loader, hyperparameters)
            ModelUtil.evaluate_model(model, test_data_loader)

if __name__ == "__main__":
    Main().run()