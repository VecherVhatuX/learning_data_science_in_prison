import os
import json
from dataclasses import dataclass
from typing import List
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim

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

class DataLoaderHelper:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def load_dataset(self):
        training_data = self.load_json_data("train.json")
        testing_data = self.load_json_data("test.json")
        return training_data, testing_data

    def load_json_data(self, file_name):
        try:
            with open(file_name, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"{file_name} not found.")
            return None

    def preprocess_example(self, example):
        input_ids = [0] + [ord(c) for c in f"{self.hyperparameters.conversation_format_identifier} {example['input']}"] + [1]
        labels = [0] + [ord(c) for c in f"{self.hyperparameters.conversation_format_identifier} {example['output']}"] + [1]
        attention_mask = [1] * len(input_ids)
        return np.array(input_ids, dtype=np.float32), np.array(labels, dtype=np.float32), np.array(attention_mask, dtype=np.float32)

class Dataset(Dataset):
    def __init__(self, data, data_loader_helper):
        self.data = data
        self.data_loader_helper = data_loader_helper
        self.sample_indices = np.arange(len(self.data))
        self.epoch = 0
        self.batch_indices = []

        positive_samples = list(range(len(self.data)))
        negative_samples = [np.random.choice(len(self.data)) for _ in range(len(self.data) * self.data_loader_helper.hyperparameters.negative_samples_per_positive_sample)]
        self.sample_indices = np.concatenate((positive_samples, negative_samples))

    def __iter__(self):
        self.shuffle()
        self.epoch += 1
        for batch_indices in self.batch_indices:
            batch_inputs, batch_labels, batch_attention_masks = [], [], []
            for idx in batch_indices:
                if idx < len(self.data):
                    input_ids, labels, attention_mask = self.data_loader_helper.preprocess_example(self.data[idx])
                else:
                    input_ids, labels, attention_mask = self.data_loader_helper.preprocess_example(self.data[np.random.choice(len(self.data))])
                batch_inputs.append(input_ids)
                batch_labels.append(labels)
                batch_attention_masks.append(attention_mask)
            yield torch.tensor(np.array(batch_inputs)), torch.tensor(np.array(batch_labels)), torch.tensor(np.array(batch_attention_masks))

    def shuffle(self):
        np.random.shuffle(self.sample_indices)
        self.batch_indices = np.array_split(self.sample_indices, len(self.sample_indices) // self.data_loader_helper.hyperparameters.training_batch_size)

    def __len__(self):
        return len(self.batch_indices)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(128, 128)  # input shape (128,); number of hidden units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1000)

    def forward(self, x):
        x = x.view(-1, 128)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer:
    def __init__(self, hyperparameters, model):
        self.hyperparameters = hyperparameters
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_model(self, dataset):
        for epoch in range(self.hyperparameters.number_of_epochs):
            total_loss = 0
            for i, (batch_inputs, batch_labels, _) in enumerate(dataset):
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.loss_function(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / (i+1)}")
        torch.save(self.model.state_dict(), os.path.join(self.hyperparameters.output_directory_path, "final_model.pth"))

    def evaluate_model(self, dataset):
        total_loss = 0
        for batch_inputs, batch_labels, _ in dataset:
            batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(batch_inputs)
                loss = self.loss_function(outputs, batch_labels)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(list(dataset))}")

def load_hyperparameters(base_model_identifier, conversation_format_identifier, triplet_loss_training_enabled):
    return Hyperparameters(base_model_identifier=base_model_identifier, conversation_format_identifier=conversation_format_identifier, triplet_loss_training_enabled=triplet_loss_training_enabled)

def main():
    hyperparameters = load_hyperparameters("t5-base", "none", True)
    model = NeuralNetwork()
    data_loader_helper = DataLoaderHelper(hyperparameters)
    training_data, testing_data = data_loader_helper.load_dataset()
    if training_data is not None:
        dataset = Dataset(training_data, data_loader_helper)
        trainer = Trainer(hyperparameters, model)
        trainer.train_model(dataset)
        test_dataset = Dataset(testing_data, data_loader_helper)
        trainer.evaluate_model(test_dataset)

if __name__ == "__main__":
    main()