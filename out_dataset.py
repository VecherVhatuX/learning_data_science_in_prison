import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

class DataProcessor:
    @staticmethod
    def load_data(file_path: str) -> dict or np.ndarray:
        if file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True)
        else:
            return json.load(open(file_path, 'r', encoding='utf-8'))

    @staticmethod
    def load_snippets(snippet_folder_path: str) -> list:
        return [(os.path.join(snippet_folder_path, folder), os.path.join(snippet_folder_path, folder, 'snippet.json')) 
                for folder in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, folder))]

    @staticmethod
    def separate_snippets(snippets: list) -> tuple:
        bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                        for snippet_data in [DataProcessor.load_data(snippet_file_path)] if snippet_data.get('is_bug', False)]
        non_bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                            for snippet_data in [DataProcessor.load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)]
        return bug_snippets, non_bug_snippets

class TripletCreator:
    def __init__(self, num_negatives_per_positive: int):
        self.num_negatives_per_positive = num_negatives_per_positive

    def create_triplets(self, instance_id_map: dict, snippets: list) -> list:
        bug_snippets, non_bug_snippets = DataProcessor.separate_snippets(snippets)
        return [{'anchor': instance_id_map[os.path.basename(folder_path)], 'positive': positive_doc, 'negative': random.choice(non_bug_snippets)} 
                for folder_path, _ in snippets 
                for positive_doc in bug_snippets 
                for _ in range(min(self.num_negatives_per_positive, len(non_bug_snippets)))]

class BugTripletDataset(Dataset):
    def __init__(self, triplets: list, max_sequence_length: int):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']

        anchor_sequence = self.tokenizer.encode(anchor, max_length=self.max_sequence_length, padding='max_length', truncation=True, return_tensors='pt')
        positive_sequence = self.tokenizer.encode(positive, max_length=self.max_sequence_length, padding='max_length', truncation=True, return_tensors='pt')
        negative_sequence = self.tokenizer.encode(negative, max_length=self.max_sequence_length, padding='max_length', truncation=True, return_tensors='pt')

        return torch.cat([anchor_sequence, positive_sequence, negative_sequence], dim=0)

    def shuffle(self) -> None:
        random.shuffle(self.triplets)

class BugTripletModel(nn.Module):
    def __init__(self, embedding_size: int, fully_connected_size: int, dropout_rate: float):
        super(BugTripletModel, self).__init__()
        self.model = nn.Sequential(
            AutoModel.from_pretrained('bert-base-uncased'),
            nn.Linear(768, fully_connected_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fully_connected_size, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs.last_hidden_state[:, 0, :]

class Trainer:
    def __init__(self, learning_rate_value: float, epochs: int, batch_size: int, device: torch.device):
        self.learning_rate_value = learning_rate_value
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def calculate_loss(self, anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.mean((anchor_embeddings - positive_embeddings) ** 2) + torch.max(torch.mean((anchor_embeddings - negative_embeddings) ** 2) - torch.mean((anchor_embeddings - positive_embeddings) ** 2), torch.tensor(0.0).to(self.device))

    def train(self, model: BugTripletModel, dataset: BugTripletDataset) -> float:
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate_value)
        model.train()
        total_loss = 0
        for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
            batch = batch.to(self.device)
            anchor_embeddings = model(batch[:, 0, :])
            positive_embeddings = model(batch[:, 1, :])
            negative_embeddings = model(batch[:, 2, :])
            loss = self.calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataset)

    def evaluate(self, model: BugTripletModel, dataset: BugTripletDataset) -> float:
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
                batch = batch.to(self.device)
                anchor_embeddings = model(batch[:, 0, :])
                positive_embeddings = model(batch[:, 1, :])
                negative_embeddings = model(batch[:, 2, :])
                for i in range(len(anchor_embeddings)):
                    similarity_positive = torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                    similarity_negative = torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                    total_correct += int(similarity_positive > similarity_negative)
        return total_correct / len(dataset) / self.batch_size

class BugTripletModelRunner:
    def __init__(self, max_sequence_length: int, embedding_size: int, fully_connected_size: int, dropout_rate: float, learning_rate_value: float, epochs: int, batch_size: int, num_negatives_per_positive: int):
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.learning_rate_value = learning_rate_value
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_negatives_per_positive = num_negatives_per_positive
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, dataset_path: str, snippet_folder_path: str) -> tuple:
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in DataProcessor.load_data(dataset_path)}
        snippets = DataProcessor.load_snippets(snippet_folder_path)
        triplet_creator = TripletCreator(self.num_negatives_per_positive)
        triplets = triplet_creator.create_triplets(instance_id_map, snippets)
        return np.array_split(np.array(triplets), 2)

    def train_model(self, model: BugTripletModel, train_dataset: BugTripletDataset, test_dataset: BugTripletDataset) -> None:
        trainer = Trainer(self.learning_rate_value, self.epochs, self.batch_size, self.device)
        for epoch in range(self.epochs):
            loss = trainer.train(model, train_dataset)
            print(f'Epoch {epoch+1}, Loss: {loss}')
        print(f'Test Accuracy: {trainer.evaluate(model, test_dataset)}')

    def main(self, dataset_path: str, snippet_folder_path: str) -> None:
        train_triplets, test_triplets = self.prepare_data(dataset_path, snippet_folder_path)
        train_dataset = BugTripletDataset(train_triplets, self.max_sequence_length)
        test_dataset = BugTripletDataset(test_triplets, self.max_sequence_length)
        model = BugTripletModel(self.embedding_size, self.fully_connected_size, self.dropout_rate)
        model.to(self.device)
        self.train_model(model, train_dataset, test_dataset)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    model_runner = BugTripletModelRunner(max_sequence_length=512, embedding_size=128, fully_connected_size=64, dropout_rate=0.2, learning_rate_value=1e-5, epochs=5, batch_size=32, num_negatives_per_positive=1)
    model_runner.main(dataset_path, snippet_folder_path)