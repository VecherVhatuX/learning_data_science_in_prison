import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import json
import os
from transformers import BertTokenizer
from typing import List, Tuple, Dict

# Constants
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_SIZE = 128
FULLY_CONNECTED_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE_VALUE = 1e-5
EPOCHS = 5
BATCH_SIZE = 32
NUM_NEGATIVES_PER_POSITIVE = 1

class DataLoaderUtil:
    @staticmethod
    def load_data(file_path: str) -> np.ndarray or Dict:
        if file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True)
        else:
            return json.load(open(file_path, 'r', encoding='utf-8'))

    @staticmethod
    def load_snippets(folder_path: str) -> List[Tuple[str, str]]:
        snippet_paths = []
        for folder in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, folder)):
                snippet_paths.append((os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')))
        return snippet_paths

    @staticmethod
    def separate_snippets(snippets: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, bool]], List[Tuple[str, bool]]]:
        bug_snippets = []
        non_bug_snippets = []
        for folder_path, snippet_file_path in snippets:
            snippet_data = DataLoaderUtil.load_data(snippet_file_path)
            if snippet_data.get('is_bug', False):
                bug_snippets.append((snippet_data['snippet'], True))
            else:
                non_bug_snippets.append((snippet_data['snippet'], False))
        return bug_snippets, non_bug_snippets

    @staticmethod
    def create_triplets(problem_statement: str, positive_snippets: List[Tuple[str, bool]], negative_snippets: List[Tuple[str, bool]], num_negatives_per_positive: int) -> List[Dict]:
        triplets = []
        for positive_doc in positive_snippets:
            for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
                triplets.append({'anchor': problem_statement, 'positive': positive_doc[0], 'negative': random.choice(negative_snippets)[0]})
        return triplets

    @staticmethod
    def prepare_data(dataset_path: str, snippet_folder_path: str, num_negatives_per_positive: int) -> Tuple[List[Dict], List[Dict]]:
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in DataLoaderUtil.load_data(dataset_path)}
        snippets = DataLoaderUtil.load_snippets(snippet_folder_path)
        bug_snippets, non_bug_snippets = DataLoaderUtil.separate_snippets(snippets)
        triplets = []
        for folder_path, _ in snippets:
            problem_statement = instance_id_map.get(os.path.basename(folder_path))
            triplets.extend(DataLoaderUtil.create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive))
        return random_split(triplets, [int(len(triplets)*0.8), len(triplets)-int(len(triplets)*0.8)])

class DatasetUtil:
    class CustomDataset(Dataset):
        def __init__(self, triplets: List[Dict], max_sequence_length: int, tokenizer: BertTokenizer):
            self.triplets = triplets
            self.max_sequence_length = max_sequence_length
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.triplets)

        def __getitem__(self, idx: int) -> Dict:
            anchor = self.tokenizer.encode_plus(
                self.triplets[idx]['anchor'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            positive = self.tokenizer.encode_plus(
                self.triplets[idx]['positive'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            negative = self.tokenizer.encode_plus(
                self.triplets[idx]['negative'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'anchor_input_ids': anchor['input_ids'].squeeze(0),
                'anchor_attention_mask': anchor['attention_mask'].squeeze(0),
                'positive_input_ids': positive['input_ids'].squeeze(0),
                'positive_attention_mask': positive['attention_mask'].squeeze(0),
                'negative_input_ids': negative['input_ids'].squeeze(0),
                'negative_attention_mask': negative['attention_mask'].squeeze(0)
            }

class ModelUtil:
    class Model(nn.Module):
        def __init__(self, embedding_size: int, fully_connected_size: int, dropout_rate: float):
            super(ModelUtil.Model, self).__init__()
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
            self.dropout = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(768, fully_connected_size)
            self.fc2 = nn.Linear(fully_connected_size, embedding_size)

        def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.model(x, attention_mask=attention_mask)
            outputs = outputs.pooler_output
            outputs = self.dropout(outputs)
            outputs = torch.relu(self.fc1(outputs))
            outputs = self.fc2(outputs)
            return outputs

class LossUtil:
    @staticmethod
    def calculate_loss(anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor) -> torch.Tensor:
        positive_distance = torch.mean(torch.pow(anchor_embeddings - positive_embeddings, 2))
        negative_distance = torch.mean(torch.pow(anchor_embeddings - negative_embeddings, 2))
        return positive_distance + torch.max(negative_distance - positive_distance, torch.tensor(0.0))

class TrainingUtil:
    @staticmethod
    def train(model: ModelUtil.Model, device: torch.device, dataset: DatasetUtil.CustomDataset, epochs: int, learning_rate_value: float, batch_size: int, optimizer: optim.Optimizer):
        for epoch in range(epochs):
            total_loss = 0
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for batch_idx, batch in enumerate(dataloader):
                anchor_input_ids = batch['anchor_input_ids'].to(device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(device)
                positive_input_ids = batch['positive_input_ids'].to(device)
                positive_attention_mask = batch['positive_attention_mask'].to(device)
                negative_input_ids = batch['negative_input_ids'].to(device)
                negative_attention_mask = batch['negative_attention_mask'].to(device)
                anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = model(positive_input_ids, positive_attention_mask)
                negative_embeddings = model(negative_input_ids, negative_attention_mask)
                loss = LossUtil.calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

class EvaluationUtil:
    @staticmethod
    def evaluate(model: ModelUtil.Model, device: torch.device, dataset: DatasetUtil.CustomDataset, batch_size: int):
        total_correct = 0
        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch_idx, batch in enumerate(dataloader):
                anchor_input_ids = batch['anchor_input_ids'].to(device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(device)
                positive_input_ids = batch['positive_input_ids'].to(device)
                positive_attention_mask = batch['positive_attention_mask'].to(device)
                negative_input_ids = batch['negative_input_ids'].to(device)
                negative_attention_mask = batch['negative_attention_mask'].to(device)
                anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = model(positive_input_ids, positive_attention_mask)
                negative_embeddings = model(negative_input_ids, negative_attention_mask)
                for i in range(len(anchor_embeddings)):
                    similarity_positive = torch.dot(anchor_embeddings[i], positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                    similarity_negative = torch.dot(anchor_embeddings[i], negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                    total_correct += int(similarity_positive > similarity_negative)
        accuracy = total_correct / len(dataset)
        print(f'Test Accuracy: {accuracy}')

class MainUtil:
    @staticmethod
    def create_model(embedding_size: int, fully_connected_size: int, dropout_rate: float, device: torch.device) -> ModelUtil.Model:
        model = ModelUtil.Model(embedding_size, fully_connected_size, dropout_rate)
        model.to(device)
        return model

    @staticmethod
    def create_optimizer(model: ModelUtil.Model, learning_rate_value: float) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=learning_rate_value)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    train_triplets, test_triplets = DataLoaderUtil.prepare_data(dataset_path, snippet_folder_path, NUM_NEGATIVES_PER_POSITIVE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = DatasetUtil.CustomDataset(train_triplets, MAX_SEQUENCE_LENGTH, tokenizer)
    test_dataset = DatasetUtil.CustomDataset(test_triplets, MAX_SEQUENCE_LENGTH, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MainUtil.create_model(EMBEDDING_SIZE, FULLY_CONNECTED_SIZE, DROPOUT_RATE, device)
    optimizer = MainUtil.create_optimizer(model, LEARNING_RATE_VALUE)
    TrainingUtil.train(model, device, train_dataset, EPOCHS, LEARNING_RATE_VALUE, BATCH_SIZE, optimizer)
    EvaluationUtil.evaluate(model, device, test_dataset, BATCH_SIZE)

if __name__ == "__main__":
    main()