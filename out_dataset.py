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

class BugTripletModel:
    def __init__(self, 
                 max_sequence_length=512, 
                 embedding_size=128, 
                 fully_connected_size=64, 
                 dropout_rate=0.2, 
                 learning_rate_value=1e-5, 
                 epochs=5, 
                 batch_size=32, 
                 num_negatives_per_positive=1):
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.learning_rate_value = learning_rate_value
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_negatives_per_positive = num_negatives_per_positive
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self, file_path):
        if file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True)
        else:
            return json.load(open(file_path, 'r', encoding='utf-8'))

    def load_snippets(self, snippet_folder_path):
        return [(os.path.join(snippet_folder_path, folder), os.path.join(snippet_folder_path, folder, 'snippet.json')) 
                for folder in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, folder))]

    def separate_snippets(self, snippets):
        bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                        for snippet_data in [self.load_data(snippet_file_path)] if snippet_data.get('is_bug', False)]
        non_bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                            for snippet_data in [self.load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)]
        return bug_snippets, non_bug_snippets

    def create_triplets(self, instance_id_map, snippets):
        bug_snippets, non_bug_snippets = self.separate_snippets(snippets)
        return [{'anchor': instance_id_map[os.path.basename(folder_path)], 'positive': positive_doc, 'negative': random.choice(non_bug_snippets)} 
                for folder_path, _ in snippets 
                for positive_doc in bug_snippets 
                for _ in range(min(self.num_negatives_per_positive, len(non_bug_snippets)))]

    def prepare_data(self, dataset_path, snippet_folder_path):
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in self.load_data(dataset_path)}
        snippets = self.load_snippets(snippet_folder_path)
        triplets = self.create_triplets(instance_id_map, snippets)
        return np.array_split(np.array(triplets), 2)

    def tokenize_triplets(self, triplets):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        anchor_sequences = [tokenizer.encode(triplet['anchor'], max_length=self.max_sequence_length, padding='max_length', truncation=True) for triplet in triplets]
        positive_sequences = [tokenizer.encode(triplet['positive'], max_length=self.max_sequence_length, padding='max_length', truncation=True) for triplet in triplets]
        negative_sequences = [tokenizer.encode(triplet['negative'], max_length=self.max_sequence_length, padding='max_length', truncation=True) for triplet in triplets]
        return torch.tensor(np.stack((anchor_sequences, positive_sequences, negative_sequences), axis=1))

    def create_model(self):
        model = nn.Sequential(
            AutoModel.from_pretrained('bert-base-uncased'),
            nn.Linear(768, self.fully_connected_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fully_connected_size, self.embedding_size)
        )
        return model

    def calculate_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return torch.mean((anchor_embeddings - positive_embeddings) ** 2) + torch.max(torch.mean((anchor_embeddings - negative_embeddings) ** 2) - torch.mean((anchor_embeddings - positive_embeddings) ** 2), torch.tensor(0).to(self.device))

    def train(self, model, dataset):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate_value)
        total_loss = 0
        for batch in dataset:
            batch = batch.to(self.device)
            anchor_embeddings = model(batch[:, 0])
            positive_embeddings = model(batch[:, 1])
            negative_embeddings = model(batch[:, 2])
            loss = self.calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataset)

    def evaluate(self, model, dataset):
        total_correct = 0
        for batch in dataset:
            batch = batch.to(self.device)
            anchor_embeddings = model(batch[:, 0])
            positive_embeddings = model(batch[:, 1])
            negative_embeddings = model(batch[:, 2])
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
        return total_correct / len(dataset)

    def train_model(self, model, train_dataset, test_dataset):
        for epoch in range(self.epochs):
            loss = self.train(model, train_dataset)
            print(f'Epoch {epoch+1}, Loss: {loss}')
        print(f'Test Accuracy: {self.evaluate(model, test_dataset)}')

    def main(self, dataset_path, snippet_folder_path):
        train_triplets, test_triplets = self.prepare_data(dataset_path, snippet_folder_path)
        train_tokenized_triplets = self.tokenize_triplets(train_triplets)
        test_tokenized_triplets = self.tokenize_triplets(test_triplets)
        train_dataset = DataLoader(train_tokenized_triplets, batch_size=self.batch_size, shuffle=True)
        test_dataset = DataLoader(test_tokenized_triplets, batch_size=self.batch_size, shuffle=False)
        model = self.create_model()
        model.to(self.device)
        self.train_model(model, train_dataset, test_dataset)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    model = BugTripletModel()
    model.main(dataset_path, snippet_folder_path)