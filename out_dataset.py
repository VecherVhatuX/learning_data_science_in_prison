import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os

class CustomDataset(Dataset):
    def __init__(self, triplets, max_sequence_length, tokenizer):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
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

class Model(nn.Module):
    def __init__(self, embedding_size, fully_connected_size, dropout_rate):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(30522, embedding_size)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_size, fully_connected_size)
        self.fc2 = nn.Linear(fully_connected_size, embedding_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x.permute(0, 2, 1)).squeeze(2)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    positive_distance = torch.mean(torch.pow(anchor_embeddings - positive_embeddings, 2))
    negative_distance = torch.mean(torch.pow(anchor_embeddings - negative_embeddings, 2))
    return positive_distance + torch.max(negative_distance - positive_distance, torch.tensor(0.0))

def train(model, device, dataset, epochs, learning_rate_value, max_sequence_length, tokenizer, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_value)
    for epoch in range(epochs):
        total_loss = 0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, batch in enumerate(dataloader):
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            anchor_embeddings = model(anchor_input_ids)
            positive_embeddings = model(positive_input_ids)
            negative_embeddings = model(negative_input_ids)
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

def evaluate(model, device, dataset, max_sequence_length, tokenizer, batch_size):
    total_correct = 0
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, batch in enumerate(dataloader):
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            anchor_embeddings = model(anchor_input_ids)
            positive_embeddings = model(positive_input_ids)
            negative_embeddings = model(negative_input_ids)
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.dot(anchor_embeddings[i], positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.dot(anchor_embeddings[i], negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
    accuracy = total_correct / len(dataset)
    print(f'Test Accuracy: {accuracy}')

def load_data(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    else:
        return json.load(open(file_path, 'r', encoding='utf-8'))

class NegativeSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.positive_indices = [triplet['positive'] for triplet in dataset]
        self.negative_indices = [i for i in range(len(dataset)) if i not in self.positive_indices]

    def sample(self, batch_size):
        return torch.utils.data.Subset(self.dataset, self.negative_indices[:batch_size])

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    triplets = []
    for positive_doc in positive_snippets:
        for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
            triplets.append({'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)})
    return triplets

def load_snippets(folder_path):
    snippet_paths = []
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)):
            snippet_paths.append((os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')))
    return snippet_paths

def separate_snippets(snippets):
    bug_snippets = []
    non_bug_snippets = []
    for folder_path, snippet_file_path in snippets:
        snippet_data = load_data(snippet_file_path)
        if snippet_data.get('is_bug', False):
            bug_snippets.append((snippet_data['snippet'], True))
        else:
            non_bug_snippets.append((snippet_data['snippet'], False))
    return tuple(map(list, zip(*[(bug_snippets, non_bug_snippets)])))

class Sampler:
    def __init__(self, dataset, num_negatives_per_positive):
        self.dataset = dataset
        self.num_negatives_per_positive = num_negatives_per_positive
        self.positive_snippets = [triplet['positive'] for triplet in dataset]
        self.negative_snippets = [triplet['negative'] for triplet in dataset]

    def sample(self, batch_size):
        sampled_triplets = []
        for _ in range(batch_size):
            problem_statement = random.choice(self.dataset)['anchor']
            positive_snippet = random.choice(self.positive_snippets)
            negative_snippets = random.sample(self.negative_snippets, self.num_negatives_per_positive)
            sampled_triplets.append({'anchor': problem_statement, 'positive': positive_snippet, 'negative': random.choice(negative_snippets)})
        return sampled_triplets

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    num_negatives_per_positive = 1
    embedding_size = 128
    fully_connected_size = 64
    dropout_rate = 0.2
    max_sequence_length = 512
    learning_rate_value = 1e-5
    epochs = 5
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    snippets = load_snippets(snippet_folder_path)
    triplets = []
    for folder_path, _ in snippets:
        bug_snippets, non_bug_snippets = separate_snippets([(folder_path, os.path.join(folder_path, 'snippet.json'))])
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        for bug_snippet, non_bug_snippet in [(bug, non_bug) for bug in bug_snippets for non_bug in non_bug_snippets]:
            triplets.extend(create_triplets(problem_statement, [bug_snippet], non_bug_snippets, num_negatives_per_positive))
    train_triplets, test_triplets = torch.utils.data.random_split(triplets, [int(len(triplets)*0.8), len(triplets)-int(len(triplets)*0.8)])
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
    train_dataset = CustomDataset(train_triplets, max_sequence_length, tokenizer)
    test_dataset = CustomDataset(test_triplets, max_sequence_length, tokenizer)
    train_sampler = Sampler(train_dataset, num_negatives_per_positive)
    test_sampler = Sampler(test_dataset, num_negatives_per_positive)
    model = Model(embedding_size, fully_connected_size, dropout_rate)
    model.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}:')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            anchor_embeddings = model(anchor_input_ids)
            positive_embeddings = model(positive_input_ids)
            negative_embeddings = model(negative_input_ids)
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss: {total_loss / len(train_dataloader)}')
    evaluate(model, device, test_dataset, max_sequence_length, tokenizer, batch_size)

if __name__ == "__main__":
    main()