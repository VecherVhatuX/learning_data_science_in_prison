import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os

class CustomDataset(Dataset):
    def __init__(self, triplets, max_sequence_length, tokenizer, batch_size=32):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.triplets) // self.batch_size + 1

    def __getitem__(self, idx):
        batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs = []
        attention_masks = []
        for triplet in batch_triplets:
            anchor = self.tokenizer.encode_plus(
                triplet['anchor'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            positive = self.tokenizer.encode_plus(
                triplet['positive'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            negative = self.tokenizer.encode_plus(
                triplet['negative'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            inputs.extend([anchor['input_ids'].squeeze(0), positive['input_ids'].squeeze(0), negative['input_ids'].squeeze(0)])
            attention_masks.extend([anchor['attention_mask'].squeeze(0), positive['attention_mask'].squeeze(0), negative['attention_mask'].squeeze(0)])
        return {'input_ids': torch.stack(inputs), 'attention_mask': torch.stack(attention_masks)}

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

class TripletDataset(Dataset):
    def __init__(self, triplets, num_negatives_per_positive, batch_size):
        self.triplets = triplets
        self.num_negatives_per_positive = num_negatives_per_positive
        self.batch_size = batch_size
        self.positive_snippets = [triplet['positive'] for triplet in triplets]
        self.negative_snippets = [triplet['negative'] for triplet in triplets]
        self.anchor = [triplet['anchor'] for triplet in triplets]

    def __len__(self):
        return len(self.anchor) // self.batch_size + 1

    def __getitem__(self, idx):
        batch_anchor = self.anchor[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_positive_snippets = self.positive_snippets[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_negative_snippets = self.negative_snippets[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_triplets = []
        for i in range(len(batch_anchor)):
            for _ in range(self.num_negatives_per_positive):
                batch_triplets.append({'anchor': batch_anchor[i], 'positive': batch_positive_snippets[i], 'negative': random.choice(self.negative_snippets)})
        return batch_triplets

def train(model, device, dataset, epochs, learning_rate_value, max_sequence_length, tokenizer, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_value)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch_triplets in enumerate(dataset):
            batch_triplets = batch_triplets
            inputs = []
            attention_masks = []
            for triplet in batch_triplets:
                anchor = tokenizer.encode_plus(
                    triplet['anchor'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                positive = tokenizer.encode_plus(
                    triplet['positive'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                negative = tokenizer.encode_plus(
                    triplet['negative'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                inputs.extend([anchor['input_ids'].squeeze(0), positive['input_ids'].squeeze(0), negative['input_ids'].squeeze(0)])
                attention_masks.extend([anchor['attention_mask'].squeeze(0), positive['attention_mask'].squeeze(0), negative['attention_mask'].squeeze(0)])
            inputs = torch.stack(inputs)
            attention_masks = torch.stack(attention_masks)
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            inputs = torch.split(inputs, inputs.size(0) // 3, dim=0)
            attention_masks = torch.split(attention_masks, attention_masks.size(0) // 3, dim=0)
            optimizer.zero_grad()
            anchor_embeddings = model(inputs[0])
            positive_embeddings = model(inputs[1])
            negative_embeddings = model(inputs[2])
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataset)}')

def evaluate(model, device, dataset, max_sequence_length, tokenizer, batch_size):
    total_correct = 0
    with torch.no_grad():
        for batch_idx, batch_triplets in enumerate(dataset):
            batch_triplets = batch_triplets
            inputs = []
            attention_masks = []
            for triplet in batch_triplets:
                anchor = tokenizer.encode_plus(
                    triplet['anchor'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                positive = tokenizer.encode_plus(
                    triplet['positive'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                negative = tokenizer.encode_plus(
                    triplet['negative'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                inputs.extend([anchor['input_ids'].squeeze(0), positive['input_ids'].squeeze(0), negative['input_ids'].squeeze(0)])
                attention_masks.extend([anchor['attention_mask'].squeeze(0), positive['attention_mask'].squeeze(0), negative['attention_mask'].squeeze(0)])
            inputs = torch.stack(inputs)
            attention_masks = torch.stack(attention_masks)
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            inputs = torch.split(inputs, inputs.size(0) // 3, dim=0)
            attention_masks = torch.split(attention_masks, attention_masks.size(0) // 3, dim=0)
            anchor_embeddings = model(inputs[0])
            positive_embeddings = model(inputs[1])
            negative_embeddings = model(inputs[2])
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.dot(anchor_embeddings[i], positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.dot(anchor_embeddings[i], negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
    accuracy = total_correct / ((len(dataset) * batch_size) // 3)
    print(f'Test Accuracy: {accuracy}')

def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    positive_distance = torch.mean(torch.pow(anchor_embeddings - positive_embeddings, 2))
    negative_distance = torch.mean(torch.pow(anchor_embeddings - negative_embeddings, 2))
    return positive_distance + torch.max(negative_distance - positive_distance, torch.tensor(0.0))

def load_data(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    else:
        return json.load(open(file_path, 'r', encoding='utf-8'))

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

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    triplets = []
    for positive_doc in positive_snippets:
        for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
            triplets.append({'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)})
    return triplets

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
    train_data = TripletDataset(train_triplets, num_negatives_per_positive, batch_size)
    test_data = TripletDataset(test_triplets, num_negatives_per_positive, batch_size)
    model = Model(embedding_size, fully_connected_size, dropout_rate)
    model.to(device)
    train(model, device, train_data, epochs, learning_rate_value, max_sequence_length, tokenizer, batch_size)
    evaluate(model, device, test_data, max_sequence_length, tokenizer, batch_size)

if __name__ == "__main__":
    main()