import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os
from transformers import BertTokenizer

# Data loading functions
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
    return bug_snippets, non_bug_snippets

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    triplets = []
    for positive_doc in positive_snippets:
        for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
            triplets.append({'anchor': problem_statement, 'positive': positive_doc[0], 'negative': random.choice(negative_snippets)[0]})
    return triplets

# Data preparation function
def prepare_data(dataset_path, snippet_folder_path, num_negatives_per_positive):
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    snippets = load_snippets(snippet_folder_path)
    bug_snippets, non_bug_snippets = separate_snippets(snippets)
    triplets = []
    for folder_path, _ in snippets:
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        triplets.extend(create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive))
    return torch.utils.data.random_split(triplets, [int(len(triplets)*0.8), len(triplets)-int(len(triplets)*0.8)])

# Dataset class
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

# Model class
class Model(nn.Module):
    def __init__(self, embedding_size, fully_connected_size, dropout_rate):
        super(Model, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, fully_connected_size)
        self.fc2 = nn.Linear(fully_connected_size, embedding_size)

    def forward(self, x, attention_mask):
        outputs = self.model(x, attention_mask=attention_mask)
        outputs = outputs.pooler_output
        outputs = self.dropout(outputs)
        outputs = torch.relu(self.fc1(outputs))
        outputs = self.fc2(outputs)
        return outputs

# Loss calculation function
def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    positive_distance = torch.mean(torch.pow(anchor_embeddings - positive_embeddings, 2))
    negative_distance = torch.mean(torch.pow(anchor_embeddings - negative_embeddings, 2))
    return positive_distance + torch.max(negative_distance - positive_distance, torch.tensor(0.0))

# Training function
def train(model, device, dataset, epochs, learning_rate_value, batch_size, optimizer):
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
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

# Evaluation function
def evaluate(model, device, dataset, batch_size):
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

# Model creation function
def create_model(embedding_size, fully_connected_size, dropout_rate, device):
    model = Model(embedding_size, fully_connected_size, dropout_rate)
    model.to(device)
    return model

# Optimizer creation function
def create_optimizer(model, learning_rate_value):
    return optim.Adam(model.parameters(), lr=learning_rate_value)

# Main function
def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    num_negatives_per_positive = 1
    fully_connected_size = 64
    dropout_rate = 0.2
    max_sequence_length = 512
    learning_rate_value = 1e-5
    epochs = 5
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_triplets, test_triplets = prepare_data(dataset_path, snippet_folder_path, num_negatives_per_positive)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(train_triplets, max_sequence_length, tokenizer)
    test_dataset = CustomDataset(test_triplets, max_sequence_length, tokenizer)
    model = create_model(128, fully_connected_size, dropout_rate, device)
    optimizer = create_optimizer(model, learning_rate_value)
    train(model, device, train_dataset, epochs, learning_rate_value, batch_size, optimizer)
    evaluate(model, device, test_dataset, batch_size)

if __name__ == "__main__":
    main()