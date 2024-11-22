import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

# Constants
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_SIZE = 128
FULLY_CONNECTED_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE_VALUE = 1e-5
EPOCHS = 5
BATCH_SIZE = 32
NUM_NEGATIVES_PER_POSITIVE = 1

def load_data(file_path: str) -> np.ndarray or dict:
    if file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    else:
        return json.load(open(file_path, 'r', encoding='utf-8'))

def load_snippets(folder_path: str) -> list:
    return [(os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

def separate_snippets(snippets: list) -> tuple:
    bug_snippets = []
    non_bug_snippets = []
    for folder_path, snippet_file_path in snippets:
        snippet_data = load_data(snippet_file_path)
        if snippet_data.get('is_bug', False):
            bug_snippets.append((snippet_data['snippet'], True))
        else:
            non_bug_snippets.append((snippet_data['snippet'], False))
    return bug_snippets, non_bug_snippets

def create_triplets(problem_statement: str, positive_snippets: list, negative_snippets: list, num_negatives_per_positive: int) -> list:
    return [{'anchor': problem_statement, 'positive': positive_doc[0], 'negative': random.choice(negative_snippets)[0]} for positive_doc in positive_snippets for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def prepare_data(dataset_path: str, snippet_folder_path: str, num_negatives_per_positive: int) -> tuple:
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    snippets = load_snippets(snippet_folder_path)
    bug_snippets, non_bug_snippets = separate_snippets(snippets)
    triplets = []
    for folder_path, _ in snippets:
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        triplets.extend(create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive))
    train_size = int(len(triplets)*0.8)
    return triplets[:train_size], triplets[train_size:]

class CustomDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_sequence_length):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.anchor_inputs = []
        self.positive_inputs = []
        self.negative_inputs = []
        for triplet in self.triplets:
            anchor_inputs = self.tokenizer.encode_plus(
                triplet['anchor'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            positive_inputs = self.tokenizer.encode_plus(
                triplet['positive'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            negative_inputs = self.tokenizer.encode_plus(
                triplet['negative'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.anchor_inputs.append({
                'input_ids': anchor_inputs['input_ids'].flatten(),
                'attention_mask': anchor_inputs['attention_mask'].flatten()
            })
            self.positive_inputs.append({
                'input_ids': positive_inputs['input_ids'].flatten(),
                'attention_mask': positive_inputs['attention_mask'].flatten()
            })
            self.negative_inputs.append({
                'input_ids': negative_inputs['input_ids'].flatten(),
                'attention_mask': negative_inputs['attention_mask'].flatten()
            })

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return {
            'anchor_input_ids': self.anchor_inputs[idx]['input_ids'],
            'anchor_attention_mask': self.anchor_inputs[idx]['attention_mask'],
            'positive_input_ids': self.positive_inputs[idx]['input_ids'],
            'positive_attention_mask': self.positive_inputs[idx]['attention_mask'],
            'negative_input_ids': self.negative_inputs[idx]['input_ids'],
            'negative_attention_mask': self.negative_inputs[idx]['attention_mask']
        }

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(self.bert.config.hidden_size, FULLY_CONNECTED_SIZE)
        self.fc2 = nn.Linear(FULLY_CONNECTED_SIZE, EMBEDDING_SIZE)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs.pooler_output)
        outputs = torch.relu(self.fc(outputs))
        outputs = self.fc2(outputs)
        return outputs

def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    positive_distance = torch.mean((anchor_embeddings - positive_embeddings) ** 2)
    negative_distance = torch.mean((anchor_embeddings - negative_embeddings) ** 2)
    return positive_distance + torch.clamp(negative_distance - positive_distance, min=0)

def train(model, device, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch in loader:
        anchor_input_ids = batch['anchor_input_ids'].to(device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(device)
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        negative_input_ids = batch['negative_input_ids'].to(device)
        negative_attention_mask = batch['negative_attention_mask'].to(device)
        optimizer.zero_grad()
        anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
        positive_embeddings = model(positive_input_ids, positive_attention_mask)
        negative_embeddings = model(negative_input_ids, negative_attention_mask)
        loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

def evaluate(model, device, loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
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
                similarity_positive = torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
    accuracy = total_correct / len(loader.dataset)
    print(f'Test Accuracy: {accuracy}')

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    train_triplets, test_triplets = prepare_data(dataset_path, snippet_folder_path, NUM_NEGATIVES_PER_POSITIVE)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(train_triplets, tokenizer, MAX_SEQUENCE_LENGTH)
    test_dataset = CustomDataset(test_triplets, tokenizer, MAX_SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_VALUE)
    for epoch in range(EPOCHS):
        train(model, device, train_loader, optimizer, epoch)
    evaluate(model, device, test_loader)

if __name__ == "__main__":
    main()