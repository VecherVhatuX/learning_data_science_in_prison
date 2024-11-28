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
from functools import partial
from operator import itemgetter

# Data Loading
def load_data_from_json(path):
    """Loads data from a JSON file."""
    return json.load(open(path))

def fetch_snippet_folders(folder):
    """Fetches a list of snippet folders."""
    return [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
            for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

def categorize_snippets(snippets):
    """Categorizes snippets into bug and non-bug snippets."""
    bug_snippets = list(map(lambda x: [load_data_from_json(path)['snippet'] for _, path in x 
                                if load_data_from_json(path).get('is_bug', False)],
                    [snippets, snippets]))
    non_bug_snippets = list(map(lambda x: [load_data_from_json(path)['snippet'] for _, path in x 
                                if not load_data_from_json(path).get('is_bug', False)],
                    [snippets, snippets]))
    return bug_snippets[0], non_bug_snippets[0]

def generate_triplets(num_negatives, instance_id_map, snippets):
    """Generates triplets for training."""
    bug_snippets, non_bug_snippets = categorize_snippets(snippets)
    return [{'anchor': instance_id_map[os.path.basename(folder)], 
             'positive': positive_doc, 
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(num_negatives, len(non_bug_snippets)))]

# Dataset
class SnippetDataset(Dataset):
    def __init__(self, triplets, max_sequence_length):
        """Initializes the snippet dataset."""
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.triplets)

    def __getitem__(self, idx):
        """Returns a triplet from the dataset."""
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']

        encode_plus = self.tokenizer.encode_plus
        anchor_sequence = encode_plus(anchor, 
                                       max_length=self.max_sequence_length, 
                                       padding='max_length', 
                                       truncation=True, 
                                       return_tensors='pt',
                                       return_attention_mask=True)

        positive_sequence = encode_plus(positive, 
                                         max_length=self.max_sequence_length, 
                                         padding='max_length', 
                                         truncation=True, 
                                         return_tensors='pt',
                                       return_attention_mask=True)

        negative_sequence = encode_plus(negative, 
                                         max_length=self.max_sequence_length, 
                                         padding='max_length', 
                                         truncation=True, 
                                         return_tensors='pt',
                                       return_attention_mask=True)

        return {'anchor': {'input_ids': anchor_sequence['input_ids'].flatten(), 
                           'attention_mask': anchor_sequence['attention_mask'].flatten()}, 
                'positive': {'input_ids': positive_sequence['input_ids'].flatten(), 
                             'attention_mask': positive_sequence['attention_mask'].flatten()}, 
                'negative': {'input_ids': negative_sequence['input_ids'].flatten(), 
                             'attention_mask': negative_sequence['attention_mask'].flatten()}}

# Model
class TripletModel(nn.Module):
    def __init__(self, embedding_size, fully_connected_size, dropout_rate):
        """Initializes the triplet model."""
        super(TripletModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, fully_connected_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fully_connected_size, embedding_size)

    def forward(self, x):
        """Forward pass of the model."""
        anchor_input_ids = x['anchor']['input_ids']
        anchor_attention_mask = x['anchor']['attention_mask']
        positive_input_ids = x['positive']['input_ids']
        positive_attention_mask = x['positive']['attention_mask']
        negative_input_ids = x['negative']['input_ids']
        negative_attention_mask = x['negative']['attention_mask']

        anchor_output = self.bert(anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_output = self.bert(positive_input_ids, attention_mask=positive_attention_mask)
        negative_output = self.bert(negative_input_ids, attention_mask=negative_attention_mask)

        anchor_embedding = self.fc2(self.relu(self.fc1(self.dropout(anchor_output.pooler_output))))
        positive_embedding = self.fc2(self.relu(self.fc1(self.dropout(positive_output.pooler_output))))
        negative_embedding = self.fc2(self.relu(self.fc1(self.dropout(negative_output.pooler_output))))

        return anchor_embedding, positive_embedding, negative_embedding

# Training
def compute_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, device):
    """Computes the triplet loss."""
    return torch.mean((anchor_embeddings - positive_embeddings) ** 2) + torch.max(torch.mean((anchor_embeddings - negative_embeddings) ** 2) - torch.mean((anchor_embeddings - positive_embeddings) ** 2), torch.tensor(0.0).to(device))

def train_triplet_model(model, device, train_loader, optimizer, epochs):
    """Trains the triplet model."""
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = compute_triplet_loss(*model({k: v.to(device) for k, v in batch.items()}), device)
            batch_loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Batch Loss: {batch_loss.item()}')

# Evaluation
def evaluate_triplet_model(model, device, test_loader):
    """Evaluates the triplet model."""
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            anchor_embeddings, positive_embeddings, negative_embeddings = model({k: v.to(device) for k, v in batch.items()})
            for i in range(len(anchor_embeddings)):
                if torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i])) > torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i])):
                    total_correct += 1
    return total_correct / len(test_loader.dataset)

# Main
def load_data(dataset_path, snippet_folder_path):
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data_from_json(dataset_path)}
    snippets = fetch_snippet_folders(snippet_folder_path)
    return instance_id_map, snippets

def create_dataset(instance_id_map, snippets, max_sequence_length):
    triplets = generate_triplets(1, instance_id_map, snippets)
    return SnippetDataset(triplets, max_sequence_length)

def create_data_loaders(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_model(embedding_size, fully_connected_size, dropout_rate):
    return TripletModel(embedding_size, fully_connected_size, dropout_rate)

def create_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)

def train(model, device, train_loader, optimizer, epochs):
    train_triplet_model(model, device, train_loader, optimizer, epochs)

def evaluate(model, device, test_loader):
    return evaluate_triplet_model(model, device, test_loader)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    instance_id_map, snippets = load_data(dataset_path, snippet_folder_path)
    triplets = generate_triplets(1, instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    train_dataset = SnippetDataset(train_triplets, 512)
    test_dataset = SnippetDataset(test_triplets, 512)
    train_loader = create_data_loaders(train_dataset, 32, shuffle=True)
    test_loader = create_data_loaders(test_dataset, 32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(128, 64, 0.2)
    model.to(device)
    optimizer = create_optimizer(model, 1e-5)

    train(model, device, train_loader, optimizer, 5)
    print(f'Test Accuracy: {evaluate(model, device, test_loader)}')

if __name__ == "__main__":
    main()