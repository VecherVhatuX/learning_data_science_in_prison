import json
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        anchor = self.tokenizer.encode_plus(
            example['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        positive = self.tokenizer.encode_plus(
            example['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        negative = self.tokenizer.encode_plus(
            example['negative'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'anchor_input_ids': anchor['input_ids'].flatten(),
            'anchor_attention_mask': anchor['attention_mask'].flatten(),
            'positive_input_ids': positive['input_ids'].flatten(),
            'positive_attention_mask': positive['attention_mask'].flatten(),
            'negative_input_ids': negative['input_ids'].flatten(),
            'negative_attention_mask': negative['attention_mask'].flatten()
        }

def load_swebench_dataset(dataset_path):
    return np.load(dataset_path, allow_pickle=True)

def load_triplet_data(snippet_folder_path):
    return [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]

def load_snippet_file(snippet_file):
    try:
        with open(snippet_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load snippet file: {snippet_file}, error: {str(e)}")
        return []

def separate_snippets(snippets):
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [
        {'anchor': problem_statement, 'positive': positive_doc, 'negative': negative_doc}
        for positive_doc in positive_snippets
        for negative_doc in (negative_snippets if len(negative_snippets) <= num_negatives_per_positive else random.sample(negative_snippets, num_negatives_per_positive))
    ]

def create_swebench_dict(swebench_dataset):
    return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}

def create_triplet_dataset(swebench_dataset_path, snippet_folder_path, instance_id_field='instance_id', num_negatives_per_positive=3):
    swebench_dataset = load_swebench_dataset(swebench_dataset_path)
    swebench_dict = create_swebench_dict(swebench_dataset)
    triplet_data = []
    for folder in os.listdir(snippet_folder_path):
        folder_path = os.path.join(snippet_folder_path, folder)
        if os.path.isdir(folder_path):
            snippet_file = os.path.join(folder_path, 'snippet.json')
            snippets = load_snippet_file(snippet_file)
            if snippets:
                bug_snippets, non_bug_snippets = separate_snippets(snippets)
                problem_statement = swebench_dict.get(folder)
                if problem_statement:
                    triplets = create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive)
                    triplet_data.extend(triplets)
    print(f"Number of triplets: {len(triplet_data)}")
    return triplet_data

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        return torch.mean(torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1) + 1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 64)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc(pooled_output)
        pooled_output = self.relu(pooled_output)
        return pooled_output

def train(model, device, train_loader, epochs):
    model.train()
    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            anchor_output = model(anchor_input_ids, anchor_attention_mask)
            positive_output = model(positive_input_ids, positive_attention_mask)
            negative_output = model(negative_input_ids, negative_attention_mask)

            loss = criterion(anchor_output, positive_output, negative_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

def main(swebench_dataset_path, snippet_folder_path):
    dataset = create_triplet_dataset(swebench_dataset_path, snippet_folder_path)
    if not dataset:
        print("No available triplets to create the dataset.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = TripletDataset(dataset, tokenizer, max_length=512)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = Model()
    model.to(device)

    train(model, device, train_loader, epochs=5)

if __name__ == "__main__":
    swebench_dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    main(swebench_dataset_path, snippet_folder_path)