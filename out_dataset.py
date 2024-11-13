import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import json
import numpy as np

# Constants
INSTANCE_ID_FIELD = 'instance_id'
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_NEGATIVES_PER_POSITIVE = 3
EMBEDDING_DIM = 128
FC_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 1e-5
MAX_EPOCHS = 5

class TripletDataset(Dataset):
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.dataset = np.load(dataset_path, allow_pickle=True)
        self.instance_id_map = {item['instance_id']: item['problem_statement'] for item in self.dataset}
        self.folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
        self.snippets = [self.load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in self.folder_paths]
        self.bug_snippets, self.non_bug_snippets = zip(*[self.separate_snippets(snippet) for snippet in self.snippets])
        self.problem_statements = [self.instance_id_map.get(os.path.basename(folder_path)) for folder_path in self.folder_paths]
        self.triplets = [self.create_triplets(problem_statement, bug_snippets, non_bug_snippets, NUM_NEGATIVES_PER_POSITIVE) for problem_statement, bug_snippets, non_bug_snippets in zip(self.problem_statements, self.bug_snippets, self.non_bug_snippets)]
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.dataset) * NUM_NEGATIVES_PER_POSITIVE

    def __getitem__(self, index):
        folder_index = index // NUM_NEGATIVES_PER_POSITIVE
        triplet_index = index % NUM_NEGATIVES_PER_POSITIVE
        triplet = self.triplets[folder_index][triplet_index]
        anchor_encoding = self.tokenizer.encode_plus(
            triplet['anchor'],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        positive_encoding = self.tokenizer.encode_plus(
            triplet['positive'],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        negative_encoding = self.tokenizer.encode_plus(
            triplet['negative'],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_input_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_input_ids': negative_encoding['input_ids'].flatten(),
            'negative_attention_mask': negative_encoding['attention_mask'].flatten()
        }

    @staticmethod
    def load_json_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    @staticmethod
    def separate_snippets(snippets):
        return (
            [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
            [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
        )

    @staticmethod
    def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
                for positive_doc in positive_snippets
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

class TripletModel(nn.Module):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.embedding = nn.Sequential(
            nn.Linear(768, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBEDDING_DIM, FC_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

    def forward(self, anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask):
        anchor_output = self.distilbert(anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_output = self.distilbert(positive_input_ids, attention_mask=positive_attention_mask)
        negative_output = self.distilbert(negative_input_ids, attention_mask=negative_attention_mask)
        anchor_embedding = self.embedding(anchor_output.pooler_output)
        positive_embedding = self.embedding(positive_output.pooler_output)
        negative_embedding = self.embedding(negative_output.pooler_output)
        return anchor_embedding, positive_embedding, negative_embedding

    def triplet_loss(self, anchor, positive, negative):
        loss = torch.maximum(torch.zeros_like(anchor), torch.sum((anchor - positive) ** 2) - torch.sum((anchor - negative) ** 2) + 1.0)
        return loss.mean()

def train(model, device, loader, optimizer):
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
        anchor, positive, negative = model(anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask)
        loss = model.triplet_loss(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, device, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            anchor, positive, negative = model(anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask)
            loss = model.triplet_loss(anchor, positive, negative)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    dataset = TripletDataset(dataset_path, snippet_folder_path)
    train_indices, test_indices = torch.utils.data.random_split(range(len(dataset)), [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TripletModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(MAX_EPOCHS):
        loss = train(model, device, train_loader, optimizer)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
    loss = evaluate(model, device, test_loader)
    print(f'Test Loss: {loss:.4f}')

if __name__ == "__main__":
    main()