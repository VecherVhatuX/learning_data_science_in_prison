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

def load_data(file_path):
    return np.load(file_path, allow_pickle=True) if file_path.endswith('.npy') else json.load(open(file_path, 'r', encoding='utf-8'))

def load_snippets(folder_path):
    return [(os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

def separate_snippets(snippets):
    return ([snippet_data['snippet'] for _, snippet_file_path in snippets for snippet_data in [load_data(snippet_file_path)] if snippet_data.get('is_bug', False)], 
            [snippet_data['snippet'] for _, snippet_file_path in snippets for snippet_data in [load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)])

def create_triplets(instance_id_map, snippets, num_negatives_per_positive):
    return [{'anchor': instance_id_map[os.path.basename(folder_path)], 'positive': positive_doc, 'negative': random.choice(separate_snippets(snippets)[1])} 
            for folder_path, _ in snippets 
            for positive_doc in separate_snippets(snippets)[0] 
            for _ in range(min(num_negatives_per_positive, len(separate_snippets(snippets)[1])))]

def prepare_data(dataset_path, snippet_folder_path, num_negatives_per_positive):
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    snippets = load_snippets(snippet_folder_path)
    return tuple(np.array_split(np.array(create_triplets(instance_id_map, snippets, num_negatives_per_positive)), 2))

def tokenize_triplets(triplets, tokenizer, max_sequence_length):
    return [{'anchor_input_ids': tokenizer.encode_plus(triplet['anchor'], max_length=max_sequence_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')['input_ids'].flatten(),
             'anchor_attention_mask': tokenizer.encode_plus(triplet['anchor'], max_length=max_sequence_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')['attention_mask'].flatten(),
             'positive_input_ids': tokenizer.encode_plus(triplet['positive'], max_length=max_sequence_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')['input_ids'].flatten(),
             'positive_attention_mask': tokenizer.encode_plus(triplet['positive'], max_length=max_sequence_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')['attention_mask'].flatten(),
             'negative_input_ids': tokenizer.encode_plus(triplet['negative'], max_length=max_sequence_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')['input_ids'].flatten(),
             'negative_attention_mask': tokenizer.encode_plus(triplet['negative'], max_length=max_sequence_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')['attention_mask'].flatten()} 
            for triplet in triplets]

class CustomDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_sequence_length):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.triplets = tokenize_triplets(triplets, tokenizer, max_sequence_length)
        self.indices = list(range(len(self.triplets)))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[self.indices[idx]]

    def on_epoch_end(self):
        random.shuffle(self.indices)

def model():
    return nn.Sequential(
        AutoModel.from_pretrained('bert-base-uncased'),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(AutoModel.from_pretrained('bert-base-uncased').config.hidden_size, FULLY_CONNECTED_SIZE),
        nn.ReLU(),
        nn.Linear(FULLY_CONNECTED_SIZE, EMBEDDING_SIZE)
    )

def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    return torch.mean((anchor_embeddings - positive_embeddings) ** 2) + torch.clamp(torch.mean((anchor_embeddings - negative_embeddings) ** 2) - torch.mean((anchor_embeddings - positive_embeddings) ** 2), min=0)

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
        anchor_embeddings = model((anchor_input_ids, anchor_attention_mask))
        positive_embeddings = model((positive_input_ids, positive_attention_mask))
        negative_embeddings = model((negative_input_ids, negative_attention_mask))
        loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

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
            anchor_embeddings = model((anchor_input_ids, anchor_attention_mask))
            positive_embeddings = model((positive_input_ids, positive_attention_mask))
            negative_embeddings = model((negative_input_ids, negative_attention_mask))
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
    return total_correct / len(loader.dataset)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    train_triplets, test_triplets = prepare_data(dataset_path, snippet_folder_path, NUM_NEGATIVES_PER_POSITIVE)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(train_triplets, tokenizer, MAX_SEQUENCE_LENGTH)
    test_dataset = CustomDataset(test_triplets, tokenizer, MAX_SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: {k: torch.stack([d[k] for d in x]) for k in x[0]})
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: {k: torch.stack([d[k] for d in x]) for k in x[0]})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_VALUE)
    for epoch in range(EPOCHS):
        train_dataset.on_epoch_end()
        loss = train(model, device, train_loader, optimizer)
        print(f'Epoch {epoch+1}, Loss: {loss}')
    print(f'Test Accuracy: {evaluate(model, device, test_loader)}')

if __name__ == "__main__":
    main()