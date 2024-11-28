import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import json
import os
import random

# Data Loading
class DataProcessor:
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path

    def fetch_json_data(self, path):
        return json.load(open(path))

    def gather_snippet_directories(self, folder):
        return [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
                for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    def separate_snippet_types(self, snippets):
        bug_snippets = [fetch_json_data(path)['snippet'] for _, path in snippets 
                        if fetch_json_data(path).get('is_bug', False)]
        non_bug_snippets = [fetch_json_data(path)['snippet'] for _, path in snippets 
                            if not fetch_json_data(path).get('is_bug', False)]
        return bug_snippets, non_bug_snippets

    def construct_triplets(self, num_negatives, instance_id_map, snippets):
        bug_snippets, non_bug_snippets = self.separate_snippet_types(snippets)
        return [{'anchor': instance_id_map[os.path.basename(folder)], 
                 'positive': positive_doc, 
                 'negative': random.choice(non_bug_snippets)} 
                for folder, _ in snippets 
                for positive_doc in bug_snippets 
                for _ in range(min(num_negatives, len(non_bug_snippets)))]

    def load_dataset(self):
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in self.fetch_json_data(self.dataset_path)}
        snippets = self.gather_snippet_directories(self.snippet_folder_path)
        return instance_id_map, snippets

# Dataset
class CodeSnippetDataset(Dataset):
    def __init__(self, triplets, max_sequence_length):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']

        anchor_sequence = self.tokenizer.encode_plus(anchor, 
                                                       max_length=self.max_sequence_length, 
                                                       padding='max_length', 
                                                       truncation=True, 
                                                       return_tensors='pt',
                                                       return_attention_mask=True)

        positive_sequence = self.tokenizer.encode_plus(positive, 
                                                         max_length=self.max_sequence_length, 
                                                         padding='max_length', 
                                                         truncation=True, 
                                                         return_tensors='pt',
                                                       return_attention_mask=True)

        negative_sequence = self.tokenizer.encode_plus(negative, 
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
class TripletNetwork(nn.Module):
    def __init__(self, embedding_size, fully_connected_size, dropout_rate):
        super(TripletNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, fully_connected_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fully_connected_size, embedding_size)

    def forward(self, x):
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

# Training and Evaluation
class TripletTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def calculate_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return torch.mean((anchor_embeddings - positive_embeddings) ** 2) + torch.max(torch.mean((anchor_embeddings - negative_embeddings) ** 2) - torch.mean((anchor_embeddings - positive_embeddings) ** 2), torch.tensor(0.0).to(self.device))

    def train_triplet_network(self, train_loader, optimizer, epochs):
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                batch_loss = self.calculate_triplet_loss(*self.model({k: v.to(self.device) for k, v in batch.items()}))
                batch_loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}, Batch Loss: {batch_loss.item()}')

    def evaluate_triplet_network(self, test_loader):
        total_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                anchor_embeddings, positive_embeddings, negative_embeddings = self.model({k: v.to(self.device) for k, v in batch.items()})
                for i in range(len(anchor_embeddings)):
                    if torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i])) > torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i])):
                        total_correct += 1
        return total_correct / len(test_loader.dataset)

# Main
def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    data_processor = DataProcessor(dataset_path, snippet_folder_path)
    instance_id_map, snippets = data_processor.load_dataset()
    triplets = data_processor.construct_triplets(1, instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    train_dataset = CodeSnippetDataset(train_triplets, 512)
    test_dataset = CodeSnippetDataset(test_triplets, 512)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TripletNetwork(128, 64, 0.2)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    trainer = TripletTrainer(model, device)
    trainer.train_triplet_network(train_loader, optimizer, 5)
    print(f'Test Accuracy: {trainer.evaluate_triplet_network(test_loader)}')

if __name__ == "__main__":
    main()