import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_sequence_length):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor_input_ids = self.tokenizer.encode_plus(
            triplet['anchor'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        positive_input_ids = self.tokenizer.encode_plus(
            triplet['positive'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        negative_input_ids = self.tokenizer.encode_plus(
            triplet['negative'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'anchor_input_ids': anchor_input_ids['input_ids'].flatten(),
            'anchor_attention_mask': anchor_input_ids['attention_mask'].flatten(),
            'positive_input_ids': positive_input_ids['input_ids'].flatten(),
            'positive_attention_mask': positive_input_ids['attention_mask'].flatten(),
            'negative_input_ids': negative_input_ids['input_ids'].flatten(),
            'negative_attention_mask': negative_input_ids['attention_mask'].flatten(),
        }

class TripletModel(nn.Module):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        return self.fc2(F.relu(self.fc1(bert_output.pooler_output)))

class TripletNetwork:
    def __init__(self, device):
        self.device = device
        self.model = TripletModel().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def calculate_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1)
        negative_distance = (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)
        return (positive_distance - negative_distance + 0.2).clamp(min=0).mean()

    def train(self, data_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch in data_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                self.optimizer.zero_grad()
                anchor_input_ids = batch['anchor_input_ids']
                positive_input_ids = batch['positive_input_ids']
                negative_input_ids = batch['negative_input_ids']
                anchor_attention_mask = batch['anchor_attention_mask']
                positive_attention_mask = batch['positive_attention_mask']
                negative_attention_mask = batch['negative_attention_mask']
                
                anchor_output = self.model(anchor_input_ids, anchor_attention_mask)
                positive_output = self.model(positive_input_ids, positive_attention_mask)
                negative_output = self.model(negative_input_ids, negative_attention_mask)
                
                batch_loss = self.calculate_triplet_loss(anchor_output, positive_output, negative_output)
                batch_loss.backward()
                self.optimizer.step()
                print(f'Epoch {epoch+1}, Batch Loss: {batch_loss.item()}')

    def evaluate(self, data_loader):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                anchor_input_ids = batch['anchor_input_ids']
                positive_input_ids = batch['positive_input_ids']
                negative_input_ids = batch['negative_input_ids']
                anchor_attention_mask = batch['anchor_attention_mask']
                positive_attention_mask = batch['positive_attention_mask']
                negative_attention_mask = batch['negative_attention_mask']
                
                anchor_output = self.model(anchor_input_ids, anchor_attention_mask)
                positive_output = self.model(positive_input_ids, positive_attention_mask)
                negative_output = self.model(negative_input_ids, negative_attention_mask)
                
                positive_similarity = F.cosine_similarity(anchor_output, positive_output)
                negative_similarity = F.cosine_similarity(anchor_output, negative_output)
                total_correct += (positive_similarity > negative_similarity).sum().item()
        return total_correct / len(data_loader.dataset)

def load_data(dataset_path, snippet_folder_path):
    dataset = json.load(open(dataset_path))
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    snippet_directories = [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
                           for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [(folder, json.load(open(snippet_file))) for folder, snippet_file in snippet_directories]
    return instance_id_map, snippets

def create_triplets(instance_id_map, snippets):
    bug_snippets = []
    non_bug_snippets = []
    for _, snippet_file in snippets:
        snippet_data = snippet_file
        if snippet_data.get('is_bug', False):
            bug_snippets.append(snippet_data['snippet'])
        else:
            non_bug_snippets.append(snippet_data['snippet'])
    bug_snippets = [snippet for snippet in bug_snippets if snippet]
    non_bug_snippets = [snippet for snippet in non_bug_snippets if snippet]
    return [{'anchor': instance_id_map[os.path.basename(folder)], 
             'positive': positive_doc, 
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(1, len(non_bug_snippets)))]

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    
    instance_id_map, snippets = load_data(dataset_path, snippet_folder_path)
    triplets = create_triplets(instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = TripletDataset(train_triplets, tokenizer, 512)
    test_dataset = TripletDataset(test_triplets, tokenizer, 512)
    
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    network = TripletNetwork(device)
    network.train(train_data_loader, 5)
    print(f'Test Accuracy: {network.evaluate(test_data_loader)}')

if __name__ == "__main__":
    main()