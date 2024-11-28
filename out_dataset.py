import json
import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

def load_json_data(path):
    return json.load(open(path))

def fetch_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def fetch_directories(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

def load_dataset(dataset_path):
    return load_json_data(dataset_path)

def gather_snippet_directories(snippet_folder_path):
    return [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
            for f in fetch_directories(snippet_folder_path)]

def separate_files_by_type(snippet_files):
    bug_files = [path for path in snippet_files if load_json_data(path).get('is_bug', False)]
    non_bug_files = [path for path in snippet_files if not load_json_data(path).get('is_bug', False)]
    return bug_files, non_bug_files

def fetch_snippet_file_data(snippet_files):
    return [load_json_data(path)['snippet'] for path in snippet_files]

def separate_snippet_types(snippets):
    bug_snippets = []
    non_bug_snippets = []
    for _, snippet_file in snippets:
        snippet_data = load_json_data(snippet_file)
        if snippet_data.get('is_bug', False):
            bug_snippets.append(snippet_data['snippet'])
        else:
            non_bug_snippets.append(snippet_data['snippet'])
    bug_snippets = [snippet for snippet in bug_snippets if snippet]
    non_bug_snippets = [snippet for snippet in non_bug_snippets if snippet]
    return bug_snippets, non_bug_snippets

def construct_triplets(num_negatives, instance_id_map, snippets):
    bug_snippets, non_bug_snippets = separate_snippet_types(snippets)
    return [{'anchor': instance_id_map[os.path.basename(folder)], 
             'positive': positive_doc, 
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(num_negatives, len(non_bug_snippets)))]

def create_instance_id_map(dataset):
    return {item['instance_id']: item['problem_statement'] for item in dataset}

def load_snippet_folder(snippet_folder_path):
    snippet_directories = gather_snippet_directories(snippet_folder_path)
    return [(folder, load_json_data(snippet_file)) for folder, snippet_file in snippet_directories]

def load_data(dataset_path, snippet_folder_path):
    dataset = load_dataset(dataset_path)
    instance_id_map = create_instance_id_map(dataset)
    snippets = load_snippet_folder(snippet_folder_path)
    return instance_id_map, snippets

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

        anchor_encoding = self.tokenizer.encode_plus(
            anchor,
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        positive_encoding = self.tokenizer.encode_plus(
            positive,
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        negative_encoding = self.tokenizer.encode_plus(
            negative,
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'anchor_input_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_input_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_input_ids': negative_encoding['input_ids'].flatten(),
            'negative_attention_mask': negative_encoding['attention_mask'].flatten(),
        }

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        anchor_input_ids = x['anchor_input_ids']
        positive_input_ids = x['positive_input_ids']
        negative_input_ids = x['negative_input_ids']
        anchor_attention_mask = x['anchor_attention_mask']
        positive_attention_mask = x['positive_attention_mask']
        negative_attention_mask = x['negative_attention_mask']

        anchor_output = self.bert(anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_output = self.bert(positive_input_ids, attention_mask=positive_attention_mask)
        negative_output = self.bert(negative_input_ids, attention_mask=negative_attention_mask)

        anchor_embedding = self.fc2(torch.relu(self.fc1(anchor_output.pooler_output)))
        positive_embedding = self.fc2(torch.relu(self.fc1(positive_output.pooler_output)))
        negative_embedding = self.fc2(torch.relu(self.fc1(negative_output.pooler_output)))

        return anchor_embedding, positive_embedding, negative_embedding

class TripletTrainer:
    def __init__(self, model):
        self.model = model

    def calculate_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1)
        negative_distance = (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)
        return (positive_distance - negative_distance + 0.2).clamp(min=0).mean()

    def train_triplet_network(self, train_loader, optimizer, epochs):
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                anchor_embeddings, positive_embeddings, negative_embeddings = self.model(batch)
                batch_loss = self.calculate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                batch_loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}, Batch Loss: {batch_loss.item()}')

    def evaluate_triplet_network(self, test_loader):
        total_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                anchor_embeddings, positive_embeddings, negative_embeddings = self.model(batch)
                positive_similarity = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)
                negative_similarity = torch.nn.functional.cosine_similarity(anchor_embeddings, negative_embeddings)
                total_correct += (positive_similarity > negative_similarity).sum().item()
        return total_correct / len(test_loader.dataset)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    instance_id_map, snippets = load_data(dataset_path, snippet_folder_path)
    triplets = construct_triplets(1, instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)

    train_dataset = CodeSnippetDataset(train_triplets, 512)
    test_dataset = CodeSnippetDataset(test_triplets, 512)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TripletNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    trainer = TripletTrainer(model)
    trainer.train_triplet_network(train_loader, optimizer, 5)
    print(f'Test Accuracy: {trainer.evaluate_triplet_network(test_loader)}')

if __name__ == "__main__":
    main()