import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# Constants
BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 512
NUM_NEGATIVES = 1
EPOCHS = 5
LR = 1e-5

# Data Loading Functions
def load_json_data(path):
    return json.load(open(path))

def gather_snippet_directories(snippet_folder_path):
    return [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
            for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]

def load_snippet_folder(snippet_folder_path):
    snippet_directories = gather_snippet_directories(snippet_folder_path)
    return [(folder, load_json_data(snippet_file)) for folder, snippet_file in snippet_directories]

def create_instance_id_map(dataset):
    return {item['instance_id']: item['problem_statement'] for item in dataset}

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

def load_data(dataset_path, snippet_folder_path):
    dataset = load_json_data(dataset_path)
    instance_id_map = create_instance_id_map(dataset)
    snippets = load_snippet_folder(snippet_folder_path)
    return instance_id_map, snippets

def create_triplets(instance_id_map, snippets):
    return construct_triplets(NUM_NEGATIVES, instance_id_map, snippets)

def create_datasets(triplets, max_sequence_length, batch_size):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    def encode_text(text):
        return tokenizer.encode_plus(
            text,
            max_length=max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
    
    def create_dataset(triplets):
        return [{'anchor_input_ids': encode_text(triplet['anchor'])['input_ids'].flatten(),
                 'anchor_attention_mask': encode_text(triplet['anchor'])['attention_mask'].flatten(),
                 'positive_input_ids': encode_text(triplet['positive'])['input_ids'].flatten(),
                 'positive_attention_mask': encode_text(triplet['positive'])['attention_mask'].flatten(),
                 'negative_input_ids': encode_text(triplet['negative'])['input_ids'].flatten(),
                 'negative_attention_mask': encode_text(triplet['negative'])['attention_mask'].flatten()} 
                for triplet in triplets]
    
    def create_data_loader(dataset, batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    dataset = create_dataset(triplets)
    data_loader = create_data_loader(dataset, batch_size, shuffle=True)
    return data_loader

def create_model(device):
    model = nn.ModuleDict({
        'bert': AutoModel.from_pretrained('bert-base-uncased'),
        'fc1': nn.Linear(768, 128),
        'fc2': nn.Linear(128, 128)
    })
    model.to(device)
    return model

def create_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)

def calculate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    positive_distance = (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1)
    negative_distance = (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)
    return (positive_distance - negative_distance + 0.2).clamp(min=0).mean()

def train_triplet_network(model, data_loader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            anchor_input_ids = batch['anchor_input_ids']
            positive_input_ids = batch['positive_input_ids']
            negative_input_ids = batch['negative_input_ids']
            anchor_attention_mask = batch['anchor_attention_mask']
            positive_attention_mask = batch['positive_attention_mask']
            negative_attention_mask = batch['negative_attention_mask']
            
            anchor_output = model['bert'](anchor_input_ids, attention_mask=anchor_attention_mask)
            positive_output = model['bert'](positive_input_ids, attention_mask=positive_attention_mask)
            negative_output = model['bert'](negative_input_ids, attention_mask=negative_attention_mask)
            
            anchor_embedding = model['fc2'](F.relu(model['fc1'](anchor_output.pooler_output)))
            positive_embedding = model['fc2'](F.relu(model['fc1'](positive_output.pooler_output)))
            negative_embedding = model['fc2'](F.relu(model['fc1'](negative_output.pooler_output)))
            
            batch_loss = calculate_triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            batch_loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Batch Loss: {batch_loss.item()}')

def evaluate_triplet_network(model, data_loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            anchor_input_ids = batch['anchor_input_ids']
            positive_input_ids = batch['positive_input_ids']
            negative_input_ids = batch['negative_input_ids']
            anchor_attention_mask = batch['anchor_attention_mask']
            positive_attention_mask = batch['positive_attention_mask']
            negative_attention_mask = batch['negative_attention_mask']
            
            anchor_output = model['bert'](anchor_input_ids, attention_mask=anchor_attention_mask)
            positive_output = model['bert'](positive_input_ids, attention_mask=positive_attention_mask)
            negative_output = model['bert'](negative_input_ids, attention_mask=negative_attention_mask)
            
            anchor_embedding = model['fc2'](F.relu(model['fc1'](anchor_output.pooler_output)))
            positive_embedding = model['fc2'](F.relu(model['fc1'](positive_output.pooler_output)))
            negative_embedding = model['fc2'](F.relu(model['fc1'](negative_output.pooler_output)))
            
            positive_similarity = F.cosine_similarity(anchor_embedding, positive_embedding)
            negative_similarity = F.cosine_similarity(anchor_embedding, negative_embedding)
            total_correct += (positive_similarity > negative_similarity).sum().item()
    return total_correct / len(data_loader.dataset)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    
    instance_id_map, snippets = load_data(dataset_path, snippet_folder_path)
    triplets = create_triplets(instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_loader = create_datasets(train_triplets, MAX_SEQUENCE_LENGTH, BATCH_SIZE)
    test_data_loader = create_datasets(test_triplets, MAX_SEQUENCE_LENGTH, BATCH_SIZE)
    
    model = create_model(device)
    optimizer = create_optimizer(model, LR)
    train_triplet_network(model, train_data_loader, optimizer, EPOCHS, device)
    print(f'Test Accuracy: {evaluate_triplet_network(model, test_data_loader, device)}')

if __name__ == "__main__":
    main()