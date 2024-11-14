import os
import random
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class Config:
    INSTANCE_ID_FIELD = 'instance_id'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    NUM_NEGATIVES_PER_POSITIVE = 3
    EMBEDDING_DIM = 128
    FC_DIM = 64
    DROPOUT = 0.2
    LEARNING_RATE = 1e-5
    MAX_EPOCHS = 5

class TripletModel(nn.Module):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.embedding = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, Config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.EMBEDDING_DIM, Config.FC_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )

    def forward(self, inputs):
        anchor, positive, negative = inputs
        anchor_output = self.distilbert(anchor['input_ids'], attention_mask=anchor['attention_mask'])
        positive_output = self.distilbert(positive['input_ids'], attention_mask=positive['attention_mask'])
        negative_output = self.distilbert(negative['input_ids'], attention_mask=negative['attention_mask'])
        anchor_embedding = self.embedding(anchor_output.last_hidden_state[:, 0, :])
        positive_embedding = self.embedding(positive_output.last_hidden_state[:, 0, :])
        negative_embedding = self.embedding(negative_output.last_hidden_state[:, 0, :])
        return anchor_embedding, positive_embedding, negative_embedding

    def triplet_loss(self, anchor, positive, negative):
        return torch.mean(torch.clamp(torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1) + 1.0, min=0.0))

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        triplet = self.triplets[index]
        anchor_encoding = self.tokenizer.encode_plus(
            triplet['anchor'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        positive_encoding = self.tokenizer.encode_plus(
            triplet['positive'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        negative_encoding = self.tokenizer.encode_plus(
            triplet['negative'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'anchor': anchor_encoding,
            'positive': positive_encoding,
            'negative': negative_encoding
        }

def load_json_file(file_path: str) -> List:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

def separate_snippets(snippets: List) -> Tuple[List, List]:
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement: str, positive_snippets: List, negative_snippets: List, num_negatives_per_positive: int) -> List:
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
            for positive_doc in positive_snippets
            for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def create_triplet_dataset(dataset_path: str, snippet_folder_path: str) -> List:
    dataset = np.load(dataset_path, allow_pickle=True)
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
    bug_snippets, non_bug_snippets = zip(*[separate_snippets(snippet) for snippet in snippets])
    problem_statements = [instance_id_map.get(os.path.basename(folder_path)) for folder_path in folder_paths]
    triplets = [create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], Config.NUM_NEGATIVES_PER_POSITIVE) 
                for i, problem_statement in enumerate(problem_statements)]
    return [item for sublist in triplets for item in sublist]

def load_data(dataset_path: str, snippet_folder_path: str, tokenizer):
    triplets = create_triplet_dataset(dataset_path, snippet_folder_path)
    random.shuffle(triplets)
    train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
    train_dataset = TripletDataset(train_triplets, tokenizer)
    test_dataset = TripletDataset(test_triplets, tokenizer)
    return train_dataset, test_dataset

def train(model, dataset, optimizer):
    total_loss = 0
    for batch in DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True):
        anchor, positive, negative = batch
        anchor = {k: v.to(device) for k, v in anchor.items()}
        positive = {k: v.to(device) for k, v in positive.items()}
        negative = {k: v.to(device) for k, v in negative.items()}
        optimizer.zero_grad()
        anchor_embedding, positive_embedding, negative_embedding = model((anchor, positive, negative))
        loss = model.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)

def evaluate(model, dataset):
    total_loss = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False):
            anchor, positive, negative = batch
            anchor = {k: v.to(device) for k, v in anchor.items()}
            positive = {k: v.to(device) for k, v in positive.items()}
            negative = {k: v.to(device) for k, v in negative.items()}
            anchor_embedding, positive_embedding, negative_embedding = model((anchor, positive, negative))
            loss = model.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            total_loss += loss.item()
    return total_loss / len(dataset)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    model = TripletModel()
    model.load_state_dict(torch.load(path))
    return model

def plot_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset, test_dataset = load_data(dataset_path, snippet_folder_path, tokenizer)
    model = TripletModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    model_path = 'triplet_model.pth'
    history = {'loss': [], 'val_loss': []}
    for epoch in range(Config.MAX_EPOCHS):
        loss = train(model, train_dataset, optimizer)
        val_loss = evaluate(model, test_dataset)
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        save_model(model, model_path)
        print(f'Model saved at {model_path}')
    plot_history(history)

if __name__ == "__main__":
    main()