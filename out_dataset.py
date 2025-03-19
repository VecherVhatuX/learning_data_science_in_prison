import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TextProcessor:
    def __init__(self, data):
        self.encoder = LabelEncoder().fit(self._extract_texts(data))
    
    def _extract_texts(self, data):
        return [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]
    
    def encode_text(self, text):
        return torch.tensor(self.encoder.transform([text])[0])

class TripletDatasetManager:
    def __init__(self, data):
        self.data = data
        self.text_processor = TextProcessor(data)
    
    def get_dataset(self):
        return self.data

class TripletDataset(Dataset):
    def __init__(self, data):
        self.dataset_manager = TripletDatasetManager(data)
        self.samples = self.dataset_manager.get_dataset()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'anchor_seq': self.dataset_manager.text_processor.encode_text(item['anchor']),
            'positive_seq': self.dataset_manager.text_processor.encode_text(item['positive']),
            'negative_seq': self.dataset_manager.text_processor.encode_text(item['negative'])
        }

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
    
    def forward(self, anchor, positive, negative):
        return (
            self.network(self.embedding(anchor)),
            self.network(self.embedding(positive)),
            self.network(self.embedding(negative))
        )

def calculate_triplet_loss(anchor, positive, negative):
    return torch.mean(torch.clamp(0.2 + torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1), min=0))

def train_model(model, train_loader, valid_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    history = []
    
    for _ in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            loss = calculate_triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        eval_loss, accuracy = evaluate_model(model, valid_loader)
        history.append((train_loss, eval_loss, accuracy))
    return history

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            total_loss += calculate_triplet_loss(anchor, positive, negative).item()
            correct += (torch.sum(anchor * positive, dim=1) > torch.sum(anchor * negative, dim=1)).float().sum()
    return total_loss / len(data_loader), correct / len(data_loader)

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in history], label='Training Loss')
    plt.plot([x[1] for x in history], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([x[2] for x in history], label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def save_trained_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')

def load_trained_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model

def visualize_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in data_loader:
            anchor, _, _ = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            embeddings.append(anchor.numpy())
    embeddings = np.concatenate(embeddings)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c='Spectral')
    plt.title('3D Embedding Visualization')
    plt.show()

def load_and_prepare_data(file_path, root_dir):
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def generate_triplets(mapping, snippet_files):
    return [{'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
            for dir, _ in snippet_files for bug_sample, non_bug_samples in [json.load(open(path)) for path in snippet_files]]

def run_pipeline():
    dataset_path, snippets_dir = 'datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_and_prepare_data(dataset_path, snippets_dir)
    data = generate_triplets(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = DataLoader(TripletDataset(train_data.tolist()), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TripletDataset(valid_data.tolist()), batch_size=32)
    model = EmbeddingModel(vocab_size=len(train_loader.dataset.dataset_manager.text_processor.encoder.classes_) + 1, embed_dim=128)
    history = train_model(model, train_loader, valid_loader, epochs=5)
    plot_training_history(history)
    save_trained_model(model, 'model.pth')
    visualize_embeddings(model, valid_loader)

def add_enhanced_feature():
    print("New feature added: Enhanced visualization with 3D embeddings.")
    return

if __name__ == "__main__":
    run_pipeline()
    add_enhanced_feature()