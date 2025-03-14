import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def collect_texts(data):
    return [text for entry in data for text in (entry['anchor'], entry['positive'], entry['negative'])]

def tokenize_sequences(tokenizer, entry):
    return {
        'anchor_seq': torch.tensor(tokenizer.transform([entry['anchor']])[0],
        'positive_seq': torch.tensor(tokenizer.transform([entry['positive']])[0]),
        'negative_seq': torch.tensor(tokenizer.transform([entry['negative']])[0])
    }

def randomize_samples(samples):
    random.shuffle(samples)
    return samples

def create_triplets(mapping, bug_samples, non_bug_samples):
    return [
        {
            'anchor': mapping[os.path.basename(folder)],
            'positive': bug_sample,
            'negative': random.choice(non_bug_samples)
        }
        for folder, _ in snippet_files
        for bug_sample in bug_samples
    ]

def load_data(file_path, root_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)
    mapping = {entry['instance_id']: entry['problem_statement'] for entry in data}
    snippet_files = [(folder, os.path.join(root_dir, 'snippet.json')) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    return mapping, snippet_files

def prepare_data(mapping, snippet_files):
    return create_triplets(mapping, *zip(*[json.load(open(path)) for path in snippet_files]))

class DataHandler:
    def __init__(self, data):
        self.data = data
        self.tokenizer = LabelEncoder().fit(collect_texts(data))

    def get_data(self):
        return self.data

class TripletData(Dataset):
    def __init__(self, data):
        self.data = DataHandler(data)
        self.samples = self.data.get_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return tokenize_sequences(self.data.tokenizer, self.samples[index])

class EmbeddingNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
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

def compute_loss(anchor, positive, negative):
    return torch.mean(torch.clamp(0.2 + torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1), min=0))

def train_network(model, train_data, valid_data, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    history = []
    for _ in range(epochs):
        for batch in train_data:
            optimizer.zero_grad()
            anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            loss = compute_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            history.append((loss.item(), *evaluate_network(model, valid_data)))
    return history

def evaluate_network(model, data):
    loss = sum(compute_loss(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])).item() for batch in data)
    correct = sum(count_correct(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])) for batch in data)
    return loss / len(data), correct / len(data.dataset)

def count_correct(anchor, positive, negative):
    return torch.sum((torch.sum(anchor * positive, dim=1) > torch.sum(anchor * negative, dim=1)).item()

def plot_training(history):
    train_loss, val_loss, train_acc = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def save_network(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')

def load_network(model, path):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model

def visualize_embeddings(model, data):
    embeddings = []
    labels = []
    for batch in data:
        anchor, _, _ = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        embeddings.extend(anchor.detach().numpy())
        labels.extend(batch['anchor_seq'].numpy())
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Spectral')
    plt.colorbar()
    plt.title('2D Embedding Visualization')
    plt.show()

def execute():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_directory = 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_data(dataset_path, snippets_directory)
    data = prepare_data(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = DataLoader(TripletData(train_data.tolist()), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TripletData(valid_data.tolist()), batch_size=32)
    model = EmbeddingNetwork(vocab_size=len(train_loader.dataset.data.tokenizer.classes_) + 1, embedding_dim=128)
    history = train_network(model, train_loader, valid_loader, epochs=5)
    plot_training(history)
    save_network(model, 'model.pth')
    visualize_embeddings(model, valid_loader)

if __name__ == "__main__":
    execute()