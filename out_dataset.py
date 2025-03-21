import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def build_text_encoder(dataset):
    texts = [text for item in dataset for text in (item['anchor'], item['positive'], item['negative'])]
    encoder = LabelEncoder().fit(texts)
    return lambda text: torch.tensor(encoder.transform([text])[0], dtype=torch.long)

def build_data_handler(dataset):
    text_encoder = build_text_encoder(dataset)
    return lambda: dataset, text_encoder

def build_triplet_data_generator(dataset):
    data_handler = build_data_handler(dataset)
    return lambda idx: {
        'anchor': data_handler()[1](dataset[idx]['anchor']),
        'positive': data_handler()[1](dataset[idx]['positive']),
        'negative': data_handler()[1](dataset[idx]['negative'])
    }

class EmbeddingNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def compute_triplet_loss(anchor, positive, negative):
    return torch.mean(torch.clamp(0.2 + torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1), min=0))

def train_network(model, train_data, valid_data, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            anchor, positive, negative = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
            loss = compute_triplet_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append((total_loss / len(train_data), *evaluate_network(model, valid_data)))
    return history

def evaluate_network(model, data_loader):
    total_loss = 0
    correct = 0
    for batch in data_loader:
        anchor, positive, negative = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
        total_loss += compute_triplet_loss(anchor, positive, negative).item()
        correct += torch.sum((torch.sum(anchor * positive, dim=1) > torch.sum(anchor * negative, dim=1)).item()
    return total_loss / len(data_loader), correct / len(data_loader)

def plot_training_progress(history):
    fig = plt.figure(figsize=(10, 5))
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

def save_model_weights(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')

def load_model_weights(model, path):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model

def visualize_embeddings(model, data_loader):
    embeddings = []
    for batch in data_loader:
        anchor, _, _ = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
        embeddings.append(anchor.detach().numpy())
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.concatenate(embeddings)[:, 0], np.concatenate(embeddings)[:, 1], np.concatenate(embeddings)[:, 2], c='Spectral')
    ax.set_title('3D Embedding Visualization')
    plt.show()

def load_data(file_path, root_dir):
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def generate_triplet_data(mapping, snippet_files):
    return [{'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)} for dir, path in snippet_files for bug_sample, non_bug_samples in [json.load(open(path))]]

def execute_pipeline():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_dir = 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_data(dataset_path, snippets_dir)
    data = generate_triplet_data(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = torch.utils.data.DataLoader(train_data.tolist(), batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data.tolist(), batch_size=32)
    model = EmbeddingNetwork(vocab_size=len(train_loader.dataset[0]['anchor']) + 1, embedding_dim=128)
    history = train_network(model, train_loader, valid_loader, epochs=5)
    plot_training_progress(history)
    save_model_weights(model, 'model.pth')
    visualize_embeddings(model, valid_loader)

def add_enhanced_feature():
    print("New feature added: Enhanced visualization with 3D embeddings.")

if __name__ == "__main__":
    execute_pipeline()
    add_enhanced_feature()