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
    return [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]

def encode_data(encoder, item):
    return {
        'anchor_seq': torch.tensor(encoder.transform([item['anchor']])[0]),
        'positive_seq': torch.tensor(encoder.transform([item['positive']])[0]),
        'negative_seq': torch.tensor(encoder.transform([item['negative']])[0])
    }

def shuffle_data(data):
    random.shuffle(data)
    return data

def create_triplets(mapping, bug_samples, non_bug_samples):
    return [
        {'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
        for dir, _ in snippet_files for bug_sample in bug_samples
    ]

def load_data(file_path, root_dir):
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def prepare_data(mapping, snippet_files):
    return create_triplets(mapping, *zip(*[json.load(open(path)) for path in snippet_files]))

class DataHandler:
    def __init__(self, data):
        self.data = data
        self.encoder = LabelEncoder().fit(collect_texts(data))
    def get_data(self):
        return self.data

class TripletData(Dataset):
    def __init__(self, data):
        self.data = DataHandler(data)
        self.samples = self.data.get_data()
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return encode_data(self.data.encoder, self.samples[idx])

class EmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingNet, self).__init__()
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

def compute_loss(anchor, positive, negative):
    return torch.mean(torch.clamp(0.2 + torch.norm(anchor - positive, dim=1) - torch.norm(anchor - negative, dim=1), min=0))

def train_net(model, train_data, valid_data, epochs):
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    history = []
    for _ in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            loss = compute_loss(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_data)
        eval_loss, accuracy = evaluate_net(model, valid_data)
        history.append((train_loss, eval_loss, accuracy))
    return history

def evaluate_net(model, data):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in data:
            anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            total_loss += compute_loss(anchor, positive, negative).item()
            correct += count_correct(anchor, positive, negative)
    return total_loss / len(data), correct / len(data.dataset)

def count_correct(anchor, positive, negative):
    return torch.sum((torch.sum(anchor * positive, dim=1) > torch.sum(anchor * negative, dim=1)).item())

def plot_results(history):
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

def save_net(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')

def load_net(model, path):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model

def visualize_embeddings_3d(model, data):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in data:
            anchor, _, _ = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
            embeddings.append(anchor.detach().numpy())
    embeddings = np.concatenate(embeddings)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c='Spectral')
    plt.title('3D Embedding Visualization')
    plt.show()

def execute_pipeline():
    dataset_path, snippets_dir = 'datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_data(dataset_path, snippets_dir)
    data = prepare_data(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = DataLoader(TripletData(train_data.tolist()), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TripletData(valid_data.tolist()), batch_size=32)
    model = EmbeddingNet(vocab_size=len(train_loader.dataset.data.encoder.classes_) + 1, embed_dim=128)
    history = train_net(model, train_loader, valid_loader, epochs=5)
    plot_results(history)
    save_net(model, 'model.pth')
    visualize_embeddings_3d(model, valid_loader)

def add_new_feature():
    print("New feature added: Enhanced visualization with 3D embeddings.")
    return

if __name__ == "__main__":
    execute_pipeline()
    add_new_feature()