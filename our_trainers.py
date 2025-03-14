import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class WordVectorGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(WordVectorGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(embed_size, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = self.pooling(embedded.transpose(1, 2)).squeeze(2)
        projected = self.projection(pooled)
        batch_normed = self.batch_norm(projected)
        return self.layer_norm(batch_normed)

class TripletDataLoader(Dataset):
    def __init__(self, samples, labels, num_negatives):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor = self.samples[index]
        anchor_label = self.labels[index]
        positive = random.choice(self.samples[self.labels == anchor_label])
        negatives = random.sample(self.samples[self.labels != anchor_label].tolist(), self.num_negatives)
        return torch.tensor(anchor, dtype=torch.long), torch.tensor(positive, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)

def calculate_triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - negative, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))

def train_model(model, dataset, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for anchor, positive, negative in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            anchor_vec = model(anchor)
            positive_vec = model(positive)
            negative_vec = model(negative)
            loss = calculate_triplet_loss(anchor_vec, positive_vec, negative_vec)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss / len(dataset))
    return loss_history

def evaluate_model(model, samples, labels, k=5):
    embeddings = model(torch.tensor(samples, dtype=torch.long)).detach().numpy()
    show_metrics(compute_metrics(embeddings, labels, k))
    plot_embeddings(embeddings, labels)

def show_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model

def get_embeddings(model, samples):
    return model(torch.tensor(samples, dtype=torch.long)).detach().numpy()

def plot_embeddings(embeddings, labels):
    plt.figure(figsize=(8, 8))
    tsne = TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def compute_metrics(embeddings, labels, k=5):
    dist_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]
    true_pos = np.sum(labels[nearest] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[nearest] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_pos / k)
    recall = np.mean(true_pos / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training(lr, batch_size, epochs, num_negatives, vocab_size, embed_size, data_size):
    samples, labels = generate_data(data_size)
    dataset = TripletDataLoader(samples, labels, num_negatives)
    model = WordVectorGenerator(vocab_size, embed_size)
    save_model(model, "embedder_model.pth")
    loss_history = train_model(model, dataset, epochs, lr)
    plot_loss(loss_history)
    evaluate_model(model, samples, labels)

def show_architecture(model):
    print(model)

def train_with_early_stop(model, dataset, epochs, lr, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        total_loss = 0.0
        for anchor, positive, negative in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            anchor_vec = model(anchor)
            positive_vec = model(positive)
            negative_vec = model(negative)
            loss = calculate_triplet_loss(anchor_vec, positive_vec, negative_vec)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return loss_history

def generate_data(data_size):
    samples = np.random.randint(0, 100, (data_size, 10))
    labels = np.random.randint(0, 10, data_size)
    return samples, labels

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    show_architecture(WordVectorGenerator(101, 10))