import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch.utils.data import Dataset, DataLoader


class Embedder(nn.Module):
    def __init__(self, emb_dim, feat_dim):
        super().__init__()
        self.emb_layer = nn.Embedding(emb_dim, feat_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layer = nn.Linear(feat_dim, feat_dim)
        self.batch_norm = nn.BatchNorm1d(feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, input_data):
        embedded = self.emb_layer(input_data)
        pooled = self.avg_pool(embedded.transpose(1, 2)).squeeze(-1)
        normalized = self.layer_norm(self.batch_norm(self.fc_layer(pooled)))
        return normalized


class TripletData(Dataset):
    def __init__(self, samples, labels, neg_count):
        self.samples = samples
        self.labels = labels
        self.neg_count = neg_count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_sample, anchor_label = self.samples[index], self.labels[index]
        positive_index = random.choice(np.where(self.labels == anchor_label)[0])
        negative_indices = random.sample(np.where(self.labels != anchor_label)[0].tolist(), self.neg_count)
        positive_sample = self.samples[positive_index]
        negative_samples = self.samples[negative_indices]
        return anchor_sample, positive_sample, negative_samples


def triplet_loss(margin=1.0):
    def loss_fn(anchor, positive, negative):
        pos_distance = torch.norm(anchor - positive, dim=1)
        neg_distance = torch.norm(anchor.unsqueeze(1) - negative, dim=2)
        return torch.mean(torch.clamp(pos_distance - torch.min(neg_distance, dim=1).values + margin, min=0.0))
    return loss_fn


def train(model, dataset, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = triplet_loss()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for anchors, positives, negatives in DataLoader(dataset, batch_size=32, shuffle=True):
            anchors, positives = anchors.float(), positives.float()
            optimizer.zero_grad()
            loss = loss_fn(anchors, positives, negatives)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(dataset))
    return loss_history


def evaluate_model(model, samples, labels, k=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(samples, dtype=torch.long))
    metrics = calculate_knn_metrics(embeddings.numpy(), labels, k)
    display_metrics(metrics)
    plot_embeddings(embeddings.numpy(), labels)


def display_metrics(metrics):
    print(f"KNN Accuracy: {metrics[0]:.4f}")
    print(f"KNN Precision: {metrics[1]:.4f}")
    print(f"KNN Recall: {metrics[2]:.4f}")
    print(f"KNN F1-score: {metrics[3]:.4f}")


def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, size)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model


def get_embeddings(model, input_data):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(input_data, dtype=torch.long)).numpy()


def plot_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.show()


def calculate_knn_metrics(embeddings, labels, k=5):
    distance_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1:k + 1]
    
    tp_count = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    precision = np.mean(tp_count / k)
    recall = np.mean(tp_count / np.sum(labels == labels[:, np.newaxis], axis=1))
    
    accuracy = np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def execute_training_pipeline(lr, batch_size, epochs, neg_count, emb_dim, feat_dim, data_size):
    samples, labels = generate_data(data_size)
    triplet_data = TripletData(samples, labels, neg_count)
    model = Embedder(emb_dim, feat_dim)
    loss_history = train(model, triplet_data, epochs, lr)
    save_model(model, "triplet_model.pth")
    plot_training_loss(loss_history)
    evaluate_model(model, samples, labels)


def create_additional_data(size):
    return np.random.randn(size, 10), np.random.randint(0, 2, size)


def plot_embedding_distribution(embeddings):
    plt.figure(figsize=(10, 6))
    plt.hist(embeddings.flatten(), bins=30, alpha=0.7, label='Embedding Values Distribution')
    plt.title('Embedding Values Distribution Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    execute_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)