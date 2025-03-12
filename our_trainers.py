import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch.utils.data import Dataset, DataLoader


class NeuralEmbedder(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super(NeuralEmbedder, self).__init__()
        self.embedding = nn.Embedding(embedding_dim, feature_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(feature_dim, feature_dim)
        self.batch_norm = nn.BatchNorm1d(feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        x = self.dense(x)
        x = self.batch_norm(x)
        return self.layer_norm(x)


class TripletDataset(Dataset):
    def __init__(self, samples, labels, num_negatives):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.indices = np.arange(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor = self.samples[idx]
        anchor_label = self.labels[idx]
        pos_idx = random.choice(np.where(self.labels == anchor_label)[0])
        negative_indices = random.sample(np.where(self.labels != anchor_label)[0].tolist(), self.num_negatives)
        positive = self.samples[pos_idx]
        negatives = [self.samples[i] for i in negative_indices]
        return anchor, positive, negatives


def compute_triplet_loss(margin=1.0):
    def loss_fn(anchor, positive, negative_samples):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor.unsqueeze(1) - torch.tensor(negative_samples, dtype=torch.float32), dim=2)
        return torch.mean(torch.clamp(pos_dist - torch.min(neg_dist, dim=1).values + margin, min=0.0))
    return loss_fn


def train_model(model, dataset, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = compute_triplet_loss()

    for epoch in range(epochs):
        for anchors, positives, negatives in DataLoader(dataset, batch_size=32, shuffle=True):
            anchors, positives = anchors.float(), positives.float()
            optimizer.zero_grad()
            loss_value = loss_function(anchors, positives, negatives)
            loss_value.backward()
            optimizer.step()


def evaluate(model, data_samples, data_labels, k_neighbors=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(data_samples, dtype=torch.long))
    metrics = compute_knn_metrics(embeddings.numpy(), data_labels, k_neighbors)
    display_metrics(metrics)
    visualize_embeddings(embeddings.numpy(), data_labels)


def display_metrics(metrics):
    print(f"Validation KNN Accuracy: {metrics[0]}")
    print(f"Validation KNN Precision: {metrics[1]}")
    print(f"Validation KNN Recall: {metrics[2]}")
    print(f"Validation KNN F1-score: {metrics[3]}")


def create_random_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def extract_embeddings(model, data_samples):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(data_samples, dtype=torch.long)).numpy()


def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()


def compute_knn_metrics(embeddings, labels, k_neighbors=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k_neighbors + 1]
    
    true_positive_count = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    precision = np.mean(true_positive_count / k_neighbors)
    recall = np.mean(true_positive_count / np.sum(labels == labels[:, np.newaxis], axis=1))
    
    accuracy = np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def run_training_pipeline(learning_rate, batch_size, num_epochs, num_negatives, embedding_dim, feature_dim, data_size):
    samples, labels = create_random_data(data_size)
    triplet_dataset = TripletDataset(samples, labels, num_negatives)
    model = NeuralEmbedder(embedding_dim, feature_dim)
    train_model(model, triplet_dataset, num_epochs, learning_rate)
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    model(torch.tensor(input_ids, dtype=torch.long))
    save_model(model, "triplet_model.pth")
    predicted_embeddings = extract_embeddings(model, samples)
    evaluate(model, samples, labels)


def load_model(model_class, file_path):
    model = model_class()
    model.load_state_dict(torch.load(file_path))
    return model


def generate_additional_data(size):
    return np.random.randn(size, 10), np.random.randint(0, 2, size)


def plot_embedding_distribution(embeddings, labels):
    plt.figure(figsize=(10, 6))
    plt.hist(embeddings, bins=30, alpha=0.7, label='Embedding Distribution')
    plt.title('Distribution of Embeddings')
    plt.xlabel('Embedding Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)