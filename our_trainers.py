import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random


def create_embedder(embedding_size, feature_size):
    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_layer = nn.Embedding(embedding_size, feature_size)
            self.pooling_layer = nn.AdaptiveAvgPool1d(1)
            self.fc_layer = nn.Linear(feature_size, feature_size)
            self.batch_norm = nn.BatchNorm1d(feature_size)
            self.layer_norm = nn.LayerNorm(feature_size)

        def forward(self, x):
            x = self.embedding_layer(x)
            x = self.pooling_layer(x.transpose(1, 2)).squeeze(-1)
            x = self.fc_layer(x)
            x = self.batch_norm(x)
            x = self.layer_norm(x)
            return x

    return Embedder()


def sample_triplet(anchor, anchor_label, labels, n_negatives):
    pos_indices = np.where(labels == anchor_label.item())[0]
    positive_idx = random.choice(pos_indices.tolist())
    negative_indices = random.sample(np.where(labels != anchor_label.item())[0].tolist(), n_negatives)
    return anchor, labels[positive_idx], [labels[i] for i in negative_indices]


class TripletDataSet(torch.utils.data.Dataset):
    def __init__(self, samples, labels, n_negatives):
        self.samples = samples
        self.labels = labels
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor = self.samples[index]
        anchor_label = self.labels[index]
        return sample_triplet(anchor, anchor_label, self.labels, self.n_negatives)


def create_triplet_loss(margin=1.0):
    def loss_function(anchor, positive, negatives):
        distance_ap = torch.norm(anchor - positive, dim=1)
        distance_an = torch.norm(anchor.unsqueeze(1) - torch.stack(negatives), dim=2)
        loss_value = torch.max(distance_ap - distance_an.min(dim=1)[0] + margin, torch.tensor(0.0, device=anchor.device))
        return loss_value.mean()
    return loss_function


def train_model(embedding_model, dataset, num_epochs):
    optimizer = optim.Adam(embedding_model.parameters())
    loss_fn = create_triplet_loss()
    embedding_model.train()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        for anchors, positives, negatives in data_loader:
            optimizer.zero_grad()
            anchor_embeds = embedding_model(anchors)
            positive_embeds = embedding_model(positives)
            negative_embeds = [embedding_model(neg) for neg in negatives]
            loss_value = loss_fn(anchor_embeds, positive_embeds, negative_embeds)
            loss_value.backward()
            optimizer.step()


def validate_model(embedding_model, samples, labels, k=5):
    embedding_model.eval()
    with torch.no_grad():
        embeddings = embedding_model(torch.tensor(samples)).numpy()
        print("Validation KNN Accuracy:", knn_accuracy(embeddings, labels, k))
        print("Validation KNN Precision:", knn_precision(embeddings, labels, k))
        print("Validation KNN Recall:", knn_recall(embeddings, labels, k))
        print("Validation KNN F1-score:", knn_f1(embeddings, labels, k))
        plot_embeddings(embeddings, labels)


def generate_random_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))


def initialize_dataset(samples, labels, n_negatives):
    return TripletDataSet(samples, labels, n_negatives)


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def get_embeddings(model, samples):
    with torch.no_grad():
        return model(torch.tensor(samples)).numpy()


def plot_embeddings(embeddings, labels):
    tsne_model = TSNE(n_components=2)
    reduced_embeds = tsne_model.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeds[:, 0], reduced_embeds[:, 1], c=labels)
    plt.colorbar()
    plt.show()


def knn_accuracy(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    return np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))


def knn_precision(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    true_positive = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    return np.mean(true_positive / k)


def knn_recall(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    true_positive = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    return np.mean(true_positive / np.sum(labels == labels[:, np.newaxis], axis=1))


def knn_f1(embeddings, labels, k=5):
    precision = knn_precision(embeddings, labels, k)
    recall = knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)


def run_pipeline(learning_rate, batch_size, num_epochs, n_negatives, embedding_size, feature_size, data_size):
    samples, labels = generate_random_data(data_size)
    dataset = initialize_dataset(samples, labels, n_negatives)
    model = create_embedder(embedding_size, feature_size)
    train_model(model, dataset, num_epochs)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape((1, 10))
    output = model(input_ids)
    save_model(model, "triplet_model.pth")
    predicted_embeddings = get_embeddings(model, samples)
    validate_model(model, samples, labels)


if __name__ == "__main__":
    run_pipeline(1e-4, 32, 10, 5, 101, 10, 100)