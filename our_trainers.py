import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

class TripletModel(nn.Module):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(embedding_dim, num_features)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_features, num_features)
        self.bn = nn.BatchNorm1d(num_features)
        self.ln = nn.LayerNorm(num_features)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.ln(x)
        return x

def get_triplet(anchor, anchor_label, labels, num_negatives):
    positive_idx = random.choice(np.where(labels == anchor_label.item())[0].tolist())
    negative_idx = random.sample(np.where(labels != anchor_label.item())[0].tolist(), num_negatives)
    positive = labels[positive_idx]
    negatives = [labels[i] for i in negative_idx]
    return anchor, positive, negatives

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, num_negatives):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor = self.samples[idx]
        anchor_label = self.labels[idx]
        return get_triplet(anchor, anchor_label, self.labels, self.num_negatives)

def triplet_loss(margin=1.0):
    def loss(anchor, positive, negatives):
        d_ap = torch.norm(anchor - positive, dim=1)
        d_an = torch.norm(anchor.unsqueeze(1) - torch.stack(negatives), dim=2)
        loss = torch.max(d_ap - d_an.min(dim=1)[0] + margin, torch.tensor(0.0, device=anchor.device))
        return loss.mean()
    return loss

def train_model(model, dataset, epochs):
    optimizer = optim.Adam(model.parameters())
    criterion = triplet_loss()
    model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for anchors, positives, negatives in dataloader:
            optimizer.zero_grad()
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = [model(neg) for neg in negatives]
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

def validate_model(model, samples, labels, k=5):
    model.eval()
    with torch.no_grad():
        predicted_embeddings = model(torch.tensor(samples)).numpy()
        print("Validation KNN Accuracy:", knn_accuracy(predicted_embeddings, labels, k))
        print("Validation KNN Precision:", knn_precision(predicted_embeddings, labels, k))
        print("Validation KNN Recall:", knn_recall(predicted_embeddings, labels, k))
        print("Validation KNN F1-score:", knn_f1(predicted_embeddings, labels, k))
        visualize_embeddings(predicted_embeddings, labels)

def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))

def create_dataset(samples, labels, num_negatives):
    return TripletDataset(samples, labels, num_negatives)

def create_model(embedding_dim, num_features):
    return TripletModel(embedding_dim, num_features)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def get_predicted_embeddings(model, samples):
    with torch.no_grad():
        return model(torch.tensor(samples)).numpy()

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
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

def run_pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features, size):
    samples, labels = generate_data(size)
    dataset = create_dataset(samples, labels, num_negatives)
    model = create_model(embedding_dim, num_features)
    train_model(model, dataset, epochs)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape((1, 10))
    output = model(input_ids)
    save_model(model, "triplet_model.pth")
    predicted_embeddings = get_predicted_embeddings(model, samples)
    validate_model(model, samples, labels)

if __name__ == "__main__":
    run_pipeline(1e-4, 32, 10, 5, 101, 10, 100)