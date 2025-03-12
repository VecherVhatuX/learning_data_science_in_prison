import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch.utils.data import Dataset, DataLoader


class Embedder(nn.Module):
    def __init__(self, emb_dim, feat_dim):
        super(Embedder, self).__init__()
        self.emb_layer = nn.Embedding(emb_dim, feat_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layer = nn.Linear(feat_dim, feat_dim)
        self.batch_norm = nn.BatchNorm1d(feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, input_data):
        embedded = self.emb_layer(input_data)
        pooled = self.avg_pool(embedded.transpose(1, 2)).squeeze(-1)
        dense_output = self.fc_layer(pooled)
        normalized = self.batch_norm(dense_output)
        return self.layer_norm(normalized)


class TripletData(Dataset):
    def __init__(self, data_samples, data_labels, neg_count):
        self.data_samples = data_samples
        self.data_labels = data_labels
        self.neg_count = neg_count
        self.sample_indices = np.arange(len(self.data_samples))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        anchor_sample = self.data_samples[index]
        anchor_label = self.data_labels[index]
        positive_index = random.choice(np.where(self.data_labels == anchor_label)[0])
        negative_indices = random.sample(np.where(self.data_labels != anchor_label)[0].tolist(), self.neg_count)
        positive_sample = self.data_samples[positive_index]
        negative_samples = [self.data_samples[i] for i in negative_indices]
        return anchor_sample, positive_sample, negative_samples


def triplet_loss_function(margin=1.0):
    def loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings):
        pos_distance = torch.norm(anchor_embeddings - positive_embeddings, dim=1)
        neg_distance = torch.norm(anchor_embeddings.unsqueeze(1) - torch.tensor(negative_embeddings, dtype=torch.float32), dim=2)
        return torch.mean(torch.clamp(pos_distance - torch.min(neg_distance, dim=1).values + margin, min=0.0))
    return loss_fn


def train(embedding_model, dataset, num_epochs, lr):
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=lr)
    loss_fn = triplet_loss_function()

    for epoch in range(num_epochs):
        for anchors, positives, negatives in DataLoader(dataset, batch_size=32, shuffle=True):
            anchors, positives = anchors.float(), positives.float()
            optimizer.zero_grad()
            loss_value = loss_fn(anchors, positives, negatives)
            loss_value.backward()
            optimizer.step()


def evaluate_model(embedding_model, samples, labels, k=5):
    embedding_model.eval()
    with torch.no_grad():
        embedded_output = embedding_model(torch.tensor(samples, dtype=torch.long))
    metrics = calculate_knn_metrics(embedded_output.numpy(), labels, k)
    display_metrics(metrics)
    plot_embeddings(embedded_output.numpy(), labels)


def display_metrics(metrics):
    print(f"KNN Accuracy: {metrics[0]}")
    print(f"KNN Precision: {metrics[1]}")
    print(f"KNN Recall: {metrics[2]}")
    print(f"KNN F1-score: {metrics[3]}")


def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))


def save_model_state(model_instance, path):
    torch.save(model_instance.state_dict(), path)


def get_embeddings(model_instance, input_data):
    model_instance.eval()
    with torch.no_grad():
        return model_instance(torch.tensor(input_data, dtype=torch.long)).numpy()


def plot_embeddings(embeddings, data_labels):
    tsne_model = TSNE(n_components=2)
    reduced_data = tsne_model.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data_labels, cmap='viridis')
    plt.colorbar()
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


def execute_training_pipeline(lr, batch_sz, epochs, neg_count, emb_dim, feat_dim, data_sz):
    samples, labels = generate_data(data_sz)
    triplet_data = TripletData(samples, labels, neg_count)
    model_instance = Embedder(emb_dim, feat_dim)
    train(model_instance, triplet_data, epochs, lr)
    sample_input = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    model_instance(torch.tensor(sample_input, dtype=torch.long))
    save_model_state(model_instance, "triplet_model.pth")
    predicted_embeddings = get_embeddings(model_instance, samples)
    evaluate_model(model_instance, samples, labels)


def load_model_instance(model_class, path):
    model_instance = model_class()
    model_instance.load_state_dict(torch.load(path))
    return model_instance


def create_additional_data(size):
    return np.random.randn(size, 10), np.random.randint(0, 2, size)


def plot_embedding_values_distribution(embeddings, labels):
    plt.figure(figsize=(10, 6))
    plt.hist(embeddings, bins=30, alpha=0.7, label='Embedding Values Distribution')
    plt.title('Embedding Values Distribution Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    execute_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)