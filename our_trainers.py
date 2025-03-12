import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random


class NeuralEmbedder(nn.Module):
    def __init__(self, embedding_dim, feature_dim):
        super(NeuralEmbedder, self).__init__()
        self.network = nn.Sequential(
            nn.Embedding(embedding_dim, feature_dim),
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, inputs):
        return self.network(inputs)


class CustomTripletDataset(torch.utils.data.Dataset):
    def __init__(self, data_samples, data_labels, num_negatives):
        self.data_samples = data_samples
        self.data_labels = data_labels
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        anchor = self.data_samples[idx]
        anchor_label = self.data_labels[idx]
        return self.sample_triplet(anchor, anchor_label)

    def sample_triplet(self, anchor, anchor_label):
        positive_indices = np.where(self.data_labels == anchor_label.item())[0]
        positive_sample_idx = random.choice(positive_indices.tolist())
        negative_samples = random.sample(np.where(self.data_labels != anchor_label.item())[0].tolist(), self.num_negatives)
        return anchor, self.data_labels[positive_sample_idx], [self.data_labels[i] for i in negative_samples]

    def shuffle_data(self):
        indices = np.arange(len(self.data_samples))
        np.random.shuffle(indices)
        self.data_samples = self.data_samples[indices]
        self.data_labels = self.data_labels[indices]


def build_triplet_dataset(data_samples, data_labels, num_negatives):
    return CustomTripletDataset(data_samples, data_labels, num_negatives)


def compute_triplet_loss(margin_value=1.0):
    def loss_fn(anchor, positive, negative_samples):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor.unsqueeze(1) - torch.stack(negative_samples), dim=2)
        return torch.mean(torch.clamp(pos_dist - neg_dist.min(dim=1)[0] + margin_value, min=0.0))
    return loss_fn


def train_neural_network(model, data_set, epochs):
    optimizer = optim.Adam(model.parameters())
    loss_function = compute_triplet_loss()
    model.train()
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        data_set.shuffle_data()
        for anchors, positives, negatives in data_loader:
            optimizer.zero_grad()
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = torch.stack([model(neg) for neg in negatives])
            loss_value = loss_function(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss_value.backward()
            optimizer.step()


def evaluate_model(model, data_samples, data_labels, k_neighbors=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(data_samples)).numpy()
        metrics = compute_knn_metrics(embeddings, data_labels, k_neighbors)
        display_evaluation_metrics(metrics)
        visualize_embeddings(embeddings, data_labels)


def display_evaluation_metrics(metrics):
    print(f"Validation KNN Accuracy: {metrics[0]}")
    print(f"Validation KNN Precision: {metrics[1]}")
    print(f"Validation KNN Recall: {metrics[2]}")
    print(f"Validation KNN F1-score: {metrics[3]}")


def create_random_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def extract_model_embeddings(model, data_samples):
    with torch.no_grad():
        return model(torch.tensor(data_samples)).numpy()


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


def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def run_training_pipeline(learning_rate, batch_size, num_epochs, num_negatives, embedding_dim, feature_dim, data_size):
    samples, labels = create_random_data(data_size)
    triplet_dataset = build_triplet_dataset(samples, labels, num_negatives)
    model = NeuralEmbedder(embedding_dim, feature_dim)
    train_neural_network(model, triplet_dataset, num_epochs)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape((1, 10))
    model(input_ids)
    save_model(model, "triplet_model.pth")
    predicted_embeddings = extract_model_embeddings(model, samples)
    evaluate_model(model, samples, labels)

def load_model(file_path, embedding_dim, feature_dim):
    model = NeuralEmbedder(embedding_dim, feature_dim)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

if __name__ == "__main__":
    run_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)