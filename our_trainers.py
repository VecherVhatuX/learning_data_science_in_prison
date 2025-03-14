import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_size, feature_size):
        super(EmbeddingNetwork, self).__init__()
        self.embedding = nn.Embedding(embedding_size, feature_size)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(feature_size, feature_size)
        self.batch_norm = nn.BatchNorm1d(feature_size)
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(2)
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        return x

def generate_random_data(data_size):
    return (np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 2, data_size)

class TripletDataset(Dataset):
    def __init__(self, data_samples, data_labels, num_negatives):
        self.data_samples = data_samples
        self.data_labels = data_labels
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        anchor_sample = self.data_samples[idx]
        anchor_label = self.data_labels[idx]
        positive_sample = random.choice(self.data_samples[self.data_labels == anchor_label])
        negative_samples = random.sample(self.data_samples[self.data_labels != anchor_label].tolist(), self.num_negatives)
        return torch.tensor(anchor_sample, dtype=torch.long), torch.tensor(positive_sample, dtype=torch.long), torch.tensor(negative_samples, dtype=torch.long)

def compute_triplet_loss(margin=1.0):
    def loss_fn(anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - negative, dim=2), dim=1)[0]
        return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))
    return loss_fn

def train_embedding_network(network, dataset, num_epochs, lr):
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = compute_triplet_loss()
    loss_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for anchors, positives, negatives in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            anchors_emb = network(anchors)
            positives_emb = network(positives)
            negatives_emb = network(negatives)
            loss = loss_fn(anchors_emb, positives_emb, negatives_emb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(dataset))
    return loss_history

def evaluate_network(network, data_samples, data_labels, k=5):
    embeddings = network(torch.tensor(data_samples, dtype=torch.long)).detach().numpy()
    display_performance_metrics(compute_knn_metrics(embeddings, data_labels, k))
    visualize_embeddings(embeddings, data_labels)

def display_performance_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_network(network, filepath):
    torch.save(network.state_dict(), filepath)

def load_network(network_class, filepath):
    network = network_class()
    network.load_state_dict(torch.load(filepath))
    return network

def extract_embeddings(network, input_data):
    return network(torch.tensor(input_data, dtype=torch.long)).detach().numpy()

def visualize_embeddings(embeddings, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def compute_knn_metrics(embeddings, labels, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)
    predictions = knn.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1

def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training_pipeline(lr, batch_size, num_epochs, num_negatives, embedding_size, feature_size, data_size):
    data_samples, data_labels = generate_random_data(data_size)
    triplet_dataset = TripletDataset(data_samples, data_labels, num_negatives)
    network = EmbeddingNetwork(embedding_size, feature_size)
    loss_history = train_embedding_network(network, triplet_dataset, num_epochs, lr)
    save_network(network, "triplet_network.pth")
    plot_loss_history(loss_history)
    evaluate_network(network, data_samples, data_labels)

def display_network_summary(network):
    print(network)

if __name__ == "__main__":
    run_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    display_network_summary(EmbeddingNetwork(101, 10))