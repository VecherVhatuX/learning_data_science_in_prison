import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class NeuralEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(NeuralEmbedder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.projection_layer = nn.Linear(embed_size, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, input_ids):
        embedded = self.embedding_layer(input_ids)
        pooled = self.pooling_layer(embedded.transpose(1, 2)).squeeze(2)
        projected = self.projection_layer(pooled)
        batch_normed = self.batch_norm(projected)
        return self.layer_norm(batch_normed)

class TripletDataset(Dataset):
    def __init__(self, samples, targets, num_negatives):
        self.samples = samples
        self.targets = targets
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_sample = self.samples[index]
        anchor_label = self.targets[index]
        positive_sample = random.choice(self.samples[self.targets == anchor_label])
        negative_samples = random.sample(self.samples[self.targets != anchor_label].tolist(), self.num_negatives)
        return torch.tensor(anchor_sample, dtype=torch.long), torch.tensor(positive_sample, dtype=torch.long), torch.tensor(negative_samples, dtype=torch.long)

def compute_triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = torch.norm(anchor - positive, dim=1)
    negative_distance = torch.min(torch.norm(anchor.unsqueeze(1) - negative, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(positive_distance - negative_distance + margin, min=0.0))

def train_embedder(model, dataset, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for anchor, positive, negative in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            loss = compute_triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss / len(dataset))
    return loss_history

def evaluate_embedder(model, samples, targets, k=5):
    embeddings = model(torch.tensor(samples, dtype=torch.long)).detach().numpy()
    display_metrics(calculate_metrics(embeddings, targets, k))
    visualize_embeddings(embeddings, targets)

def display_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_embedder(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_embedder(model_class, filepath):
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    return model

def extract_embeddings(model, samples):
    return model(torch.tensor(samples, dtype=torch.long)).detach().numpy()

def visualize_embeddings(embeddings, targets):
    plt.figure(figsize=(8, 8))
    tsne_results = TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=targets, cmap='viridis')
    plt.colorbar()
    plt.show()

def calculate_metrics(embeddings, targets, k=5):
    distance_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k + 1]
    true_positives = np.sum(targets[nearest_neighbors] == targets[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(targets[nearest_neighbors] == targets[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(targets == targets[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def execute_training(learning_rate, batch_size, num_epochs, num_negatives, vocab_size, embed_size, data_size):
    samples, targets = generate_random_samples(data_size)
    dataset = TripletDataset(samples, targets, num_negatives)
    model = NeuralEmbedder(vocab_size, embed_size)
    save_embedder(model, "embedder_model.pth")
    loss_history = train_embedder(model, dataset, num_epochs, learning_rate)
    plot_training_loss(loss_history)
    evaluate_embedder(model, samples, targets)

def display_model_architecture(model):
    print(model)

def train_with_early_stopping(model, dataset, num_epochs, learning_rate, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        for anchor, positive, negative in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            loss = compute_triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(dataset)
        loss_history.append(average_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return loss_history

def generate_random_samples(data_size):
    samples = np.random.randint(0, 100, (data_size, 10))
    targets = np.random.randint(0, 10, data_size)
    return samples, targets

if __name__ == "__main__":
    execute_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_architecture(NeuralEmbedder(101, 10))