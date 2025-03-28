import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from sklearn.cluster import KMeans

class WordVectorGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(WordVectorGenerator, self).__init__()
        self.vector_layer = nn.Embedding(vocab_size, embed_dim)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.linear_layer = nn.Linear(embed_dim, embed_dim)
        self.batch_norm_layer = nn.BatchNorm1d(embed_dim)
        self.layer_norm_layer = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.vector_layer(x)
        x = self.pooling_layer(x.transpose(1, 2)).squeeze(2)
        x = self.linear_layer(x)
        x = self.batch_norm_layer(x)
        x = self.layer_norm_layer(x)
        return x

class TripletSampler:
    def __init__(self, data, labels, neg_samples):
        self.data = data
        self.labels = labels
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        pos = random.choice(self.data[self.labels == self.labels[idx]])
        neg = random.sample(self.data[self.labels != self.labels[idx]].tolist(), self.neg_samples)
        return anchor, pos, neg

def calculate_triplet_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))

def train_model(model, data_sampler, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    loss_history = []

    for _ in range(epochs):
        batch_loss = []
        for a, p, n in data_sampler:
            optimizer.zero_grad()
            loss = calculate_triplet_loss(model(a), model(p), model(n)) + 0.01 * sum(torch.norm(param, p=2) for param in model.parameters())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        lr_scheduler.step()
        loss_history.append(np.mean(batch_loss))
    return loss_history

def evaluate_model(model, data, labels, k=5):
    embeddings = model(torch.tensor(data, dtype=torch.int32)).detach().numpy()
    distance_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
    correct = np.sum(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)

    accuracy = np.mean(np.any(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1))
    precision = np.mean(correct / k)
    recall = np.mean(correct / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    tsne = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, vocab_size, embed_dim):
    model = model_class(vocab_size, embed_dim)
    model.load_state_dict(torch.load(path))
    return model

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 10, size)

def visualize_embeddings(model, data, labels):
    embeddings = model(torch.tensor(data, dtype=torch.int32)).detach().numpy()
    tsne = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def visualize_similarity(model, data):
    embeddings = model(torch.tensor(data, dtype=torch.int32)).detach().numpy()
    cosine_similarity = np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings, axis=1))

    plt.figure(figsize=(8, 8))
    plt.imshow(cosine_similarity, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')
    plt.show()

def visualize_distribution(model, data):
    embeddings = model(torch.tensor(data, dtype=torch.int32)).detach().numpy()
    plt.figure(figsize=(8, 8))
    plt.hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Embedding Value Distribution')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.show()

def visualize_lr_schedule(optimizer, scheduler, epochs):
    lr_history = []
    for _ in range(epochs):
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, label='Learning Rate', color='red')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

def visualize_clusters(model, data, labels, n_clusters=5):
    embeddings = model(torch.tensor(data, dtype=torch.int32)).detach().numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar()
    plt.title('Embedding Clusters')
    plt.show()

def visualize_histogram(model, data):
    embeddings = model(torch.tensor(data, dtype=torch.int32)).detach().numpy()
    plt.figure(figsize=(8, 8))
    plt.hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Embedding Value Histogram')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    data, labels = generate_data(100)
    sampler = TripletSampler(data, labels, 5)
    dataset = torch.utils.data.DataLoader(sampler, batch_size=32, shuffle=True)
    model = WordVectorGenerator(101, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    loss_history = train_model(model, dataset, 10, 1e-4)
    save_model(model, "embedding_model.pth")
    plot_loss(loss_history)
    evaluate_model(model, data, labels)
    visualize_embeddings(load_model(WordVectorGenerator, "embedding_model.pth", 101, 10), *generate_data(100))
    visualize_similarity(model, data)
    visualize_distribution(model, data)
    visualize_lr_schedule(optimizer, lr_scheduler, 10)
    visualize_clusters(model, data, labels)
    visualize_histogram(model, data)