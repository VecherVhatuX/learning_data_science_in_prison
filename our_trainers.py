import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class NeuralMapper(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        x = self.linear(x)
        x = self.bn(x)
        x = self.ln(x)
        return x

class TripletSampler(Dataset):
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

def compute_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))

def train_network(model, loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_history = []

    for _ in range(epochs):
        epoch_loss = []
        for anchor, pos, neg in loader:
            optimizer.zero_grad()
            loss = compute_loss(model(anchor), model(pos), model(neg)) + 0.01 * sum(torch.norm(p, p=2) for p in model.parameters())
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        scheduler.step()
        loss_history.append(np.mean(epoch_loss))
    return loss_history

def assess_model(model, data, labels, k=5):
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]
    true_positives = np.sum(labels[neighbors] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[neighbors] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def store_model(model, path):
    torch.save(model.state_dict(), path)

def retrieve_model(model_class, path, vocab_size, embed_dim):
    model = model_class(vocab_size, embed_dim)
    model.load_state_dict(torch.load(path))
    return model

def display_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def create_random_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

def show_embeddings(model, data, labels):
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    tsne = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def display_similarity(model, data):
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    cosine_sim = np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings, axis=1))
    plt.figure(figsize=(8, 8))
    plt.imshow(cosine_sim, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')
    plt.show()

def plot_distribution(model, data):
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    plt.figure(figsize=(8, 8))
    plt.hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Embedding Value Distribution')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    data, labels = create_random_data(100)
    dataset = TripletSampler(data, labels, 5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = NeuralMapper(101, 10)
    loss_history = train_network(model, loader, 10, 1e-4)
    store_model(model, "embedding_model.pth")
    display_loss(loss_history)
    assess_model(model, data, labels)
    show_embeddings(retrieve_model(NeuralMapper, "embedding_model.pth", 101, 10), *create_random_data(100))
    display_similarity(model, data)
    plot_distribution(model, data)