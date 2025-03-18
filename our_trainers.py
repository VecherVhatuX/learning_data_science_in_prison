import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(WordEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(2)
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        return x

class TripletData(Dataset):
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

def triplet_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))

def train_model(model, loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_history = []

    for _ in range(epochs):
        epoch_loss = []
        for anchor, pos, neg in loader:
            optimizer.zero_grad()
            loss = triplet_loss(model(anchor), model(pos), model(neg)) + 0.01 * sum(torch.norm(p, p=2) for p in model.parameters())
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        scheduler.step()
        loss_history.append(np.mean(epoch_loss))
    return loss_history

def evaluate(model, data, labels, k=5):
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]
    tp = np.sum(labels[neighbors] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[neighbors] == labels[:, np.newaxis], axis=1))
    precision = np.mean(tp / k)
    recall = np.mean(tp / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def save_model_state(model, path):
    torch.save(model.state_dict(), path)

def load_model_state(model_class, path, vocab_size, embed_dim):
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

def generate_random_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

def interactive_embedding_visualization(model, data, labels):
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    tsne_result = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne_result - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

if __name__ == "__main__":
    data, labels = generate_random_data(100)
    dataset = TripletData(data, labels, 5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = WordEmbeddingModel(101, 10)
    loss_hist = train_model(model, loader, 10, 1e-4)
    save_model_state(model, "word_embedding_model.pth")
    plot_loss(loss_hist)
    evaluate(model, data, labels)
    interactive_embedding_visualization(load_model_state(WordEmbeddingModel, "word_embedding_model.pth", 101, 10), *generate_random_data(100))