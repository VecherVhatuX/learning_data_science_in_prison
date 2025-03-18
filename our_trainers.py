import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        x = self.fc(x)
        x = self.bn(x)
        x = self.ln(x)
        return x

class TripletDataset(Dataset):
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

def compute_triplet_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))

def train(model, loader, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    losses = []

    for _ in range(epochs):
        epoch_loss = []
        for anchor, pos, neg in loader:
            opt.zero_grad()
            loss = compute_triplet_loss(model(anchor), model(pos), model(neg)) + 0.01 * sum(torch.norm(p, p=2) for p in model.parameters())
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
        scheduler.step()
        losses.append(np.mean(epoch_loss))
    return losses

def evaluate_model(model, data, labels, k=5):
    embeds = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    dists = np.linalg.norm(embeds[:, np.newaxis] - embeds, axis=2)
    neighs = np.argsort(dists, axis=1)[:, 1:k+1]
    tp = np.sum(labels[neighs] == labels[:, np.newaxis], axis=1)
    acc = np.mean(np.any(labels[neighs] == labels[:, np.newaxis], axis=1))
    prec = np.mean(tp / k)
    rec = np.mean(tp / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeds)[:, 0], TSNE(n_components=2).fit_transform(embeds)[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, vocab_size, embed_dim):
    model = model_class(vocab_size, embed_dim)
    model.load_state_dict(torch.load(path))
    return model

def plot_loss_history(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def generate_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

def visualize_embeddings(model, data, labels):
    embeds = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    tsne = TSNE(n_components=2).fit_transform(embeds)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def add_cosine_similarity(model, data):
    embeds = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    cos_sim = np.dot(embeds, embeds.T) / (np.linalg.norm(embeds, axis=1)[:, np.newaxis] * np.linalg.norm(embeds, axis=1))
    plt.figure(figsize=(8, 8))
    plt.imshow(cos_sim, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')
    plt.show()

if __name__ == "__main__":
    data, labels = generate_data(100)
    dataset = TripletDataset(data, labels, 5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = EmbeddingModel(101, 10)
    loss_hist = train(model, loader, 10, 1e-4)
    save_model(model, "embedding_model.pth")
    plot_loss_history(loss_hist)
    evaluate_model(model, data, labels)
    visualize_embeddings(load_model(EmbeddingModel, "embedding_model.pth", 101, 10), *generate_data(100))
    add_cosine_similarity(model, data)