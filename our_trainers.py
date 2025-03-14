import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class NeuralEmbedder(nn.Module):
    def __init__(self, vocab_dim, embed_dim):
        super(NeuralEmbedder, self).__init__()
        self.embed = nn.Embedding(vocab_dim, embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.avg_pool(x.transpose(1, 2)).squeeze(2)
        x = self.proj(x)
        x = self.bn(x)
        return self.ln(x)

class TripletSampler(Dataset):
    def __init__(self, data, targets, neg_samples):
        self.data = data
        self.targets = targets
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.targets[idx]
        pos = random.choice(self.data[self.targets == anchor_label])
        negs = random.sample(self.data[self.targets != anchor_label].tolist(), self.neg_samples)
        return torch.tensor(anchor, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(negs, dtype=torch.long)

def compute_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))

def train_embedder(model, data, epochs, learning_rate):
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for a, p, n in DataLoader(data, batch_size=32, shuffle=True):
            opt.zero_grad()
            a_vec = model(a)
            p_vec = model(p)
            n_vec = model(n)
            loss = compute_loss(a_vec, p_vec, n_vec)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data))
    return losses

def assess_model(model, data, targets, k=5):
    embeds = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    display_metrics(calculate_metrics(embeds, targets, k))
    visualize_embeds(embeds, targets)

def display_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def store_model(model, path):
    torch.save(model.state_dict(), path)

def retrieve_model(model_class, path, vocab_dim, embed_dim):
    model = model_class(vocab_dim, embed_dim)
    model.load_state_dict(torch.load(path))
    return model

def extract_embeds(model, data):
    return model(torch.tensor(data, dtype=torch.long)).detach().numpy()

def visualize_embeds(embeds, targets):
    plt.figure(figsize=(8, 8))
    tsne = TSNE(n_components=2).fit_transform(embeds)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=targets, cmap='viridis')
    plt.colorbar()
    plt.show()

def calculate_metrics(embeds, targets, k=5):
    dists = np.linalg.norm(embeds[:, np.newaxis] - embeds, axis=2)
    nearest = np.argsort(dists, axis=1)[:, 1:k + 1]
    true_pos = np.sum(targets[nearest] == targets[:, np.newaxis], axis=1)
    acc = np.mean(np.any(targets[nearest] == targets[:, np.newaxis], axis=1))
    prec = np.mean(true_pos / k)
    rec = np.mean(true_pos / np.sum(targets == targets[:, np.newaxis], axis=1))
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def execute_training(lr, batch_size, epochs, neg_samples, vocab_dim, embed_dim, data_size):
    data, targets = create_data(data_size)
    dataset = TripletSampler(data, targets, neg_samples)
    model = NeuralEmbedder(vocab_dim, embed_dim)
    store_model(model, "embedder.pth")
    losses = train_embedder(model, dataset, epochs, lr)
    plot_losses(losses)
    assess_model(model, data, targets)

def display_model_arch(model):
    print(model)

def train_with_early_stopping(model, data, epochs, lr, patience=5):
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for a, p, n in DataLoader(data, batch_size=32, shuffle=True):
            opt.zero_grad()
            a_vec = model(a)
            p_vec = model(p)
            n_vec = model(n)
            loss = compute_loss(a_vec, p_vec, n_vec)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return losses

def create_data(data_size):
    data = np.random.randint(0, 100, (data_size, 10))
    targets = np.random.randint(0, 10, data_size)
    return data, targets

if __name__ == "__main__":
    execute_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_arch(NeuralEmbedder(101, 10))