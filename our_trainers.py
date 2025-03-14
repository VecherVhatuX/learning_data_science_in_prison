import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class EmbeddingModel(nn.Module):
    def __init__(self, embed_dim, feat_dim):
        super(EmbeddingModel, self).__init__()
        self.embed = nn.Embedding(embed_dim, feat_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.ln = nn.LayerNorm(feat_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        x = self.linear(x)
        x = self.bn(x)
        x = self.ln(x)
        return x

def create_random_data(size):
    return (np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, size)

class TripletData(Dataset):
    def __init__(self, samples, labels, neg_count):
        self.samples = samples
        self.labels = labels
        self.neg_count = neg_count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor = self.samples[idx]
        label = self.labels[idx]
        positive = random.choice(self.samples[self.labels == label])
        negatives = random.sample(self.samples[self.labels != label].tolist(), self.neg_count)
        return torch.tensor(anchor, dtype=torch.long), torch.tensor(positive, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)

def triplet_loss(margin=1.0):
    def loss(anchor, pos, neg):
        pos_dist = torch.norm(anchor - pos, dim=1)
        neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0]
        return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))
    return loss

def train_model(model, data, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = triplet_loss()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, pos, neg in DataLoader(data, batch_size=32, shuffle=True):
            opt.zero_grad()
            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)
            loss = loss_func(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data))
    return losses

def evaluate_model(model, samples, labels, k=5):
    embeds = model(torch.tensor(samples, dtype=torch.long)).detach().numpy()
    show_metrics(compute_metrics(embeds, labels, k))
    plot_embeddings(embeds, labels)

def show_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model

def get_embeddings(model, data):
    return model(torch.tensor(data, dtype=torch.long)).detach().numpy()

def plot_embeddings(embeds, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeds)[:, 0], TSNE(n_components=2).fit_transform(embeds)[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def compute_metrics(embeds, labels, k=5):
    dist_matrix = np.linalg.norm(embeds[:, np.newaxis] - embeds, axis=2)
    neighbors = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]
    tp = np.sum(labels[neighbors] == labels[:, np.newaxis], axis=1)
    acc = np.mean(np.any(labels[neighbors] == labels[:, np.newaxis], axis=1))
    prec = np.mean(tp / k)
    rec = np.mean(tp / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_pipeline(lr, batch_size, epochs, neg_count, embed_dim, feat_dim, data_size):
    samples, labels = create_random_data(data_size)
    data = TripletData(samples, labels, neg_count)
    model = EmbeddingModel(embed_dim, feat_dim)
    save_model(model, "model.pth")
    losses = train_model(model, data, epochs, lr)
    plot_loss(losses)
    evaluate_model(model, samples, labels)

def show_model_summary(model):
    print(model)

def early_stopping(model, data, epochs, lr, patience=5):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = triplet_loss()
    losses = []
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, pos, neg in DataLoader(data, batch_size=32, shuffle=True):
            opt.zero_grad()
            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)
            loss = loss_func(anchor_emb, pos_emb, neg_emb)
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

if __name__ == "__main__":
    train_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    show_model_summary(EmbeddingModel(101, 10))