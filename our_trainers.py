import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_dim, embed_dim):
        super(EmbeddingModel, self).__init__()
        self.embed = nn.Embedding(vocab_dim, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        x = self.proj(x)
        x = self.bn(x)
        return self.ln(x)

class TripletData(Dataset):
    def __init__(self, data, labels, neg_samples):
        self.data = data
        self.labels = labels
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        label = self.labels[idx]
        pos = random.choice(self.data[self.labels == label])
        negs = random.sample(self.data[self.labels != label].tolist(), self.neg_samples)
        return torch.tensor(anchor, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(negs, dtype=torch.long)

def triplet_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))

def train_model(model, dataset, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_hist = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, pos, neg in DataLoader(dataset, batch_size=32, shuffle=True):
            opt.zero_grad()
            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        loss_hist.append(epoch_loss / len(dataset))
    return loss_hist

def evaluate_model(model, data, labels, k=5):
    embs = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    show_metrics(calc_metrics(embs, labels, k))
    plot_embs(embs, labels)

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

def get_embs(model, data):
    return model(torch.tensor(data, dtype=torch.long)).detach().numpy()

def plot_embs(embs, labels):
    plt.figure(figsize=(8, 8))
    tsne = TSNE(n_components=2).fit_transform(embs)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def calc_metrics(embs, labels, k=5):
    dist_mat = np.linalg.norm(embs[:, np.newaxis] - embs, axis=2)
    nn = np.argsort(dist_mat, axis=1)[:, 1:k + 1]
    tp = np.sum(labels[nn] == labels[:, np.newaxis], axis=1)
    acc = np.mean(np.any(labels[nn] == labels[:, np.newaxis], axis=1))
    prec = np.mean(tp / k)
    rec = np.mean(tp / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1

def plot_loss(loss_hist):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training(lr, batch_size, epochs, neg_samples, vocab_dim, embed_dim, data_size):
    data, labels = generate_random_data(data_size)
    dataset = TripletData(data, labels, neg_samples)
    model = EmbeddingModel(vocab_dim, embed_dim)
    save_model(model, "embedding_model.pth")
    loss_hist = train_model(model, dataset, epochs, lr)
    plot_loss(loss_hist)
    evaluate_model(model, data, labels)

def show_model_summary(model):
    print(model)

def train_with_early_stop(model, dataset, epochs, lr, patience=5):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_hist = []
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, pos, neg in DataLoader(dataset, batch_size=32, shuffle=True):
            opt.zero_grad()
            anchor_emb = model(anchor)
            pos_emb = model(pos)
            neg_emb = model(neg)
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataset)
        loss_hist.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return loss_hist

def generate_random_data(data_size):
    data = np.random.randint(0, 100, (data_size, 10))
    labels = np.random.randint(0, 10, data_size)
    return data, labels

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    show_model_summary(EmbeddingModel(101, 10))