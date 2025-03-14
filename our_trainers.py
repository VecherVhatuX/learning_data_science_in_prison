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
        self.embed = nn.Embedding(vocab_dim, embed_dim)  # Maps indices to dense vectors
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduces sequence length to 1
        self.proj = nn.Linear(embed_dim, embed_dim)  # Projects embeddings to same dimension
        self.bn = nn.BatchNorm1d(embed_dim)  # Normalizes across batch
        self.ln = nn.LayerNorm(embed_dim)  # Normalizes across features

    def forward(self, x):
        x = self.embed(x)  # Convert indices to embeddings
        x = self.pool(x.transpose(1, 2)).squeeze(2)  # Pool and remove extra dimension
        x = self.proj(x)  # Apply linear transformation
        x = self.bn(x)  # Normalize batch
        return self.ln(x)  # Normalize features

class TripletData(Dataset):
    def __init__(self, data, labels, neg_samples):
        self.data = data  # Input data
        self.labels = labels  # Corresponding labels
        self.neg_samples = neg_samples  # Number of negative samples per anchor

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        anchor = self.data[idx]  # Anchor sample
        label = self.labels[idx]  # Anchor label
        pos = random.choice(self.data[self.labels == label])  # Positive sample
        negs = random.sample(self.data[self.labels != label].tolist(), self.neg_samples)  # Negative samples
        return torch.tensor(anchor, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(negs, dtype=torch.long)

def triplet_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)  # Distance between anchor and positive
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2), dim=1)[0]  # Minimum distance to negatives
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))  # Triplet loss calculation

def train_model(model, dataset, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)  # Optimizer
    loss_hist = []  # Track loss over epochs

    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, pos, neg in DataLoader(dataset, batch_size=32, shuffle=True):
            opt.zero_grad()  # Reset gradients
            anchor_emb = model(anchor)  # Embed anchor
            pos_emb = model(pos)  # Embed positive
            neg_emb = model(neg)  # Embed negatives
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)  # Compute loss
            loss.backward()  # Backpropagate
            opt.step()  # Update weights
            epoch_loss += loss.item()  # Accumulate loss
        loss_hist.append(epoch_loss / len(dataset))  # Average loss per epoch
    return loss_hist

def evaluate_model(model, data, labels, k=5):
    embs = model(torch.tensor(data, dtype=torch.long)).detach().numpy()  # Get embeddings
    show_metrics(calc_metrics(embs, labels, k))  # Display metrics
    plot_embs(embs, labels)  # Visualize embeddings

def show_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")  # Print accuracy
    print(f"Precision: {metrics[1]:.4f}")  # Print precision
    print(f"Recall: {metrics[2]:.4f}")  # Print recall
    print(f"F1-score: {metrics[3]:.4f}")  # Print F1-score

def save_model(model, path):
    torch.save(model.state_dict(), path)  # Save model weights

def load_model(model_class, path):
    model = model_class()  # Initialize model
    model.load_state_dict(torch.load(path))  # Load weights
    return model

def get_embs(model, data):
    return model(torch.tensor(data, dtype=torch.long)).detach().numpy()  # Extract embeddings

def plot_embs(embs, labels):
    plt.figure(figsize=(8, 8))  # Create figure
    tsne = TSNE(n_components=2).fit_transform(embs)  # Reduce dimensions
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')  # Plot embeddings
    plt.colorbar()  # Add colorbar
    plt.show()  # Display plot

def calc_metrics(embs, labels, k=5):
    dist_mat = np.linalg.norm(embs[:, np.newaxis] - embs, axis=2)  # Compute pairwise distances
    nn = np.argsort(dist_mat, axis=1)[:, 1:k + 1]  # Find nearest neighbors
    tp = np.sum(labels[nn] == labels[:, np.newaxis], axis=1)  # True positives
    acc = np.mean(np.any(labels[nn] == labels[:, np.newaxis], axis=1))  # Accuracy
    prec = np.mean(tp / k)  # Precision
    rec = np.mean(tp / np.sum(labels == labels[:, np.newaxis], axis=1))  # Recall
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0  # F1-score
    return acc, prec, rec, f1

def plot_loss(loss_hist):
    plt.figure(figsize=(10, 5))  # Create figure
    plt.plot(loss_hist, label='Loss', color='blue')  # Plot loss
    plt.title('Training Loss Over Epochs')  # Set title
    plt.xlabel('Epochs')  # Label x-axis
    plt.ylabel('Loss')  # Label y-axis
    plt.legend()  # Add legend
    plt.show()  # Display plot

def run_training(lr, batch_size, epochs, neg_samples, vocab_dim, embed_dim, data_size):
    data, labels = generate_random_data(data_size)  # Generate random data
    dataset = TripletData(data, labels, neg_samples)  # Create dataset
    model = EmbeddingModel(vocab_dim, embed_dim)  # Initialize model
    save_model(model, "embedding_model.pth")  # Save initial model
    loss_hist = train_model(model, dataset, epochs, lr)  # Train model
    plot_loss(loss_hist)  # Plot loss
    evaluate_model(model, data, labels)  # Evaluate model

def show_model_summary(model):
    print(model)  # Print model architecture

def train_with_early_stop(model, dataset, epochs, lr, patience=5):
    opt = optim.Adam(model.parameters(), lr=lr)  # Optimizer
    loss_hist = []  # Track loss
    best_loss = float('inf')  # Track best loss
    no_improve = 0  # Track epochs without improvement

    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, pos, neg in DataLoader(dataset, batch_size=32, shuffle=True):
            opt.zero_grad()  # Reset gradients
            anchor_emb = model(anchor)  # Embed anchor
            pos_emb = model(pos)  # Embed positive
            neg_emb = model(neg)  # Embed negatives
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)  # Compute loss
            loss.backward()  # Backpropagate
            opt.step()  # Update weights
            epoch_loss += loss.item()  # Accumulate loss
        avg_loss = epoch_loss / len(dataset)  # Average loss
        loss_hist.append(avg_loss)  # Record loss

        if avg_loss < best_loss:
            best_loss = avg_loss  # Update best loss
            no_improve = 0  # Reset counter
        else:
            no_improve += 1  # Increment counter
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")  # Stop training
                break
    return loss_hist

def generate_random_data(data_size):
    data = np.random.randint(0, 100, (data_size, 10))  # Random data
    labels = np.random.randint(0, 10, data_size)  # Random labels
    return data, labels

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)  # Run training
    show_model_summary(EmbeddingModel(101, 10))  # Show model summary