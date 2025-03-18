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
        self.embed = nn.Embedding(vocab_size, embed_dim)  # Maps indices to dense vectors
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduces sequence length to 1
        self.fc = nn.Linear(embed_dim, embed_dim)  # Linear transformation
        self.bn = nn.BatchNorm1d(embed_dim)  # Normalizes across the batch
        self.ln = nn.LayerNorm(embed_dim)  # Normalizes across features

    def forward(self, x):
        x = self.embed(x)  # Convert indices to embeddings
        x = self.pool(x.transpose(1, 2)).squeeze(2)  # Pool and remove extra dimension
        x = self.fc(x)  # Apply linear layer
        x = self.bn(x)  # Batch normalization
        x = self.ln(x)  # Layer normalization
        return x

class TripletDataset(Dataset):
    def __init__(self, data, labels, neg_samples):
        self.data = data  # Input data
        self.labels = labels  # Corresponding labels
        self.neg_samples = neg_samples  # Number of negative samples per anchor

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        anchor = self.data[idx]  # Anchor sample
        pos = random.choice(self.data[self.labels == self.labels[idx]])  # Positive sample
        neg = random.sample(self.data[self.labels != self.labels[idx]].tolist(), self.neg_samples)  # Negative samples
        return anchor, pos, neg

def compute_triplet_loss(anchor, pos, neg, margin=1.0):
    pos_dist = torch.norm(anchor - pos, dim=1)  # Distance to positive sample
    neg_dist = torch.min(torch.norm(anchor.unsqueeze(1) - neg, dim=2)  # Distance to closest negative sample
    return torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))  # Triplet loss

def train(model, loader, epochs, lr):
    opt = optim.Adam(model.parameters(), lr=lr)  # Optimizer
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)  # Learning rate scheduler
    losses = []  # Track loss over epochs

    for _ in range(epochs):
        epoch_loss = []
        for anchor, pos, neg in loader:
            opt.zero_grad()  # Clear gradients
            loss = compute_triplet_loss(model(anchor), model(pos), model(neg)) + 0.01 * sum(torch.norm(p, p=2) for p in model.parameters())  # Loss with L2 regularization
            loss.backward()  # Backpropagation
            opt.step()  # Update weights
            epoch_loss.append(loss.item())  # Store loss
        scheduler.step()  # Adjust learning rate
        losses.append(np.mean(epoch_loss))  # Average epoch loss
    return losses

def evaluate_model(model, data, labels, k=5):
    embeds = model(torch.tensor(data, dtype=torch.long)).detach().numpy()  # Get embeddings
    dists = np.linalg.norm(embeds[:, np.newaxis] - embeds, axis=2)  # Pairwise distances
    neighs = np.argsort(dists, axis=1)[:, 1:k+1]  # Nearest neighbors
    tp = np.sum(labels[neighs] == labels[:, np.newaxis], axis=1)  # True positives
    acc = np.mean(np.any(labels[neighs] == labels[:, np.newaxis], axis=1))  # Accuracy
    prec = np.mean(tp / k)  # Precision
    rec = np.mean(tp / np.sum(labels == labels[:, np.newaxis], axis=1))  # Recall
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0  # F1-score
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeds)[:, 0], TSNE(n_components=2).fit_transform(embeds)[:, 1], c=labels, cmap='viridis')  # Visualize embeddings
    plt.colorbar()
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)  # Save model weights

def load_model(model_class, path, vocab_size, embed_dim):
    model = model_class(vocab_size, embed_dim)  # Initialize model
    model.load_state_dict(torch.load(path))  # Load weights
    return model

def plot_loss_history(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')  # Plot loss curve
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def generate_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)  # Random data and labels

def visualize_embeddings(model, data, labels):
    embeds = model(torch.tensor(data, dtype=torch.long)).detach().numpy()  # Get embeddings
    tsne = TSNE(n_components=2).fit_transform(embeds)  # Reduce to 2D
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')  # Scatter plot
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)  # Interactive click
    plt.show()

if __name__ == "__main__":
    data, labels = generate_data(100)  # Generate random data
    dataset = TripletDataset(data, labels, 5)  # Create dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Data loader
    model = EmbeddingModel(101, 10)  # Initialize model
    loss_hist = train(model, loader, 10, 1e-4)  # Train model
    save_model(model, "embedding_model.pth")  # Save model
    plot_loss_history(loss_hist)  # Plot loss
    evaluate_model(model, data, labels)  # Evaluate model
    visualize_embeddings(load_model(EmbeddingModel, "embedding_model.pth", 101, 10), *generate_data(100))  # Visualize embeddings