import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class WordVectorGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(WordVectorGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(embedding_size, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(2)
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        return x

class TripletDataset(Dataset):
    def __init__(self, data, labels, negative_samples):
        self.data = data
        self.labels = labels
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = torch.tensor(self.data[idx], dtype=torch.long)
        positive = torch.tensor(random.choice(self.data[self.labels == self.labels[idx]]), dtype=torch.long)
        negatives = torch.tensor(random.sample(self.data[self.labels != self.labels[idx]].tolist(), self.negative_samples), dtype=torch.long)
        return anchor, positive, negatives

def calculate_triplet_loss(anchor, positive, negative, margin=1.0):
    return torch.mean(torch.clamp(torch.norm(anchor - positive, dim=1) - torch.min(torch.norm(anchor.unsqueeze(1) - negative, dim=2), dim=1) + margin, min=0.0))

def train_vector_generator(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = add_learning_rate_scheduler(optimizer)
    loss_history = []
    for _ in range(num_epochs):
        epoch_loss = 0
        for anchor, positive, negative in dataloader:
            optimizer.zero_grad()
            loss = calculate_triplet_loss(model(anchor), model(positive), model(negative)) + add_custom_regularization(model)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        loss_history.append(epoch_loss / len(dataloader))
    return loss_history

def evaluate_model(model, data, labels, k=5):
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    metrics = compute_performance_metrics(embeddings, labels, k)
    show_metrics(metrics)
    plot_embeddings(embeddings, labels)

def show_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(model_class, file_path, vocab_size, embedding_size):
    model = model_class(vocab_size, embedding_size)
    model.load_state_dict(torch.load(file_path))
    return model

def generate_embeddings(model, data):
    return model(torch.tensor(data, dtype=torch.long)).detach().numpy()

def plot_embeddings(embeddings, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def compute_performance_metrics(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_neighbors = np.argsort(distances, axis=1)[:, 1:k + 1]
    true_positives = np.sum(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training(learning_rate, batch_size, num_epochs, negative_samples, vocab_size, embedding_size, data_size):
    data, labels = generate_data(data_size)
    model = WordVectorGenerator(vocab_size, embedding_size)
    save_model(model, "word_vector_generator.pth")
    dataset = TripletDataset(data, labels, negative_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = train_vector_generator(model, dataloader, num_epochs, learning_rate)
    plot_loss_history(loss_history)
    evaluate_model(model, data, labels)

def display_model_architecture(model):
    print(model)

def train_with_early_termination(model, dataloader, num_epochs, learning_rate, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = add_learning_rate_scheduler(optimizer)
    best_loss = float('inf')
    no_improvement = 0
    loss_history = []
    for epoch in range(num_epochs):
        avg_loss = sum(calculate_triplet_loss(model(anchor), model(positive), model(negative)) + add_custom_regularization(model) for anchor, positive, negative in dataloader) / len(dataloader)
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(avg_loss.item())
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    return loss_history

def generate_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

def visualize_embeddings_interactive(model, data, labels):
    embeddings = generate_embeddings(model, data)
    tsne_result = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne_result - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def add_custom_regularization(model, lambda_reg=0.01):
    return lambda_reg * sum(torch.norm(param, p=2) for param in model.parameters())

def add_learning_rate_scheduler(optimizer, step_size=30, gamma=0.1):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_architecture(WordVectorGenerator(101, 10))
    visualize_embeddings_interactive(load_model(WordVectorGenerator, "word_vector_generator.pth", 101, 10), *generate_data(100))