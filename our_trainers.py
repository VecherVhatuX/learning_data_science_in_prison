import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class WordVectorGenerator(nn.Module):
    """
    A neural network model for generating word embeddings.
    The model consists of an embedding layer, pooling layer, projection layer, and normalization layers.
    """
    def __init__(self, vocab_size, embedding_size):
        super(WordVectorGenerator, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.projection_layer = nn.Linear(embedding_size, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, input_ids):
        """
        Forward pass through the model.
        Args:
            input_ids (torch.Tensor): Input tensor containing word indices.
        Returns:
            torch.Tensor: Normalized word embeddings.
        """
        embeddings = self.embedding_layer(input_ids)
        pooled = self.pooling_layer(embeddings.transpose(1, 2)).squeeze(2)
        projected = self.projection_layer(pooled)
        normalized = self.batch_norm(projected)
        return self.layer_norm(normalized)

class TripletDataLoader(Dataset):
    """
    A custom dataset class for loading triplet data (anchor, positive, negative samples).
    """
    def __init__(self, data, labels, negative_samples):
        self.data = data
        self.labels = labels
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a triplet (anchor, positive, negatives) for a given index.
        Args:
            index (int): Index of the data point.
        Returns:
            tuple: (anchor, positive, negatives) as torch.Tensor.
        """
        anchor = self.data[index]
        anchor_label = self.labels[index]
        positive = random.choice(self.data[self.labels == anchor_label])
        negatives = random.sample(self.data[self.labels != anchor_label].tolist(), self.negative_samples)
        return torch.tensor(anchor, dtype=torch.long), torch.tensor(positive, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)

def calculate_triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Calculates the triplet loss for a batch of anchor, positive, and negative samples.
    Args:
        anchor (torch.Tensor): Anchor embeddings.
        positive (torch.Tensor): Positive embeddings.
        negative (torch.Tensor): Negative embeddings.
        margin (float): Margin for the triplet loss.
    Returns:
        torch.Tensor: Triplet loss value.
    """
    positive_distance = torch.norm(anchor - positive, dim=1)
    negative_distance = torch.min(torch.norm(anchor.unsqueeze(1) - negative, dim=2), dim=1)[0]
    return torch.mean(torch.clamp(positive_distance - negative_distance + margin, min=0.0))

def train_vector_generator(model, dataset, num_epochs, learning_rate):
    """
    Trains the WordVectorGenerator model using triplet loss.
    Args:
        model (WordVectorGenerator): The model to train.
        dataset (TripletDataLoader): The dataset containing triplets.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    Returns:
        list: Loss history over epochs.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for anchor, positive, negative in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            anchor_vector = model(anchor)
            positive_vector = model(positive)
            negative_vector = model(negative)
            loss = calculate_triplet_loss(anchor_vector, positive_vector, negative_vector)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(dataset))
    return loss_history

def evaluate_model(model, data, labels, k=5):
    """
    Evaluates the model by computing performance metrics and visualizing embeddings.
    Args:
        model (WordVectorGenerator): The trained model.
        data (np.array): Input data.
        labels (np.array): Corresponding labels.
        k (int): Number of nearest neighbors for evaluation.
    """
    embeddings = model(torch.tensor(data, dtype=torch.long)).detach().numpy()
    show_metrics(compute_performance_metrics(embeddings, labels, k))
    plot_embeddings(embeddings, labels)

def show_metrics(metrics):
    """
    Displays performance metrics (accuracy, precision, recall, F1-score).
    Args:
        metrics (tuple): Tuple containing accuracy, precision, recall, and F1-score.
    """
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_model(model, file_path):
    """
    Saves the model's state dictionary to a file.
    Args:
        model (WordVectorGenerator): The model to save.
        file_path (str): Path to save the model.
    """
    torch.save(model.state_dict(), file_path)

def load_model(model_class, file_path, vocab_size, embedding_size):
    """
    Loads a model from a saved state dictionary.
    Args:
        model_class (class): The model class to instantiate.
        file_path (str): Path to the saved model.
        vocab_size (int): Vocabulary size for the model.
        embedding_size (int): Embedding size for the model.
    Returns:
        WordVectorGenerator: The loaded model.
    """
    model = model_class(vocab_size, embedding_size)
    model.load_state_dict(torch.load(file_path))
    return model

def generate_embeddings(model, data):
    """
    Generates embeddings for the input data using the model.
    Args:
        model (WordVectorGenerator): The trained model.
        data (np.array): Input data.
    Returns:
        np.array: Generated embeddings.
    """
    return model(torch.tensor(data, dtype=torch.long)).detach().numpy()

def plot_embeddings(embeddings, labels):
    """
    Visualizes embeddings using t-SNE.
    Args:
        embeddings (np.array): Embeddings to visualize.
        labels (np.array): Corresponding labels.
    """
    plt.figure(figsize=(8, 8))
    tsne_result = TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def compute_performance_metrics(embeddings, labels, k=5):
    """
    Computes performance metrics (accuracy, precision, recall, F1-score) for the embeddings.
    Args:
        embeddings (np.array): Embeddings to evaluate.
        labels (np.array): Corresponding labels.
        k (int): Number of nearest neighbors for evaluation.
    Returns:
        tuple: (accuracy, precision, recall, F1-score).
    """
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_neighbors = np.argsort(distances, axis=1)[:, 1:k + 1]
    true_positives = np.sum(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def plot_loss_history(loss_history):
    """
    Plots the training loss history over epochs.
    Args:
        loss_history (list): List of loss values over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training(learning_rate, batch_size, num_epochs, negative_samples, vocab_size, embedding_size, data_size):
    """
    Runs the training pipeline for the WordVectorGenerator model.
    Args:
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        negative_samples (int): Number of negative samples per triplet.
        vocab_size (int): Vocabulary size for the model.
        embedding_size (int): Embedding size for the model.
        data_size (int): Size of the dataset to generate.
    """
    data, labels = generate_data(data_size)
    dataset = TripletDataLoader(data, labels, negative_samples)
    model = WordVectorGenerator(vocab_size, embedding_size)
    save_model(model, "word_vector_generator.pth")
    loss_history = train_vector_generator(model, dataset, num_epochs, learning_rate)
    plot_loss_history(loss_history)
    evaluate_model(model, data, labels)

def display_model_architecture(model):
    """
    Displays the architecture of the model.
    Args:
        model (WordVectorGenerator): The model to display.
    """
    print(model)

def train_with_early_termination(model, dataset, num_epochs, learning_rate, patience=5):
    """
    Trains the model with early termination based on loss improvement.
    Args:
        model (WordVectorGenerator): The model to train.
        dataset (TripletDataLoader): The dataset containing triplets.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before stopping.
    Returns:
        list: Loss history over epochs.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for anchor, positive, negative in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            anchor_vector = model(anchor)
            positive_vector = model(positive)
            negative_vector = model(negative)
            loss = calculate_triplet_loss(anchor_vector, positive_vector, negative_vector)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

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
    """
    Generates random data and labels for training.
    Args:
        data_size (int): Size of the dataset to generate.
    Returns:
        tuple: (data, labels) as np.array.
    """
    data = np.random.randint(0, 100, (data_size, 10))
    labels = np.random.randint(0, 10, data_size)
    return data, labels

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_architecture(WordVectorGenerator(101, 10))