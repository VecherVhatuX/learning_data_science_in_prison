import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random


class Embedder(nn.Module):
    def __init__(self, embedding_size, feature_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(embedding_size, feature_size),
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.LayerNorm(feature_size)
        )

    def forward(self, x):
        return self.model(x)


def get_triplet(anchor, anchor_label, labels, n_negatives):
    pos_indices = np.where(labels == anchor_label.item())[0]
    positive_idx = random.choice(pos_indices.tolist())
    negative_indices = random.sample(np.where(labels != anchor_label.item())[0].tolist(), n_negatives)
    return anchor, labels[positive_idx], [labels[i] for i in negative_indices]


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, n_negatives):
        self.samples = samples
        self.labels = labels
        self.n_negatives = n_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor = self.samples[index]
        anchor_label = self.labels[index]
        return get_triplet(anchor, anchor_label, self.labels, self.n_negatives)

    def shuffle(self):
        indices = np.arange(len(self.samples))
        np.random.shuffle(indices)
        self.samples = self.samples[indices]
        self.labels = self.labels[indices]


def create_triplet_dataset(samples, labels, n_negatives):
    return TripletDataset(samples, labels, n_negatives)


def triplet_loss_fn(margin=1.0):
    def loss(anchor, positive, negatives):
        dist_ap = torch.norm(anchor - positive, dim=1)
        dist_an = torch.norm(anchor.unsqueeze(1) - torch.stack(negatives), dim=2)
        return torch.mean(torch.clamp(dist_ap - dist_an.min(dim=1)[0] + margin, min=0.0))
    return loss


def train_model(model, dataset, num_epochs):
    optimizer = optim.Adam(model.parameters())
    loss_function = triplet_loss_fn()
    model.train()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        dataset.shuffle()
        for anchors, positives, negatives in loader:
            optimizer.zero_grad()
            anchor_embeds = model(anchors)
            positive_embeds = model(positives)
            negative_embeds = torch.stack([model(neg) for neg in negatives])
            loss_value = loss_function(anchor_embeds, positive_embeds, negative_embeds)
            loss_value.backward()
            optimizer.step()


def validate_model(model, samples, labels, k=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(samples)).numpy()
        metrics = calculate_knn_metrics(embeddings, labels, k)
        print_validation_metrics(metrics)
        display_embeddings(embeddings, labels)


def print_validation_metrics(metrics):
    print(f"Validation KNN Accuracy: {metrics[0]}")
    print(f"Validation KNN Precision: {metrics[1]}")
    print(f"Validation KNN Recall: {metrics[2]}")
    print(f"Validation KNN F1-score: {metrics[3]}")


def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))


def save_model_to_file(model, file_path):
    torch.save(model.state_dict(), file_path)


def extract_embeddings_from_model(model, samples):
    with torch.no_grad():
        return model(torch.tensor(samples)).numpy()


def display_embeddings(embeddings, labels):
    tsne_model = TSNE(n_components=2)
    reduced_embeddings = tsne_model.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()


def calculate_knn_metrics(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k + 1]
    
    true_positive = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    precision = np.mean(true_positive / k)
    recall = np.mean(true_positive / np.sum(labels == labels[:, np.newaxis], axis=1))
    
    accuracy = np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def execute_training_pipeline(learning_rate, batch_size, num_epochs, n_negatives, embedding_size, feature_size, data_size):
    samples, labels = generate_data(data_size)
    dataset = create_triplet_dataset(samples, labels, n_negatives)
    model = Embedder(embedding_size, feature_size)
    train_model(model, dataset, num_epochs)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape((1, 10))
    model(input_ids)
    save_model_to_file(model, "triplet_model.pth")
    predicted_embeddings = extract_embeddings_from_model(model, samples)
    validate_model(model, samples, labels)


if __name__ == "__main__":
    execute_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)