import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int, num_features: int):
        super(EmbeddingModel, self).__init__()
        self.embedding_layer = nn.Embedding(embedding_dim, num_features, padding_idx=0)
        self.pooling_layer = nn.AdaptiveAvgPool1d((1,))
        self.flatten_layer = nn.Flatten()
        self.dense_layer = nn.Linear(num_features, num_features)
        self.batch_norm_layer = nn.BatchNorm1d(num_features)
        self.l2_norm_layer = nn.LazyInstanceNorm1d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(x)
        x = self.pooling_layer(x.transpose(1, 2)).transpose(1, 2)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.batch_norm_layer(x)
        x = self.l2_norm_layer(x)
        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(x)

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor) -> torch.Tensor:
        positive_distances = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
        negative_distances = torch.norm(anchor_embeddings.unsqueeze(1) - negative_embeddings, dim=-1)
        loss = torch.clamp(positive_distances - negative_distances.min(dim=1).values + self.margin, min=0.0)
        return loss.mean()

class InputDataset(Dataset):
    def __init__(self, input_ids: np.ndarray):
        self.input_ids = input_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.input_ids[index]

class TripletDataset(Dataset):
    def __init__(self, samples: np.ndarray, labels: np.ndarray, num_negatives: int, batch_size: int, shuffle: bool = True):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.samples))
        self.on_epoch_end()

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.samples) // self.batch_size

    def __getitem__(self, index: int) -> dict:
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        anchor_idx = np.random.choice(batch_indices, size=self.batch_size, replace=False)
        anchor_label = self.labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_label])
        return {
            'anchor': self.samples[anchor_idx],
            'positive': self.samples[positive_idx],
            'negative': self.samples[negative_idx]
        }

def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str) -> None:
    model.load_state_dict(torch.load(path))

def calculate_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    return torch.norm(embedding1 - embedding2, dim=-1)

def calculate_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    return torch.sum(embedding1 * embedding2, dim=-1) / (torch.norm(embedding1, dim=-1) * torch.norm(embedding2, dim=-1))

def calculate_cosine_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    return 1 - calculate_similarity(embedding1, embedding2)

def get_nearest_neighbors(embeddings: torch.Tensor, target_embedding: torch.Tensor, k: int = 5) -> torch.Tensor:
    distances = calculate_distance(embeddings, target_embedding)
    return torch.argsort(distances)[:k]

def get_similar_embeddings(embeddings: torch.Tensor, target_embedding: torch.Tensor, k: int = 5) -> torch.Tensor:
    similarities = calculate_similarity(embeddings, target_embedding)
    return torch.argsort(-similarities)[:k]

def calculate_knn_accuracy(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    correct = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = torch.argsort(distances)[1:k+1]
        if labels[i] in labels[indices]:
            correct += 1
    return correct / len(embeddings)

def calculate_knn_precision(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    precision = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = torch.argsort(distances)[1:k+1]
        precision += len(torch.where(labels[indices] == labels[i])[0]) / k
    return precision / len(embeddings)

def calculate_knn_recall(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    recall = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = torch.argsort(distances)[1:k+1]
        recall += len(torch.where(labels[indices] == labels[i])[0]) / len(torch.where(labels == labels[i])[0])
    return recall / len(embeddings)

def calculate_knn_f1(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    precision = calculate_knn_precision(embeddings, labels, k)
    recall = calculate_knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)

def train(model: nn.Module, dataset: Dataset, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int) -> None:
    for epoch in range(epochs):
        for batch in dataset:
            anchor_embeddings = model(batch['anchor'])
            positive_embeddings = model(batch['positive'])
            negative_embeddings = model(batch['negative'])
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def test(model: nn.Module, dataset: Dataset) -> torch.Tensor:
    return model(torch.from_numpy(dataset.samples))

def get_model(embedding_dim: int, num_features: int) -> nn.Module:
    return EmbeddingModel(embedding_dim, num_features)

def get_criterion(margin: float = 1.0) -> nn.Module:
    return TripletLoss(margin)

def get_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate)

def get_dataset(samples: np.ndarray, labels: np.ndarray, num_negatives: int, batch_size: int, shuffle: bool = True) -> Dataset:
    return TripletDataset(samples, labels, num_negatives, batch_size, shuffle)

def get_input_dataset(input_ids: np.ndarray) -> Dataset:
    return InputDataset(input_ids)

def main() -> None:
    np.random.seed(42)

    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    learning_rate = 1e-4
    embedding_dim = 101
    num_features = 10

    model = get_model(embedding_dim, num_features)
    criterion = get_criterion()
    optimizer = get_optimizer(model, learning_rate)
    dataset = get_dataset(samples, labels, num_negatives, batch_size)

    train(model, dataset, criterion, optimizer, epochs)

    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).reshape((1, 10))
    output = model(torch.from_numpy(input_ids))

    save_model(model, "triplet_model.pth")

    predicted_embeddings = test(model, dataset)
    print(predicted_embeddings)

    distance = calculate_distance(output, output)
    print(distance)

    similarity = calculate_similarity(output, output)
    print(similarity)

    cosine_distance = calculate_cosine_distance(output, output)
    print(cosine_distance)

    all_embeddings = model(torch.from_numpy(dataset.samples))
    nearest_neighbors = get_nearest_neighbors(all_embeddings, output, k=5)
    print(nearest_neighbors)

    similar_embeddings = get_similar_embeddings(all_embeddings, output, k=5)
    print(similar_embeddings)

    print("KNN Accuracy:", calculate_knn_accuracy(all_embeddings, torch.from_numpy(labels), k=5))

    print("KNN Precision:", calculate_knn_precision(all_embeddings, torch.from_numpy(labels), k=5))

    print("KNN Recall:", calculate_knn_recall(all_embeddings, torch.from_numpy(labels), k=5))

    print("KNN F1-score:", calculate_knn_f1(all_embeddings, torch.from_numpy(labels), k=5))

if __name__ == "__main__":
    main()