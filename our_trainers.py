import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, num_features):
        super(EmbeddingModel, self).__init__()
        self.model = nn.Sequential(
            nn.Embedding(embedding_dim, num_features),
            nn.Transpose(1, 2),
            nn.AvgPool1d(kernel_size=10),
            nn.Flatten(),
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.Lambda(lambda x: F.normalize(x, p=2, dim=1))
        )

    def forward(self, x):
        return self.model(x)

class TripletDataset(Dataset):
    def __init__(self, samples, labels, num_negatives, batch_size, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.samples))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        anchor_idx = np.random.choice(batch_indices, size=self.batch_size)
        anchor_label = self.labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_label])
        return {
            'anchor': torch.tensor([self.samples[i] for i in anchor_idx], dtype=torch.long),
            'positive': torch.tensor([self.samples[i] for i in positive_idx], dtype=torch.long),
            'negative': torch.tensor([self.samples[i] for i in negative_idx], dtype=torch.long)
        }

class InputDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return torch.mean(torch.clamp(torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1) - torch.norm(anchor_embeddings.unsqueeze(1) - negative_embeddings, p=2, dim=2).min(dim=1)[0] + 1.0, min=0.0))

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, batch):
        self.optimizer.zero_grad()
        anchor_embeddings = self.model(batch['anchor'])
        positive_embeddings = self.model(batch['positive'])
        negative_embeddings = self.model(batch['negative'])
        loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, dataset, epochs, batch_size):
        for epoch in range(epochs):
            total_loss = 0.0
            dataset.on_epoch_end()
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for i, batch in enumerate(data_loader):
                total_loss += self.train_step(batch).item()
            print(f'Epoch: {epoch + 1}, Loss: {total_loss / (i + 1):.3f}')

    def evaluate(self, dataset, batch_size):
        total_loss = 0.0
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                anchor_embeddings = self.model(batch['anchor'])
                positive_embeddings = self.model(batch['positive'])
                negative_embeddings = self.model(batch['negative'])
                total_loss += self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        print(f'Validation Loss: {total_loss / (i + 1):.3f}')

    def predict(self, dataset, batch_size):
        predictions = []
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    output = self.model(batch['anchor'])
                else:
                    output = self.model(batch)
                predictions.extend(output.cpu().numpy())
        return np.array(predictions)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def calculate_distance(embedding1, embedding2):
    return torch.norm(embedding1 - embedding2, p=2, dim=1)

def calculate_similarity(embedding1, embedding2):
    return torch.sum(embedding1 * embedding2, dim=1) / (torch.norm(embedding1, p=2, dim=1) * torch.norm(embedding2, p=2, dim=1))

def calculate_cosine_distance(embedding1, embedding2):
    return 1 - calculate_similarity(embedding1, embedding2)

def get_nearest_neighbors(embeddings, target_embedding, k=5):
    distances = calculate_distance(embeddings, target_embedding)
    _, indices = torch.topk(-distances, k)
    return indices

def get_similar_embeddings(embeddings, target_embedding, k=5):
    similarities = calculate_similarity(embeddings, target_embedding)
    _, indices = torch.topk(similarities, k)
    return indices

def calculate_knn_accuracy(embeddings, labels, k=5):
    correct = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        _, indices = torch.topk(-distances, k + 1)
        nearest_labels = labels[indices[1:]]
        if labels[i] in nearest_labels:
            correct += 1
    return correct / len(embeddings)

def calculate_knn_precision(embeddings, labels, k=5):
    precision = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        _, indices = torch.topk(-distances, k + 1)
        nearest_labels = labels[indices[1:]]
        precision += len(torch.where(nearest_labels == labels[i])[0]) / k
    return precision / len(embeddings)

def calculate_knn_recall(embeddings, labels, k=5):
    recall = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        _, indices = torch.topk(-distances, k + 1)
        nearest_labels = labels[indices[1:]]
        recall += len(torch.where(nearest_labels == labels[i])[0]) / len(torch.where(labels == labels[i])[0])
    return recall / len(embeddings)

def calculate_knn_f1(embeddings, labels, k=5):
    precision = calculate_knn_precision(embeddings, labels, k)
    recall = calculate_knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    learning_rate = 1e-4

    model = EmbeddingModel(101, 10)
    loss_fn = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, loss_fn)
    dataset = TripletDataset(samples, labels, num_negatives, batch_size)
    trainer.train(dataset, epochs, batch_size)

    input_ids = torch.tensor(np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10)), dtype=torch.long)
    input_dataset = InputDataset(input_ids)
    output = trainer.predict(input_dataset, batch_size=1)
    print(output)

    save_model(model, "triplet_model.pth")

    trainer.evaluate(dataset, batch_size)

    predicted_embeddings = trainer.predict(input_dataset, batch_size=1)
    print(predicted_embeddings)

    distance = calculate_distance(torch.tensor(predicted_embeddings[0], dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float))
    print(distance)

    similarity = calculate_similarity(torch.tensor(predicted_embeddings[0], dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float))
    print(similarity)

    cosine_distance = calculate_cosine_distance(torch.tensor(predicted_embeddings[0], dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float))
    print(cosine_distance)

    all_embeddings = trainer.predict(dataset, batch_size=32)
    nearest_neighbors = get_nearest_neighbors(torch.tensor(all_embeddings, dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float), k=5)
    print(nearest_neighbors)

    similar_embeddings = get_similar_embeddings(torch.tensor(all_embeddings, dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float), k=5)
    print(similar_embeddings)

    print("KNN Accuracy:", calculate_knn_accuracy(torch.tensor(all_embeddings, dtype=torch.float), torch.tensor(labels, dtype=torch.long), k=5))

    print("KNN Precision:", calculate_knn_precision(torch.tensor(all_embeddings, dtype=torch.float), torch.tensor(labels, dtype=torch.long), k=5))

    print("KNN Recall:", calculate_knn_recall(torch.tensor(all_embeddings, dtype=torch.float), torch.tensor(labels, dtype=torch.long), k=5))

    print("KNN F1-score:", calculate_knn_f1(torch.tensor(all_embeddings, dtype=torch.float), torch.tensor(labels, dtype=torch.long), k=5))

if __name__ == "__main__":
    main()