import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

class TripletModel(nn.Module):
    def __init__(self, num_embeddings, features):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, features)
        self.avg_pool = nn.AvgPool1d(kernel_size=10)
        self.fc = nn.Linear(features, features)
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.avg_pool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def calculate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    return torch.mean(torch.clamp(torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1) - torch.norm(anchor_embeddings.unsqueeze(1) - negative_embeddings, p=2, dim=2).min(dim=1)[0] + 1.0, min=0.0))

class Dataset(Dataset):
    def __init__(self, samples, labels, num_negatives, batch_size, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)
        batch_indices = indices[index*self.batch_size:(index+1)*self.batch_size]
        anchor_idx = np.random.choice(batch_indices, size=self.batch_size)
        anchor_label = self.labels[anchor_idx]

        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_label])

        return {
            'anchor_input_ids': torch.tensor([self.samples[i] for i in anchor_idx], dtype=torch.long),
            'positive_input_ids': torch.tensor([self.samples[i] for i in positive_idx], dtype=torch.long),
            'negative_input_ids': torch.tensor([self.samples[i] for i in negative_idx], dtype=torch.long)
        }

class Trainer:
    def __init__(self, model, learning_rate, batch_size):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def train_step(self, batch):
        self.optimizer.zero_grad()
        anchor_embeddings = self.model(batch['anchor_input_ids'])
        positive_embeddings = self.model(batch['positive_input_ids'])
        negative_embeddings = self.model(batch['negative_input_ids'])
        loss = calculate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        self.optimizer.step()

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                self.train_step(batch)
                total_loss += calculate_triplet_loss(self.model(batch['anchor_input_ids']), self.model(batch['positive_input_ids']), self.model(batch['negative_input_ids']))
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(self, dataset):
        total_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                total_loss += calculate_triplet_loss(self.model(batch['anchor_input_ids']), self.model(batch['positive_input_ids']), self.model(batch['negative_input_ids']))
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def predict(self, input_ids, batch_size):
        predictions = []
        dataloader = DataLoader(input_ids, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in dataloader:
                output = self.model(batch)
                predictions.extend(output.numpy())
        return np.array(predictions)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load_model(path, model):
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
        _, indices = torch.topk(-distances, k+1)
        nearest_labels = labels[indices[1:]]
        if labels[i] in nearest_labels:
            correct += 1
    return correct / len(embeddings)

def calculate_knn_precision(embeddings, labels, k=5):
    precision = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        _, indices = torch.topk(-distances, k+1)
        nearest_labels = labels[indices[1:]]
        precision += len(torch.where(nearest_labels == labels[i])[0]) / k
    return precision / len(embeddings)

def calculate_knn_recall(embeddings, labels, k=5):
    recall = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        _, indices = torch.topk(-distances, k+1)
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

    model = TripletModel(num_embeddings=101, features=10)
    trainer = Trainer(model, learning_rate, batch_size)
    dataset = Dataset(samples, labels, num_negatives, batch_size)
    trainer.train(dataset, epochs)

    input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10))
    output = trainer.predict(torch.tensor(input_ids, dtype=torch.long), batch_size=1)
    print(output)

    trainer.save_model("triplet_model.pth")

    trainer.evaluate(dataset)

    predicted_embeddings = trainer.predict(torch.tensor(np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10)), dtype=torch.long), batch_size=1)
    print(predicted_embeddings)

    distance = calculate_distance(torch.tensor(predicted_embeddings[0], dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float))
    print(distance)

    similarity = calculate_similarity(torch.tensor(predicted_embeddings[0], dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float))
    print(similarity)

    cosine_distance = calculate_cosine_distance(torch.tensor(predicted_embeddings[0], dtype=torch.float), torch.tensor(predicted_embeddings[0], dtype=torch.float))
    print(cosine_distance)

    all_embeddings = trainer.predict(torch.tensor(np.array(samples, dtype=np.int32), dtype=torch.long), batch_size=32)
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