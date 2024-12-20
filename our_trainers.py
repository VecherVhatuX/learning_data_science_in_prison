import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

class TripletModel(nn.Module):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(embedding_dim, num_features)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_features, num_features)
        self.bn = nn.BatchNorm1d(num_features)
        self.ln = nn.LayerNorm(num_features)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x.transpose(1, 2)).transpose(1, 2)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.ln(x)
        return x

class TripletDataset(Dataset):
    def __init__(self, samples, labels, num_negatives):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_idx = idx
        anchor_label = self.labels[idx]
        positive_idx = random.choice(np.where(self.labels == anchor_label)[0].tolist())
        negative_idx = random.sample(np.where(self.labels != anchor_label)[0].tolist(), self.num_negatives)
        return self.samples[anchor_idx], self.samples[positive_idx], [self.samples[i] for i in negative_idx]

def create_triplet_loss(margin=1.0):
    def triplet_loss(anchor, positive, negative):
        d_ap = torch.norm(anchor - positive, dim=1)
        d_an = torch.norm(anchor.unsqueeze(1) - torch.stack(negative, dim=1), dim=2)
        loss = torch.maximum(d_ap - torch.min(d_an, dim=1)[0] + margin, torch.zeros_like(d_ap))
        return torch.mean(loss)
    return triplet_loss

def train(model, device, loader, optimizer, loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            anchor, positive, negative = [x.to(device) for x in batch]
            optimizer.zero_grad()
            anchor_embeddings = model(anchor)
            positive_embeddings = model(positive)
            negative_embeddings = [model(x) for x in negative]
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

def validate(model, device, samples, labels, k=5):
    model.eval()
    with torch.no_grad():
        predicted_embeddings = model(samples.to(device))
        predicted_embeddings = predicted_embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        print("Validation KNN Accuracy:", knn_accuracy(predicted_embeddings, labels, k))
        print("Validation KNN Precision:", knn_precision(predicted_embeddings, labels, k))
        print("Validation KNN Recall:", knn_recall(predicted_embeddings, labels, k))
        print("Validation KNN F1-score:", knn_f1(predicted_embeddings, labels, k))
        embedding_visualization(predicted_embeddings, labels)
    model.train()

def generate_data(size):
    return torch.from_numpy(np.random.randint(0, 100, (size, 10))), torch.from_numpy(np.random.randint(0, 2, (size,)))

def create_dataset(samples, labels, num_negatives):
    return TripletDataset(samples, labels, num_negatives)

def create_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_model(embedding_dim, num_features, device):
    model = TripletModel(embedding_dim, num_features).to(device)
    return model

def create_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

def create_loss_function():
    return create_triplet_loss()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def get_predicted_embeddings(model, samples, device):
    return model(samples.to(device)).cpu().numpy()

def calculate_distances(output):
    print(torch.norm(output - output, dim=1).numpy())
    print(torch.sum(output * output, dim=1) / (torch.norm(output, dim=1) * torch.norm(output, dim=1)).numpy())
    print(1 - torch.sum(output * output, dim=1) / (torch.norm(output, dim=1) * torch.norm(output, dim=1)).numpy())

def calculate_neighbors(predicted_embeddings, output, k=5):
    print(torch.argsort(torch.norm(torch.tensor(predicted_embeddings) - output, dim=1), dim=1)[:,:k].numpy())
    print(torch.argsort(torch.sum(torch.tensor(predicted_embeddings) * output, dim=1) / (torch.norm(torch.tensor(predicted_embeddings), dim=1) * torch.norm(output, dim=1)), dim=1, descending=True)[:,:k].numpy())

def embedding_visualization(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
    plt.show()

def knn_accuracy(embeddings, labels, k=5):
    return np.mean(np.any(np.equal(labels[np.argsort(np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), axis=1)[:, 1:k+1]], labels[:, np.newaxis]), axis=1))

def knn_precision(embeddings, labels, k=5):
    return np.mean(np.sum(np.equal(labels[np.argsort(np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), axis=1)[:, 1:k+1]], labels[:, np.newaxis]), axis=1) / k)

def knn_recall(embeddings, labels, k=5):
    return np.mean(np.sum(np.equal(labels[np.argsort(np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), axis=1)[:, 1:k+1]], labels[:, np.newaxis]), axis=1) / np.sum(np.equal(labels, labels[:, np.newaxis]), axis=1))

def knn_f1(embeddings, labels, k=5):
    precision = knn_precision(embeddings, labels, k)
    recall = knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)

def pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features, size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples, labels = generate_data(size)
    dataset = create_dataset(samples, labels, num_negatives)
    loader = create_data_loader(dataset, batch_size)
    model = create_model(embedding_dim, num_features, device)
    optimizer = create_optimizer(model, learning_rate)
    loss_fn = create_loss_function()
    train(model, device, loader, optimizer, loss_fn, epochs)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int32).reshape((1, 10)).to(device)
    output = model(input_ids)
    save_model(model, "triplet_model.pth")
    predicted_embeddings = get_predicted_embeddings(model, samples, device)
    calculate_distances(output)
    calculate_neighbors(predicted_embeddings, output.cpu().numpy())
    validate(model, device, samples, labels)

if __name__ == "__main__":
    pipeline(1e-4, 32, 10, 5, 101, 10, 100)