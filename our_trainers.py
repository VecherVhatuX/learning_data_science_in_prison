import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_network(num_embeddings, embedding_dim, margin):
    class TripletNetwork(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, margin):
            super(TripletNetwork, self).__init__()
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.dense = nn.Linear(embedding_dim, embedding_dim)
            self.normalize = nn.BatchNorm1d(embedding_dim)

        def forward(self, inputs):
            embedding = self.embedding(inputs).permute(0, 2, 1)
            pooling = self.pooling(embedding).squeeze(2)
            dense = self.dense(pooling)
            normalize = self.normalize(dense)
            outputs = normalize / normalize.norm(dim=1, keepdim=True)
            return outputs
    return TripletNetwork(num_embeddings, embedding_dim, margin)

def create_dataset(samples, labels, batch_size, num_negatives):
    class TripletDataset(Dataset):
        def __init__(self, samples, labels, batch_size, num_negatives):
            self.samples = samples
            self.labels = labels
            self.batch_size = batch_size
            self.num_negatives = num_negatives

        def __len__(self):
            return -(-len(self.samples) // self.batch_size)

        def __getitem__(self, idx):
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, len(self.samples))
            anchor_idx = np.arange(start_idx, end_idx)
            anchor_labels = self.labels[anchor_idx]

            positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_labels])
            negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_labels])

            return {
                'anchor_input_ids': torch.tensor(self.samples[anchor_idx]),
                'positive_input_ids': torch.tensor(self.samples[positive_idx]),
                'negative_input_ids': torch.tensor(self.samples[negative_idx])
            }
    return TripletDataset(samples, labels, batch_size, num_negatives)

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    anchor_positive_distance = (anchor_embeddings - positive_embeddings).norm(dim=1)
    anchor_negative_distance = (anchor_embeddings[:, None] - negative_embeddings).norm(dim=2)
    min_anchor_negative_distance = anchor_negative_distance.min(dim=1)[0]
    return (anchor_positive_distance - min_anchor_negative_distance + margin).clamp(min=0).mean()

def train_step(network, data, margin, optimizer):
    optimizer.zero_grad()
    anchor_embeddings = network(data['anchor_input_ids'])
    positive_embeddings = network(data['positive_input_ids'])
    negative_embeddings = network(data['negative_input_ids'])
    loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
    loss.backward()
    optimizer.step()
    return loss.item()

def train(network, dataset, epochs, margin, lr):
    data_loader = DataLoader(dataset, batch_size=None, shuffle=True)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(data_loader):
            total_loss += train_step(network, data, margin, optimizer)
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

def evaluate(network, dataset, margin):
    data_loader = DataLoader(dataset, batch_size=None, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            anchor_embeddings = network(data['anchor_input_ids'])
            positive_embeddings = network(data['positive_input_ids'])
            negative_embeddings = network(data['negative_input_ids'])
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            total_loss += loss.item()
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict(network, input_ids):
    return network(input_ids)

def save_model(network, path):
    torch.save(network.state_dict(), path)

def load_model(network, path):
    network.load_state_dict(torch.load(path))

def main():
    np.random.seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    network = create_network(num_embeddings, embedding_dim, margin)
    dataset = create_dataset(samples, labels, batch_size, num_negatives)
    train(network, dataset, epochs, margin, lr)
    input_ids = torch.tensor([1, 2, 3, 4, 5])[None, :]
    output = predict(network, input_ids)
    print(output)
    save_model(network, "triplet_model.pth")
    load_model(network, "triplet_model.pth")
    print("Model saved and loaded successfully.")

if __name__ == "__main__":
    main()