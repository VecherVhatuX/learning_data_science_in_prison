import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from functools import partial

# Define the TripletNetwork class
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

# Define the triplet loss function
def triplet_loss_function(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    return (torch.norm(anchor_embeddings - positive_embeddings, dim=1) 
            - torch.norm(anchor_embeddings[:, None] - negative_embeddings, dim=2).min(dim=1)[0] + margin).clamp(min=0).mean()

# Define the TripletDataset class
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

# Define the create_triplet_data_loader function
def create_triplet_data_loader(samples, labels, batch_size, num_negatives):
    return TripletDataset(samples, labels, batch_size, num_negatives)

# Define the create_triplet_architecture function
def create_triplet_architecture(num_embeddings, embedding_dim, margin):
    return TripletNetwork(num_embeddings, embedding_dim, margin)

# Define the train_triplet_network function
def train_triplet_network(network, dataset, epochs, margin, learning_rate):
    data_loader = DataLoader(dataset, batch_size=None, shuffle=True)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            anchor_embeddings = network(data['anchor_input_ids'])
            positive_embeddings = network(data['positive_input_ids'])
            negative_embeddings = network(data['negative_input_ids'])
            loss = triplet_loss_function(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

# Define the evaluate_triplet_network function
def evaluate_triplet_network(network, dataset, margin):
    data_loader = DataLoader(dataset, batch_size=None, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            anchor_embeddings = network(data['anchor_input_ids'])
            positive_embeddings = network(data['positive_input_ids'])
            negative_embeddings = network(data['negative_input_ids'])
            loss = triplet_loss_function(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            total_loss += loss.item()
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

# Define the predict_with_triplet_network function
def predict_with_triplet_network(network, input_ids):
    return network(input_ids)

# Define the save_triplet_model function
def save_triplet_model(network, path):
    torch.save(network.state_dict(), path)

# Define the load_triplet_model function
def load_triplet_model(network, path):
    network.load_state_dict(torch.load(path))

# Define the main function
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
    learning_rate = 1e-4

    network = create_triplet_architecture(num_embeddings, embedding_dim, margin)
    dataset = create_triplet_data_loader(samples, labels, batch_size, num_negatives)
    train_triplet_network(network, dataset, epochs, margin, learning_rate)
    input_ids = torch.tensor([1, 2, 3, 4, 5])[None, :]
    output = predict_with_triplet_network(network, input_ids)
    print(output)
    save_triplet_model(network, "triplet_model.pth")
    load_triplet_model(network, "triplet_model.pth")
    print("Model saved and loaded successfully.")

    # Evaluate the model
    evaluate_triplet_network(network, dataset, margin)

    # Use the model for prediction
    predicted_embeddings = predict_with_triplet_network(network, torch.tensor([1, 2, 3, 4, 5])[None, :])
    print(predicted_embeddings)

    # Define a function to calculate the distance between two embeddings
    def calculate_distance(embedding1, embedding2):
        return torch.norm(embedding1 - embedding2, dim=1)

    # Calculate the distance between two predicted embeddings
    distance = calculate_distance(predicted_embeddings, predicted_embeddings)
    print(distance)

    # Define a function to calculate the similarity between two embeddings
    def calculate_similarity(embedding1, embedding2):
        return torch.dot(embedding1, embedding2) / (torch.norm(embedding1) * torch.norm(embedding2))

    # Calculate the similarity between two predicted embeddings
    similarity = calculate_similarity(predicted_embeddings, predicted_embeddings)
    print(similarity)

if __name__ == "__main__":
    main()