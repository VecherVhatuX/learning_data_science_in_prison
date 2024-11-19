import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TripletNetwork(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, margin):
        super(TripletNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d((1,))
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.normalize = nn.BatchNorm1d(embedding_dim)
        self.margin = margin

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        pooling = self.pooling(embedding.permute(0, 2, 1)).squeeze(2)
        dense = self.dense(pooling)
        normalize = self.normalize(dense)
        outputs = normalize / torch.norm(normalize, dim=1, keepdim=True)
        return outputs

    def triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return torch.mean(torch.clamp(torch.norm(anchor_embeddings - positive_embeddings, dim=1) 
                                      - torch.norm(anchor_embeddings.unsqueeze(1) - negative_embeddings, dim=2).min(dim=1)[0] + self.margin, min=0))

class TripletDataset(Dataset):
    def __init__(self, samples, labels, num_negatives):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.sample_indices = np.arange(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        np.random.shuffle(self.sample_indices)
        idx = self.sample_indices[idx]
        anchor_idx = idx
        anchor_label = self.labels[idx]

        positive_idx = np.random.choice(np.where(self.labels == anchor_label)[0], size=1)[0]
        negative_idx = np.random.choice(np.where(self.labels != anchor_label)[0], size=self.num_negatives, replace=False)

        return {
            'anchor_input_ids': torch.tensor(self.samples[anchor_idx], dtype=torch.long),
            'positive_input_ids': torch.tensor(self.samples[positive_idx], dtype=torch.long),
            'negative_input_ids': torch.tensor(self.samples[negative_idx], dtype=torch.long)
        }

def train_triplet_network(network, dataset, epochs, learning_rate, batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        total_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i, data in enumerate(dataloader):
            anchor_input_ids = data['anchor_input_ids'].to(device)
            positive_input_ids = data['positive_input_ids'].to(device)
            negative_input_ids = data['negative_input_ids'].to(device)

            optimizer.zero_grad()
            anchor_embeddings = network(anchor_input_ids)
            positive_embeddings = network(positive_input_ids)
            negative_embeddings = network(negative_input_ids)
            loss = network.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

def evaluate_triplet_network(network, dataset, batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()
    total_loss = 0.0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            anchor_input_ids = data['anchor_input_ids'].to(device)
            positive_input_ids = data['positive_input_ids'].to(device)
            negative_input_ids = data['negative_input_ids'].to(device)

            anchor_embeddings = network(anchor_input_ids)
            positive_embeddings = network(positive_input_ids)
            negative_embeddings = network(negative_input_ids)
            loss = network.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss.item()
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict_with_triplet_network(network, input_ids, batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()
    predictions = []
    dataloader = DataLoader(input_ids, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = network(data)
            predictions.extend(output.cpu().numpy())
    return predictions

def save_triplet_model(network, path):
    torch.save(network.state_dict(), path)

def load_triplet_model(network, path):
    network.load_state_dict(torch.load(path))

def calculate_distance(embedding1, embedding2):
    return torch.norm(embedding1 - embedding2, dim=1)

def calculate_similarity(embedding1, embedding2):
    return torch.sum(embedding1 * embedding2, dim=1) / (torch.norm(embedding1, dim=1) * torch.norm(embedding2, dim=1))

def calculate_cosine_distance(embedding1, embedding2):
    return 1 - calculate_similarity(embedding1, embedding2)

def get_nearest_neighbors(embeddings, target_embedding, k=5):
    distances = calculate_distance(embeddings, target_embedding)
    _, indices = torch.topk(distances, k, largest=False)
    return indices

def get_similar_embeddings(embeddings, target_embedding, k=5):
    similarities = calculate_similarity(embeddings, target_embedding)
    _, indices = torch.topk(similarities, k, largest=True)
    return indices

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

    network = TripletNetwork(num_embeddings, embedding_dim, margin)
    dataset = TripletDataset(samples, labels, num_negatives)
    train_triplet_network(network, dataset, epochs, learning_rate, batch_size)
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long).unsqueeze(0)
    output = predict_with_triplet_network(network, input_ids, batch_size=1)
    print(output)
    save_triplet_model(network, "triplet_model.pth")
    loaded_network = TripletNetwork(num_embeddings, embedding_dim, margin)
    load_triplet_model(loaded_network, "triplet_model.pth")
    print("Model saved and loaded successfully.")

    evaluate_triplet_network(network, dataset, batch_size)

    predicted_embeddings = predict_with_triplet_network(network, torch.tensor([1, 2, 3, 4, 5], dtype=torch.long).unsqueeze(0), batch_size=1)
    print(predicted_embeddings)

    distance = calculate_distance(torch.tensor(predicted_embeddings[0]), torch.tensor(predicted_embeddings[0]))
    print(distance)

    similarity = calculate_similarity(torch.tensor(predicted_embeddings[0]), torch.tensor(predicted_embeddings[0]))
    print(similarity)

    cosine_distance = calculate_cosine_distance(torch.tensor(predicted_embeddings[0]), torch.tensor(predicted_embeddings[0]))
    print(cosine_distance)

    all_embeddings = predict_with_triplet_network(network, torch.tensor(samples, dtype=torch.long), batch_size=32)
    nearest_neighbors = get_nearest_neighbors(torch.tensor(all_embeddings), torch.tensor(predicted_embeddings[0]), k=5)
    print(nearest_neighbors)

    similar_embeddings = get_similar_embeddings(torch.tensor(all_embeddings), torch.tensor(predicted_embeddings[0]), k=5)
    print(similar_embeddings)

if __name__ == "__main__":
    main()