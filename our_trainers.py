import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def create_triplet_network(num_embeddings, embedding_dim, margin):
    return nn.Sequential(
        nn.Embedding(num_embeddings, embedding_dim),
        lambda x: x.permute(0, 2, 1),
        nn.AdaptiveAvgPool1d((1,)),
        lambda x: x.squeeze(2),
        nn.Linear(embedding_dim, embedding_dim),
        nn.BatchNorm1d(embedding_dim),
        lambda x: x / torch.norm(x, dim=1, keepdim=True),
    )

def create_triplet_loss(margin):
    def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
        return torch.mean(torch.clamp(
            torch.norm(anchor_embeddings - positive_embeddings, dim=1) 
            - torch.norm(anchor_embeddings.unsqueeze(1) - negative_embeddings, dim=2).min(dim=1)[0] + margin, min=0
        ))
    return triplet_loss

def create_triplet_dataset(samples, labels, num_negatives):
    def __getitem__(idx):
        np.random.shuffle(np.arange(len(samples)))
        idx = np.arange(len(samples))[idx]
        anchor_idx = idx
        anchor_label = labels[idx]

        positive_idx = np.random.choice(np.where(labels == anchor_label)[0], size=1)[0]
        negative_idx = np.random.choice(np.where(labels != anchor_label)[0], size=num_negatives, replace=False)

        return {
            'anchor_input_ids': torch.tensor(samples[anchor_idx], dtype=torch.long),
            'positive_input_ids': torch.tensor(samples[positive_idx], dtype=torch.long),
            'negative_input_ids': torch.tensor(samples[negative_idx], dtype=torch.long)
        }

    return __getitem__

def create_epoch_shuffle_dataset(dataset):
    indices = np.arange(len(dataset))

    def __getitem__(idx):
        np.random.shuffle(indices)
        return dataset(indices[idx])

    def on_epoch_end():
        np.random.shuffle(indices)

    return __getitem__, on_epoch_end

def create_samples_and_labels():
    np.random.seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    return samples, labels

def train_triplet_network(network, dataset, epochs, learning_rate, batch_size):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    triplet_loss = create_triplet_loss(1.0)

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
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
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
    triplet_loss = create_triplet_loss(1.0)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            anchor_input_ids = data['anchor_input_ids'].to(device)
            positive_input_ids = data['positive_input_ids'].to(device)
            negative_input_ids = data['negative_input_ids'].to(device)

            anchor_embeddings = network(anchor_input_ids)
            positive_embeddings = network(positive_input_ids)
            negative_embeddings = network(negative_input_ids)
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
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
    samples, labels = create_samples_and_labels()
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    learning_rate = 1e-4

    network = create_triplet_network(num_embeddings, embedding_dim, margin)
    dataset = create_triplet_dataset(samples, labels, num_negatives)
    train_triplet_network(network, dataset, epochs, learning_rate, batch_size)
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long).unsqueeze(0)
    output = predict_with_triplet_network(network, input_ids, batch_size=1)
    print(output)
    save_triplet_model(network, "triplet_model.pth")
    loaded_network = create_triplet_network(num_embeddings, embedding_dim, margin)
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