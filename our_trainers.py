import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

def reorder_data(samples):
    """Randomly rearranges the input data for sampling."""
    np.random.shuffle(samples)
    return samples

def construct_triplet_components(samples, labels, batch_size, num_negatives):
    """Generates the indices for the anchor, positive, and negative samples."""
    batch = np.random.choice(len(samples), batch_size, replace=False)
    anchor_idx = batch
    positive_idx = np.array([np.random.choice(np.where(labels == labels[anchor])[0]) for anchor in anchor_idx])
    while np.any(positive_idx == anchor_idx):
        positive_idx = np.array([np.random.choice(np.where(labels == labels[anchor])[0]) for anchor in anchor_idx])

    negative_indices = [np.random.choice(np.where(labels != labels[anchor])[0], num_negatives, replace=False) for anchor in anchor_idx]
    negative_indices = [np.setdiff1d(negative_idx, [anchor]) for anchor, negative_idx in zip(anchor_idx, negative_indices)]

    return anchor_idx, positive_idx, negative_indices

def prepare_triplet_batch(samples, labels, batch_size, num_negatives, idx):
    """Creates a batch of triplet data for training."""
    anchor_idx, positive_idx, negative_indices = construct_triplet_components(samples, labels, batch_size, num_negatives)
    return {
        'anchor_input_ids': torch.tensor(samples[anchor_idx]),
        'positive_input_ids': torch.tensor(samples[positive_idx]),
        'negative_input_ids': torch.stack([torch.tensor(samples[negative_idx]) for negative_idx in negative_indices]),
    }

class TripletDataset(data.Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        """Initializes the dataset for triplet training."""
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        """Calculates the number of batches in the dataset."""
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        """Returns a batch of triplet data."""
        return prepare_triplet_batch(self.samples, self.labels, self.batch_size, self.num_negatives, idx)

def standardize_vectors(embeddings):
    """Normalizes the input vectors to have unit length."""
    return embeddings / torch.norm(embeddings, dim=1, keepdim=True)

def evaluate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    """Calculates the triplet loss for the given embeddings."""
    return torch.mean(torch.clamp(margin + 
                                  torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1) - 
                                  torch.sum((anchor_embeddings - negative_embeddings[:, 0, :]) ** 2, dim=1), 
                                  min=0.0))

class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """Initializes the embedding model."""
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """Performs the forward pass through the embedding model."""
        x = self.embedding(x)
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        return x

def train_model(model, dataset, optimizer, margin, epochs):
    """Trains the model using the triplet loss."""
    for epoch in range(epochs):
        total_loss = 0
        for batch in data.DataLoader(dataset, batch_size=1, shuffle=True):
            optimizer.zero_grad()
            anchor_input_ids = batch["anchor_input_ids"].squeeze()
            positive_input_ids = batch["positive_input_ids"].squeeze()
            negative_input_ids = batch["negative_input_ids"].squeeze()
            anchor_embeddings = standardize_vectors(model(anchor_input_ids))
            positive_embeddings = standardize_vectors(model(positive_input_ids))
            negative_embeddings = standardize_vectors(model(negative_input_ids))
            loss = evaluate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

def persist_model(model, filename):
    """Saves the model to a file."""
    torch.save(model.state_dict(), filename)

def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = EmbeddingModel(100, 10)
    dataset = TripletDataset(reorder_data(samples.copy()), labels, batch_size, num_negatives)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    margin = 1.0

    train_model(model, dataset, optimizer, margin, epochs)
    persist_model(model, "model.pth")

if __name__ == "__main__":
    main()