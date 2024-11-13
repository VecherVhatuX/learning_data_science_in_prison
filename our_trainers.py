import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

class TripletData(data.Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        """
        Initialize the triplet dataset with given parameters.

        :param samples: Input data samples
        :param labels: Labels corresponding to the samples
        :param batch_size: Batch size for data loading
        :param num_negatives: Number of negative samples per anchor
        """
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = np.arange(len(samples))

    def __len__(self):
        """
        Calculate the number of batches.

        :return: Number of batches
        """
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        """
        Get a batch of triplet data.

        :param idx: Batch index
        :return: A dictionary containing anchor, positive, and negative input IDs
        """
        np.random.shuffle(self.indices)
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        batch = self.indices[start_idx:end_idx]
        anchor_idx = batch
        positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])
        while np.any(positive_idx == anchor_idx):
            positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])

        negative_indices = [np.random.choice(np.where(self.labels != self.labels[anchor])[0], self.num_negatives, replace=False) for anchor in anchor_idx]
        negative_indices = [np.setdiff1d(negative_idx, [anchor]) for anchor, negative_idx in zip(anchor_idx, negative_indices)]
        anchor_input_ids = torch.tensor(self.samples[anchor_idx])
        positive_input_ids = torch.tensor(self.samples[positive_idx])
        negative_input_ids = torch.stack([torch.tensor(self.samples[negative_idx]) for negative_idx in negative_indices])
        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

class TripletLossModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initialize the triplet loss model.

        :param num_embeddings: Number of embeddings
        :param embedding_dim: Dimension of the embeddings
        """
        super(TripletLossModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input data
        :return: Embedded representation
        """
        x = self.embedding(x)
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        return x

    def standardize_vectors(self, embeddings):
        """
        Standardize the vectors to have unit norm.

        :param embeddings: Embedded vectors
        :return: Standardized vectors
        """
        return embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    def triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings, margin):
        """
        Calculate the triplet loss.

        :param anchor_embeddings: Anchor embeddings
        :param positive_embeddings: Positive embeddings
        :param negative_embeddings: Negative embeddings
        :param margin: Margin value
        :return: Triplet loss
        """
        return torch.mean(torch.clamp(margin + 
                                      torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1) - 
                                      torch.sum((anchor_embeddings - negative_embeddings[:, 0, :]) ** 2, dim=1), 
                                      min=0.0))

def train_model(model, dataset, optimizer, margin, epochs):
    """
    Train the model.

    :param model: Model instance
    :param dataset: Dataset instance
    :param optimizer: Optimizer instance
    :param margin: Margin value
    :param epochs: Number of epochs
    """
    for epoch in range(epochs):
        total_loss = 0
        data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
        for batch in data_loader:
            optimizer.zero_grad()
            anchor_input_ids = batch["anchor_input_ids"].squeeze()
            positive_input_ids = batch["positive_input_ids"].squeeze()
            negative_input_ids = batch["negative_input_ids"].squeeze()
            anchor_embeddings = model.standardize_vectors(model.forward(anchor_input_ids))
            positive_embeddings = model.standardize_vectors(model.forward(positive_input_ids))
            negative_embeddings = model.standardize_vectors(model.forward(negative_input_ids))
            loss = model.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

def persist_model(model, filename):
    """
    Save the model.

    :param model: Model instance
    :param filename: Filename to save the model
    """
    torch.save(model.state_dict(), filename)

def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletLossModel(100, 10)
    dataset = TripletData(samples, labels, batch_size, num_negatives)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    margin = 1.0

    train_model(model, dataset, optimizer, margin, epochs)
    persist_model(model, "model.pth")

if __name__ == "__main__":
    main()