import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = reorder_data(samples.copy())
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.epoch = 0

    def __len__(self):
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_indices = self._construct_triplet_components(idx)
        anchor_input_ids = torch.tensor(self.samples[anchor_idx])
        positive_input_ids = torch.tensor(self.samples[positive_idx])
        negative_input_ids = torch.stack([torch.tensor(self.samples[negative_idx]) for negative_idx in negative_indices])
        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

    def _construct_triplet_components(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        batch = np.arange(start_idx, end_idx)
        anchor_idx = batch
        positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])
        while np.any(positive_idx == anchor_idx):
            positive_idx = np.array([np.random.choice(np.where(self.labels == self.labels[anchor])[0]) for anchor in anchor_idx])

        negative_indices = [np.random.choice(np.where(self.labels != self.labels[anchor])[0], self.num_negatives, replace=False) for anchor in anchor_idx]
        negative_indices = [np.setdiff1d(negative_idx, [anchor]) for anchor, negative_idx in zip(anchor_idx, negative_indices)]
        return anchor_idx, positive_idx, negative_indices

    def __iter__(self):
        self.epoch += 1
        self.samples = reorder_data(self.samples.copy())
        return super().__iter__()

class TripletEmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(TripletEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        return x

    def get_triplet_embeddings(self, anchor_input_ids, positive_input_ids, negative_input_ids):
        anchor_embeddings = self.standardize_vectors(self.forward(anchor_input_ids))
        positive_embeddings = self.standardize_vectors(self.forward(positive_input_ids))
        negative_embeddings = self.standardize_vectors(self.forward(negative_input_ids))
        return anchor_embeddings, positive_embeddings, negative_embeddings

    def standardize_vectors(self, embeddings):
        return embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    def evaluate_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings, margin):
        return torch.mean(torch.clamp(margin + 
                                      torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1) - 
                                      torch.sum((anchor_embeddings - negative_embeddings[:, 0, :]) ** 2, dim=1), 
                                      min=0.0))


def reorder_data(samples):
    np.random.shuffle(samples)
    return samples


def train_model(model, dataset, optimizer, margin, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for batch in data.DataLoader(dataset, batch_size=1, shuffle=True):
            optimizer.zero_grad()
            anchor_input_ids = batch["anchor_input_ids"].squeeze()
            positive_input_ids = batch["positive_input_ids"].squeeze()
            negative_input_ids = batch["negative_input_ids"].squeeze()
            anchor_embeddings, positive_embeddings, negative_embeddings = model.get_triplet_embeddings(anchor_input_ids, positive_input_ids, negative_input_ids)
            loss = model.evaluate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")


def persist_model(model, filename):
    torch.save(model.state_dict(), filename)


def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletEmbeddingModel(100, 10)
    dataset = Dataset(samples, labels, batch_size, num_negatives)
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    margin = 1.0

    train_model(model, dataset, optimizer, margin, epochs)
    persist_model(model, "model.pth")


if __name__ == "__main__":
    main()