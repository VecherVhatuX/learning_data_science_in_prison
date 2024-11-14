import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Data Preparation
def prepare_data():
    np.random.seed(42)
    torch.manual_seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    return samples, labels, batch_size, num_negatives, epochs

# Model Initialization
def initialize_model(num_embeddings, embedding_dim, num_negatives):
    model = TripletModel(num_embeddings, embedding_dim, num_negatives)
    return model

# Loss Function Initialization
def initialize_loss_fn(margin=1.0):
    loss_fn = TripletLoss(margin)
    return loss_fn

# Optimizer Initialization
def initialize_optimizer(model, lr=1e-4):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer

# Dataset Class
class TripletDataset(Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def get_indices(self):
        return np.random.permutation(len(self.samples))

    def get_anchor_idx(self, indices, idx):
        return indices[idx * self.batch_size:(idx + 1) * self.batch_size]

    def get_positive_idx(self, anchor_idx):
        positive_idx = []
        for anchor in anchor_idx:
            idx = torch.where(self.labels == self.labels[anchor])[0]
            positive_idx.append(torch.randint(0, len(idx[idx != anchor]), (1,)).item())
        return positive_idx

    def get_negative_idx(self, anchor_idx):
        negative_indices = []
        for anchor in anchor_idx:
            idx = torch.where(self.labels != self.labels[anchor])[0]
            negative_idx = torch.randperm(len(idx))[:self.num_negatives]
            negative_indices.extend(negative_idx)
        return negative_indices

    def get_data(self, anchor_idx, positive_idx, negative_idx):
        anchor_input_ids = self.samples[anchor_idx]
        positive_input_ids = self.samples[torch.tensor(positive_idx, dtype=torch.long)]
        negative_input_ids = self.samples[torch.tensor(negative_idx, dtype=torch.long)].numpy().reshape(self.batch_size, self.num_negatives, -1)
        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

    def __getitem__(self, idx):
        indices = self.get_indices()
        anchor_idx = self.get_anchor_idx(indices, idx)
        positive_idx = self.get_positive_idx(anchor_idx)
        negative_idx = self.get_negative_idx(anchor_idx)
        return self.get_data(anchor_idx, positive_idx, negative_idx)

# Model Class
class TripletModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_negatives):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.num_negatives = num_negatives

    def normalize_embeddings(self, embeddings):
        return embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    def embed(self, input_ids):
        embeddings = self.embedding(input_ids)
        embeddings = self.pooling(embeddings.permute(0, 2, 1)).squeeze()
        return self.normalize_embeddings(embeddings)

    def embed_negative(self, negative_input_ids):
        return self.embed(torch.tensor(negative_input_ids, dtype=torch.long).view(-1, negative_input_ids.shape[2]))

    def forward(self, inputs):
        anchor_input_ids = inputs['anchor_input_ids']
        positive_input_ids = inputs['positive_input_ids']
        negative_input_ids = inputs['negative_input_ids']

        anchor_embeddings = self.embed(anchor_input_ids)
        positive_embeddings = self.embed(positive_input_ids)
        negative_embeddings = self.embed_negative(negative_input_ids)
        return anchor_embeddings, positive_embeddings, negative_embeddings

# Loss Function Class
class TripletLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def calculate_loss(self, anchor, positive, negative):
        return torch.clamp(torch.norm(anchor - positive, dim=1) - torch.norm(anchor.unsqueeze(1) - negative, dim=2) + self.margin, min=0.0)

    def __call__(self, anchor, positive, negative):
        return torch.mean(self.calculate_loss(anchor, positive, negative))

# Trainer Class
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        self.optimizer.zero_grad()
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0.0

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i, data in enumerate(dataset):
                loss = self.train_step(data)
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

# Evaluator Class
class Evaluator:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def evaluate_step(self, data):
        with torch.no_grad():
            anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
            if len(positive_embeddings) > 0:
                loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                return loss.item()
            return 0.0

    def evaluate(self, dataset):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            loss = self.evaluate_step(data)
            total_loss += loss
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

# Predictor Class
class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_ids):
        return self.model.embed(input_ids)

# Main Function
def main():
    samples, labels, batch_size, num_negatives, epochs = prepare_data()
    model = initialize_model(101, 10, num_negatives)
    loss_fn = initialize_loss_fn()
    optimizer = initialize_optimizer(model)
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    data_loader = DataLoader(dataset, batch_size=1)

    trainer = Trainer(model, optimizer, loss_fn)
    evaluator = Evaluator(model, loss_fn)

    trainer.train(data_loader, epochs)
    evaluator.evaluate(data_loader)

    torch.save(model.state_dict(), 'model.pth')

    predictor = Predictor(model)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    output = predictor.predict(input_ids)
    print(output)

if __name__ == "__main__":
    main()