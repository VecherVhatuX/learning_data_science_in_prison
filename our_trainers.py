import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

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

class TripletLoss:
    def __init__(self, margin=1.0):
        self.margin = margin

    def calculate_loss(self, anchor, positive, negative):
        return torch.clamp(torch.norm(anchor - positive, dim=1) - torch.norm(anchor.unsqueeze(1) - negative, dim=2) + self.margin, min=0.0)

    def __call__(self, anchor, positive, negative):
        return torch.mean(self.calculate_loss(anchor, positive, negative))

class TripletTrainer:
    def __init__(self, model, optimizer, loss_fn, epochs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs

    def train_step(self, data):
        self.optimizer.zero_grad()
        anchor_embeddings, positive_embeddings, negative_embeddings = self.model(data)
        if len(positive_embeddings) > 0:
            loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0.0

    def train(self, dataset):
        for epoch in range(self.epochs):
            total_loss = 0
            for i, data in enumerate(dataset):
                loss = self.train_step(data)
                total_loss += loss
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

class TripletEvaluator:
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

class TripletPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, input_ids):
        return self.model.embed(input_ids)

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    data_loader = DataLoader(dataset, batch_size=1)

    model = TripletModel(num_embeddings, embedding_dim, num_negatives)
    loss_fn = TripletLoss(margin)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    trainer = TripletTrainer(model, optimizer, loss_fn, epochs)
    trainer.train(data_loader)

    evaluator = TripletEvaluator(model, loss_fn)
    evaluator.evaluate(data_loader)

    torch.save(model.state_dict(), 'model.pth')

    predictor = TripletPredictor(model)
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    output = predictor.predict(input_ids)
    print(output)

if __name__ == "__main__":
    main()