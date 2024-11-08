import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Model
class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids)

# Dataset
class TripletDataset(Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = list(range(len(samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_idx = idx
        positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])
        while positive_idx == anchor_idx:
            positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])

        negative_indices = random.sample([i for i, label in enumerate(self.labels) if label != self.labels[anchor_idx]], self.num_negatives)
        negative_indices = [i for i in negative_indices if i != anchor_idx]

        return {
            'anchor_input_ids': torch.tensor(self.samples[anchor_idx]),
            'anchor_attention_mask': torch.ones_like(torch.tensor(self.samples[anchor_idx]), dtype=torch.long),
            'positive_input_ids': torch.tensor(self.samples[positive_idx]),
            'positive_attention_mask': torch.ones_like(torch.tensor(self.samples[positive_idx]), dtype=torch.long),
            'negative_input_ids': torch.stack([torch.tensor(self.samples[i]) for i in negative_indices]),
            'negative_attention_mask': torch.ones_like(torch.stack([torch.tensor(self.samples[i]) for i in negative_indices]), dtype=torch.long)
        }

    def on_epoch_end(self):
        random.shuffle(self.indices)

    def __iter__(self):
        self.on_epoch_end()
        for idx in self.indices:
            yield self.__getitem__(idx)

# Trainer
class TripletLossTrainer:
    def __init__(self, model, device, triplet_margin=1.0, learning_rate=1e-4):
        self.model = model.to(device)
        self.triplet_margin = triplet_margin
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(1).expand_as(hidden_state).float()
        sum_embeddings = (hidden_state * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def normalize_embeddings(self, embeddings):
        return embeddings / embeddings.norm(dim=1, keepdim=True)

    def triplet_margin_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return (self.triplet_margin + (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1) - (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)).clamp(min=0).mean()

    def train_step(self, inputs):
        anchor_input_ids = inputs["anchor_input_ids"].to(self.device)
        anchor_attention_mask = inputs["anchor_attention_mask"].to(self.device)
        positive_input_ids = inputs["positive_input_ids"].to(self.device)
        positive_attention_mask = inputs["positive_attention_mask"].to(self.device)
        negative_input_ids = inputs["negative_input_ids"].to(self.device)
        negative_attention_mask = inputs["negative_attention_mask"].to(self.device)

        anchor_outputs = self.model(anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_outputs = self.model(positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = self.model(negative_input_ids, attention_mask=negative_attention_mask)

        anchor_embeddings = self.mean_pooling(anchor_outputs, anchor_attention_mask)
        positive_embeddings = self.mean_pooling(positive_outputs, positive_attention_mask)
        negative_embeddings = self.mean_pooling(negative_outputs, negative_attention_mask)

        anchor_embeddings = self.normalize_embeddings(anchor_embeddings)
        positive_embeddings = self.normalize_embeddings(positive_embeddings)
        negative_embeddings = self.normalize_embeddings(negative_embeddings)

        loss = self.triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataset, epochs, batch_size):
        device = self.device
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss
            print(f'Epoch {epoch+1}, loss: {total_loss / len(dataloader)}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def create_model(num_embeddings, embedding_dim):
    return EmbeddingModel(num_embeddings, embedding_dim)

def create_dataset(samples, labels, batch_size, num_negatives):
    return TripletDataset(samples, labels, batch_size, num_negatives)

def create_trainer(model, device, triplet_margin, learning_rate):
    return TripletLossTrainer(model, device, triplet_margin, learning_rate)

def train(trainer, dataset, epochs, batch_size):
    trainer.train(dataset, epochs, batch_size)

def save_model(trainer, path):
    trainer.save_model(path)

def load_model(trainer, path):
    trainer.load_model(path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = np.random.rand(100, 10)
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = create_model(100, 10)
    dataset = create_dataset(samples, labels, batch_size, num_negatives)
    trainer = create_trainer(model, device, 1.0, 1e-4)
    train(trainer, dataset, epochs, batch_size)
    save_model(trainer, "model.pth")

if __name__ == "__main__":
    main()