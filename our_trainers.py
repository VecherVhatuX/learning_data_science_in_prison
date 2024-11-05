import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

class TripletDataset(Dataset):
    def __init__(self, samples: np.ndarray, labels: np.ndarray, batch_size: int, num_negatives: int):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = list(range(len(samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
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

class TripletLossTrainer:
    def __init__(self, model: nn.Module, device: str, triplet_margin: float = 1.0, learning_rate: float = 1e-4):
        self.model = model.to(device)
        self.triplet_margin = triplet_margin
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def mean_pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(1).expand_as(hidden_state).float()
        sum_embeddings = (hidden_state * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(dim=1, keepdim=True)

    def triplet_margin_loss(self, anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor) -> torch.Tensor:
        return (self.triplet_margin + (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1) - (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)).clamp(min=0).mean()

    def train_step(self, inputs: dict) -> float:
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

    def train(self, dataset: TripletDataset, epochs: int, batch_size: int) -> None:
        device = self.device
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                loss = self.train_step(batch)
                total_loss += loss
            print(f'Epoch {epoch+1}, loss: {total_loss / len(dataloader)}')

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TripletDataset(np.random.rand(100, 10), np.random.randint(0, 2, 100), 32, 5)
    model = EmbeddingModel(100, 10)
    trainer = TripletLossTrainer(model, device, triplet_margin=1.0, learning_rate=1e-4)
    trainer.train(dataset, epochs=10, batch_size=32)
    trainer.save_model("model.pth")

if __name__ == "__main__":
    main()