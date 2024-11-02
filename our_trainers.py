import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from typing import Optional, Callable
import random

class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

def create_optimizer(model: nn.Module, learning_rate: float) -> SGD:
    return SGD(model.parameters(), lr=learning_rate)

def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(1).expand_as(hidden_state).float()
    sum_embeddings = (hidden_state * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=1, keepdim=True)

def triplet_margin_loss(
    anchor_embeddings: torch.Tensor, 
    positive_embeddings: torch.Tensor, 
    negative_embeddings: torch.Tensor, 
    margin: float = 1.0
) -> torch.Tensor:
    return (
        margin + 
        (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1) - 
        (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)
    ).clamp(min=0).mean()

class Dataset:
    def __init__(self, samples: np.ndarray, labels: np.ndarray, batch_size: int):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.indices = list(range(len(samples)))

    def shuffle(self) -> None:
        random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        return self.get_item(idx, 5)

    def __len__(self) -> int:
        return len(self.samples)

    def get_item(self, idx: int, num_negatives: int) -> dict:
        anchor_idx = idx
        positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])
        while positive_idx == anchor_idx:
            positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])

        negative_indices = random.sample([i for i, label in enumerate(self.labels) if label != self.labels[anchor_idx]], num_negatives)
        negative_indices = [i for i in negative_indices if i != anchor_idx]

        return {
            'anchor_input_ids': torch.tensor(self.samples[anchor_idx]),
            'anchor_attention_mask': torch.ones_like(torch.tensor(self.samples[anchor_idx]), dtype=torch.long),
            'positive_input_ids': torch.tensor(self.samples[positive_idx]),
            'positive_attention_mask': torch.ones_like(torch.tensor(self.samples[positive_idx]), dtype=torch.long),
            'negative_input_ids': torch.stack([torch.tensor(self.samples[i]) for i in negative_indices]),
            'negative_attention_mask': torch.ones_like(torch.stack([torch.tensor(self.samples[i]) for i in negative_indices]), dtype=torch.long)
        }

    def get_len(self) -> int:
        return len(self.samples)

class TripletLossTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 triplet_margin: float = 1.0, 
                 layer_index: int = -1,
                 learning_rate: float = 1e-4):
        self.model = model
        self.triplet_margin = triplet_margin
        self.layer_index = layer_index
        self.optimizer = create_optimizer(model, learning_rate)

    def _get_triplet_embeddings(
        self, 
        anchor_input_ids: torch.Tensor, 
        anchor_attention_mask: torch.Tensor, 
        positive_input_ids: torch.Tensor, 
        positive_attention_mask: torch.Tensor, 
        negative_input_ids: torch.Tensor, 
        negative_attention_mask: torch.Tensor
    ) -> tuple:
        anchor_outputs = self.model(anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_outputs = self.model(positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = self.model(negative_input_ids, attention_mask=negative_attention_mask)

        anchor_hidden_state = anchor_outputs if self.layer_index == -1 else anchor_outputs.hidden_states[self.layer_index]
        positive_hidden_state = positive_outputs if self.layer_index == -1 else positive_outputs.hidden_states[self.layer_index]
        negative_hidden_state = negative_outputs if self.layer_index == -1 else negative_outputs.hidden_states[self.layer_index]

        anchor_embeddings = mean_pooling(anchor_hidden_state, anchor_attention_mask)
        positive_embeddings = mean_pooling(positive_hidden_state, positive_attention_mask)
        negative_embeddings = mean_pooling(negative_hidden_state, negative_attention_mask)

        anchor_embeddings = normalize_embeddings(anchor_embeddings)
        positive_embeddings = normalize_embeddings(positive_embeddings)
        negative_embeddings = normalize_embeddings(negative_embeddings)

        return anchor_embeddings, positive_embeddings, negative_embeddings

    def _compute_loss(self, inputs: dict) -> torch.Tensor:
        anchor_input_ids = inputs["anchor_input_ids"]
        anchor_attention_mask = inputs["anchor_attention_mask"]
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        negative_input_ids = inputs["negative_input_ids"]
        negative_attention_mask = inputs["negative_attention_mask"]

        anchor_embeddings, positive_embeddings, negative_embeddings = self._get_triplet_embeddings(
            anchor_input_ids, 
            anchor_attention_mask, 
            positive_input_ids, 
            positive_attention_mask, 
            negative_input_ids, 
            negative_attention_mask
        )

        return triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

    def train(self, dataset: Dataset, epochs: int, num_negatives: int) -> None:
        for epoch in range(epochs):
            dataset.shuffle()
            total_loss = 0
            for i in range(len(dataset)):
                inputs = dataset[i]
                loss = self.train_step(inputs)
                total_loss += loss
            print(f'Epoch {epoch+1}, loss: {total_loss / len(dataset)}')

    def train_step(self, inputs: dict) -> float:
        self.optimizer.zero_grad()
        loss = self._compute_loss(inputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == "__main__":
    dataset = Dataset(np.random.rand(100, 10), np.random.randint(0, 2, 100), 32)
    model = EmbeddingModel(100, 10)
    trainer = TripletLossTrainer(model, triplet_margin=1.0, layer_index=-1, learning_rate=1e-4)
    trainer.train(dataset, epochs=10, num_negatives=5)