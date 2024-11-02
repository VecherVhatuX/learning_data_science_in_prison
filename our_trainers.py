import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from typing import Optional, Callable
import random

# Model-related functions
class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids)

def create_optimizer(model, learning_rate):
    return SGD(model.parameters(), lr=learning_rate)

# Embedding-related functions
def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(1).expand_as(hidden_state).float()
    sum_embeddings = (hidden_state * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def normalize_embeddings(embeddings):
    return embeddings / embeddings.norm(dim=1, keepdim=True)

# Loss function
def triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    return (margin + (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1) - (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)).clamp(min=0).mean()

# Dataset-related functions
class Dataset:
    def __init__(self, samples, labels, batch_size):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.indices = list(range(len(samples)))

    def shuffle(self):
        random.shuffle(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return get_item(self, idx, 5)

    def __len__(self):
        return len(self.samples)

def get_item(dataset, idx, num_negatives):
    anchor_idx = idx
    positive_idx = random.choice([i for i, label in enumerate(dataset.labels) if label == dataset.labels[anchor_idx]])
    while positive_idx == anchor_idx:
        positive_idx = random.choice([i for i, label in enumerate(dataset.labels) if label == dataset.labels[anchor_idx]])

    negative_indices = random.sample([i for i, label in enumerate(dataset.labels) if label != dataset.labels[anchor_idx]], num_negatives)
    negative_indices = [i for i in negative_indices if i != anchor_idx]

    return {
        'anchor_input_ids': torch.tensor(dataset.samples[anchor_idx]),
        'anchor_attention_mask': torch.ones_like(torch.tensor(dataset.samples[anchor_idx]), dtype=torch.long),
        'positive_input_ids': torch.tensor(dataset.samples[positive_idx]),
        'positive_attention_mask': torch.ones_like(torch.tensor(dataset.samples[positive_idx]), dtype=torch.long),
        'negative_input_ids': torch.stack([torch.tensor(dataset.samples[i]) for i in negative_indices]),
        'negative_attention_mask': torch.ones_like(torch.stack([torch.tensor(dataset.samples[i]) for i in negative_indices]), dtype=torch.long)
    }

def get_len(dataset):
    return len(dataset.samples)

# Trainer-related functions
class TripletLossTrainer:
    def __init__(self, 
                 model, 
                 triplet_margin=1.0, 
                 layer_index=-1,
                 learning_rate=1e-4):
        self.model = model
        self.triplet_margin = triplet_margin
        self.layer_index = layer_index
        self.optimizer = create_optimizer(model, learning_rate)

    def _get_triplet_embeddings(self, anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask):
        anchor_outputs = self.model(anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_outputs = self.model(positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = self.model(negative_input_ids, attention_mask=negative_attention_mask)

        anchor_hidden_state = anchor_outputs.last_hidden_state if self.layer_index == -1 else anchor_outputs.hidden_states[self.layer_index]
        positive_hidden_state = positive_outputs.last_hidden_state if self.layer_index == -1 else positive_outputs.hidden_states[self.layer_index]
        negative_hidden_state = negative_outputs.last_hidden_state if self.layer_index == -1 else negative_outputs.hidden_states[self.layer_index]

        anchor_embeddings = mean_pooling(anchor_hidden_state, anchor_attention_mask)
        positive_embeddings = mean_pooling(positive_hidden_state, positive_attention_mask)
        negative_embeddings = mean_pooling(negative_hidden_state, negative_attention_mask)

        anchor_embeddings = normalize_embeddings(anchor_embeddings)
        positive_embeddings = normalize_embeddings(positive_embeddings)
        negative_embeddings = normalize_embeddings(negative_embeddings)

        return anchor_embeddings, positive_embeddings, negative_embeddings

    def _compute_loss(self, inputs):
        anchor_input_ids = inputs["anchor_input_ids"]
        anchor_attention_mask = inputs["anchor_attention_mask"]
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        negative_input_ids = inputs["negative_input_ids"]
        negative_attention_mask = inputs["negative_attention_mask"]

        anchor_embeddings, positive_embeddings, negative_embeddings = self._get_triplet_embeddings(anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask)

        return triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

    def train(self, dataset, epochs, num_negatives):
        for epoch in range(epochs):
            dataset.shuffle()
            total_loss = 0
            for i in range(len(dataset)):
                inputs = dataset[i]
                loss = self.train_step(inputs)
                total_loss += loss
            print(f'Epoch {epoch+1}, loss: {total_loss / len(dataset)}')

    def train_step(self, inputs):
        self.optimizer.zero_grad()
        loss = self._compute_loss(inputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def main():
    dataset = Dataset(np.random.rand(100, 10), np.random.randint(0, 2, 100), 32)
    model = EmbeddingModel(100, 10)
    trainer = TripletLossTrainer(model, triplet_margin=1.0, layer_index=-1, learning_rate=1e-4)
    trainer.train(dataset, epochs=10, num_negatives=5)

if __name__ == "__main__":
    main()