import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from typing import Optional, Callable

def create_optimizer(model, learning_rate):
    return SGD(model.parameters(), lr=learning_rate)

def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(1).expand_as(hidden_state).float()
    sum_embeddings = (hidden_state * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def normalize_embeddings(embeddings):
    return embeddings / embeddings.norm(dim=1, keepdim=True)

def triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    return (margin + (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1) - (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)).clamp(min=0).mean()

def get_triplet_embeddings(model, anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask, layer_index=-1):
    anchor_outputs = model(anchor_input_ids, attention_mask=anchor_attention_mask)
    positive_outputs = model(positive_input_ids, attention_mask=positive_attention_mask)
    negative_outputs = model(negative_input_ids, attention_mask=negative_attention_mask)

    anchor_hidden_state = anchor_outputs.last_hidden_state if layer_index == -1 else anchor_outputs.hidden_states[layer_index]
    positive_hidden_state = positive_outputs.last_hidden_state if layer_index == -1 else positive_outputs.hidden_states[layer_index]
    negative_hidden_state = negative_outputs.last_hidden_state if layer_index == -1 else negative_outputs.hidden_states[layer_index]

    anchor_embeddings = mean_pooling(anchor_hidden_state, anchor_attention_mask)
    positive_embeddings = mean_pooling(positive_hidden_state, positive_attention_mask)
    negative_embeddings = mean_pooling(negative_hidden_state, negative_attention_mask)

    anchor_embeddings = normalize_embeddings(anchor_embeddings)
    positive_embeddings = normalize_embeddings(positive_embeddings)
    negative_embeddings = normalize_embeddings(negative_embeddings)

    return anchor_embeddings, positive_embeddings, negative_embeddings

def compute_loss(model, inputs, layer_index):
    anchor_input_ids = inputs["anchor_input_ids"]
    anchor_attention_mask = inputs["anchor_attention_mask"]
    positive_input_ids = inputs["positive_input_ids"]
    positive_attention_mask = inputs["positive_attention_mask"]
    negative_input_ids = inputs["negative_input_ids"]
    negative_attention_mask = inputs["negative_attention_mask"]

    anchor_embeddings, positive_embeddings, negative_embeddings = get_triplet_embeddings(model, anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask, layer_index)

    return triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

def get_item(dataset, idx, num_negatives):
    anchor_idx = idx
    positive_idx = np.random.choice([i for i, label in enumerate(dataset.labels) if label == dataset.labels[anchor_idx]])
    while positive_idx == anchor_idx:
        positive_idx = np.random.choice([i for i, label in enumerate(dataset.labels) if label == dataset.labels[anchor_idx]])

    negative_indices = np.random.choice([i for i, label in enumerate(dataset.labels) if label != dataset.labels[anchor_idx]], num_negatives, replace=False)
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

class Dataset:
    def __init__(self, samples, labels, batch_size):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size

def train_step(model, inputs, optimizer, layer_index):
    optimizer.zero_grad()
    loss = compute_loss(model, inputs, layer_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model, dataset, epochs, num_negatives, layer_index, optimizer):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(get_len(dataset)):
            inputs = get_item(dataset, i, num_negatives)
            loss = train_step(model, inputs, optimizer, layer_index)
            total_loss += loss
        print(f'Epoch {epoch+1}, loss: {total_loss / get_len(dataset)}')

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

    def train(self, dataset, epochs, num_negatives):
        train(self.model, dataset, epochs, num_negatives, self.layer_index, self.optimizer)

if __name__ == "__main__":
    # Initialize dataset
    dataset = Dataset(np.random.rand(100, 10), np.random.randint(0, 2, 100), 32)

    # Initialize model and trainer
    model = nn.Embedding(100, 10)
    model = model.to(torch.float)
    trainer = TripletLossTrainer(model, triplet_margin=1.0, layer_index=-1, learning_rate=1e-4)

    # Train model
    trainer.train(dataset, epochs=10, num_negatives=5)