import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional, Union, Dict, Any, Callable
from trl import SFTTrainer
from transformers.trainer_utils import PredictionOutput
from torch.nn.functional import normalize
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, num_negatives, batch_size):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size

    def __getitem__(self, idx):
        anchor_idx = idx
        positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])
        while positive_idx == anchor_idx:
            positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])

        negative_indices = random.sample([i for i, label in enumerate(self.labels) if label != self.labels[anchor_idx]], self.num_negatives)
        negative_indices = [i for i in negative_indices if i != anchor_idx]

        return {
            'anchor_input_ids': self.samples[anchor_idx],
            'anchor_attention_mask': torch.ones_like(self.samples[anchor_idx], dtype=torch.long),
            'positive_input_ids': self.samples[positive_idx],
            'positive_attention_mask': torch.ones_like(self.samples[positive_idx], dtype=torch.long),
            'negative_input_ids': torch.stack([self.samples[i] for i in negative_indices]),
            'negative_attention_mask': torch.ones_like(torch.stack([self.samples[i] for i in negative_indices]), dtype=torch.long)
        }

    def __len__(self):
        return len(self.samples)

class TripletLossTrainer(Trainer):
    def __init__(self, 
                 triplet_margin: float = 1.0, 
                 triplet_loss_fn: Optional[Callable] = None,
                 layer_index=-1,
                 **kwargs):
        super().__init__(**kwargs)
        self.triplet_margin = triplet_margin
        self.triplet_loss_fn = triplet_loss_fn or nn.TripletMarginLoss(margin=triplet_margin)
        self.layer_index = layer_index
        
    
    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def compute_loss(self, model, inputs, return_outputs=False):
        anchor_input_ids = inputs["anchor_input_ids"]
        anchor_attention_mask = inputs["anchor_attention_mask"]
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        negative_input_ids = inputs["negative_input_ids"]
        negative_attention_mask = inputs["negative_attention_mask"]
        
        anchor_outputs = model(input_ids=anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_outputs = model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)
        
        anchor_hidden_state = anchor_outputs.hidden_states[self.layer_index]
        positive_hidden_state = positive_outputs.hidden_states[self.layer_index]
        negative_hidden_state = negative_outputs.hidden_states[self.layer_index]

        anchor_embeddings = self.mean_pooling(anchor_hidden_state, anchor_attention_mask)
        positive_embeddings = self.mean_pooling(positive_hidden_state, positive_attention_mask)
        negative_embeddings = self.mean_pooling(negative_hidden_state, negative_attention_mask)
        
        anchor_embeddings = normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = normalize(negative_embeddings, p=2, dim=1)
        
        loss = self.triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        if return_outputs:
            return (loss, {
                'anchor_embeddings': anchor_embeddings, 
                'positive_embeddings': positive_embeddings, 
                'negative_embeddings': negative_embeddings
            })
        else:
            return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, predictions = self.compute_loss(model, inputs, return_outputs=True)
        dummy_labels = torch.zeros(inputs["anchor_input_ids"].size(0), device=inputs["anchor_input_ids"].device)
        return (loss, predictions, dummy_labels)