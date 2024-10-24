import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict
from jax.experimental.jax2tf import lower
import numpy as np
from typing import Optional, Callable

class Dataset:
    def __init__(self, samples: np.ndarray, labels: np.ndarray, num_negatives: int, batch_size: int):
        """
        Initialize the dataset with samples, labels, number of negatives, and batch size.
        
        Args:
        samples (np.ndarray): Array of samples.
        labels (np.ndarray): Array of labels corresponding to the samples.
        num_negatives (int): Number of negative samples to generate.
        batch_size (int): Size of each batch.
        """
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size

    def __getitem__(self, idx: int):
        """
        Generate a batch of anchor, positive, and negative samples.
        
        Args:
        idx (int): Index of the anchor sample.
        
        Returns:
        dict: Dictionary containing anchor input IDs, anchor attention mask, positive input IDs, positive attention mask, negative input IDs, and negative attention mask.
        """
        anchor_idx = idx
        positive_idx = np.random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])
        while positive_idx == anchor_idx:
            positive_idx = np.random.choice([i for i, label in enumerate(self.labels) if label == self.labels[anchor_idx]])

        negative_indices = np.random.choice([i for i, label in enumerate(self.labels) if label != self.labels[anchor_idx]], self.num_negatives, replace=False)
        negative_indices = [i for i in negative_indices if i != anchor_idx]

        return {
            'anchor_input_ids': self.samples[anchor_idx],
            'anchor_attention_mask': np.ones_like(self.samples[anchor_idx], dtype=np.long),
            'positive_input_ids': self.samples[positive_idx],
            'positive_attention_mask': np.ones_like(self.samples[positive_idx], dtype=np.long),
            'negative_input_ids': np.stack([self.samples[i] for i in negative_indices]),
            'negative_attention_mask': np.ones_like(np.stack([self.samples[i] for i in negative_indices]), dtype=np.long)
        }

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        
        Returns:
        int: Length of the dataset.
        """
        return len(self.samples)


class TripletLossTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 triplet_margin: float = 1.0, 
                 triplet_loss_fn: Optional[Callable] = None,
                 layer_index: int = -1,
                 learning_rate: float = 1e-4):
        """
        Initialize the trainer with a model, triplet margin, triplet loss function, layer index, and learning rate.
        
        Args:
        model (nn.Module): Model to be trained.
        triplet_margin (float, optional): Triplet margin. Defaults to 1.0.
        triplet_loss_fn (Optional[Callable], optional): Triplet loss function. Defaults to None.
        layer_index (int, optional): Index of the layer to extract embeddings from. Defaults to -1.
        learning_rate (float, optional): Learning rate. Defaults to 1e-4.
        """
        self.model = model
        self.triplet_margin = triplet_margin
        self.triplet_loss_fn = triplet_loss_fn or self.triplet_margin_loss
        self.layer_index = layer_index
        
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))['params'],
            tx=self.sgd(learning_rate)
        )
    
    def mean_pooling(self, hidden_state: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Perform mean pooling on the hidden state using the attention mask.
        
        Args:
        hidden_state (jnp.ndarray): Hidden state of the model.
        attention_mask (jnp.ndarray): Attention mask.
        
        Returns:
        jnp.ndarray: Pooled hidden state.
        """
        input_mask_expanded = attention_mask[:, None].expand(hidden_state.shape).astype(jnp.float32)
        sum_embeddings = jnp.sum(hidden_state * input_mask_expanded, axis=1)
        sum_mask = jnp.clip(jnp.sum(input_mask_expanded, axis=1), a_min=1e-9)
        return sum_embeddings / sum_mask
    
    def compute_loss(self, params: FrozenDict, inputs: dict) -> jnp.ndarray:
        """
        Compute the triplet loss.
        
        Args:
        params (FrozenDict): Model parameters.
        inputs (dict): Dictionary containing anchor input IDs, anchor attention mask, positive input IDs, positive attention mask, negative input IDs, and negative attention mask.
        
        Returns:
        jnp.ndarray: Triplet loss.
        """
        anchor_input_ids = inputs["anchor_input_ids"]
        anchor_attention_mask = inputs["anchor_attention_mask"]
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]
        negative_input_ids = inputs["negative_input_ids"]
        negative_attention_mask = inputs["negative_attention_mask"]
        
        anchor_outputs = self.model.apply({'params': params}, input_ids=anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_outputs = self.model.apply({'params': params}, input_ids=positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = self.model.apply({'params': params}, input_ids=negative_input_ids, attention_mask=negative_attention_mask)
        
        anchor_hidden_state = anchor_outputs.hidden_states[self.layer_index]
        positive_hidden_state = positive_outputs.hidden_states[self.layer_index]
        negative_hidden_state = negative_outputs.hidden_states[self.layer_index]

        anchor_embeddings = self.mean_pooling(anchor_hidden_state, anchor_attention_mask)
        positive_embeddings = self.mean_pooling(positive_hidden_state, positive_attention_mask)
        negative_embeddings = self.mean_pooling(negative_hidden_state, negative_attention_mask)
        
        anchor_embeddings = jax.nn.normalize(anchor_embeddings, axis=1)
        positive_embeddings = jax.nn.normalize(positive_embeddings, axis=1)
        negative_embeddings = jax.nn.normalize(negative_embeddings, axis=1)
        
        loss = self.triplet_loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        return loss
    
    def triplet_margin_loss(self, anchor_embeddings: jnp.ndarray, positive_embeddings: jnp.ndarray, negative_embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the triplet margin loss.
        
        Args:
        anchor_embeddings (jnp.ndarray): Anchor embeddings.
        positive_embeddings (jnp.ndarray): Positive embeddings.
        negative_embeddings (jnp.ndarray): Negative embeddings.
        
        Returns:
        jnp.ndarray: Triplet margin loss.
        """
        return jnp.mean(jnp.maximum(self.triplet_margin + jnp.linalg.norm(anchor_embeddings - positive_embeddings, axis=1) - jnp.linalg.norm(anchor_embeddings - negative_embeddings, axis=1), 0))

    def sgd(self, learning_rate: float) -> jax.experimental.optimizers.Optimizer:
        """
        Create an SGD optimizer.
        
        Args:
        learning_rate (float): Learning rate.
        
        Returns:
        jax.experimental.optimizers.Optimizer: SGD optimizer.
        """
        return jax.experimental.optimizers.sgd(learning_rate)

    def train_step(self, state: train_state.TrainState, inputs: dict) -> train_state.TrainState:
        """
        Perform a training step.
        
        Args:
        state (train_state.TrainState): Current state of the model.
        inputs (dict): Dictionary containing anchor input IDs, anchor attention mask, positive input IDs, positive attention mask, negative input IDs, and negative attention mask.
        
        Returns:
        train_state.TrainState: Updated state of the model.
        """
        grads = jax.grad(self.compute_loss)(state.params, inputs)
        state = state.apply_gradients(grads=grads)
        return state

    def train(self, dataset: Dataset, epochs: int):
        """
        Train the model.
        
        Args:
        dataset (Dataset): Dataset to train on.
        epochs (int): Number of epochs to train for.
        """
        for epoch in range(epochs):
            for i in range(len(dataset)):
                inputs = dataset[i]
                self.state = self.train_step(self.state, inputs)
            print(f'Epoch {epoch+1}, loss: {self.compute_loss(self.state.params, inputs)}')