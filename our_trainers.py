import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict
from jax.experimental import jax2tf
from jax.experimental.jax2tf import lower
import numpy as np
from typing import Optional, Callable

def create_optimizer(learning_rate: float):
    return jax.experimental.optimizers.sgd(learning_rate)

def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = attention_mask[:, None].expand(hidden_state.shape).astype(jnp.float32)
    sum_embeddings = jnp.sum(hidden_state * input_mask_expanded, axis=1)
    sum_mask = jnp.clip(jnp.sum(input_mask_expanded, axis=1), a_min=1e-9)
    return sum_embeddings / sum_mask

def triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    return jnp.mean(jnp.maximum(margin + jnp.linalg.norm(anchor_embeddings - positive_embeddings, axis=1) - jnp.linalg.norm(anchor_embeddings - negative_embeddings, axis=1), 0))

def compute_loss(params, inputs, model, layer_index=-1):
    anchor_input_ids = inputs["anchor_input_ids"]
    anchor_attention_mask = inputs["anchor_attention_mask"]
    positive_input_ids = inputs["positive_input_ids"]
    positive_attention_mask = inputs["positive_attention_mask"]
    negative_input_ids = inputs["negative_input_ids"]
    negative_attention_mask = inputs["negative_attention_mask"]

    anchor_outputs = model.apply({'params': params}, input_ids=anchor_input_ids, attention_mask=anchor_attention_mask)
    positive_outputs = model.apply({'params': params}, input_ids=positive_input_ids, attention_mask=positive_attention_mask)
    negative_outputs = model.apply({'params': params}, input_ids=negative_input_ids, attention_mask=negative_attention_mask)

    anchor_hidden_state = anchor_outputs.hidden_states[layer_index]
    positive_hidden_state = positive_outputs.hidden_states[layer_index]
    negative_hidden_state = negative_outputs.hidden_states[layer_index]

    anchor_embeddings = mean_pooling(anchor_hidden_state, anchor_attention_mask)
    positive_embeddings = mean_pooling(positive_hidden_state, positive_attention_mask)
    negative_embeddings = mean_pooling(negative_hidden_state, negative_attention_mask)

    anchor_embeddings = jax.nn.normalize(anchor_embeddings, axis=1)
    positive_embeddings = jax.nn.normalize(positive_embeddings, axis=1)
    negative_embeddings = jax.nn.normalize(negative_embeddings, axis=1)

    return triplet_margin_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

def get_item(dataset, idx, num_negatives):
    anchor_idx = idx
    positive_idx = np.random.choice([i for i, label in enumerate(dataset.labels) if label == dataset.labels[anchor_idx]])
    while positive_idx == anchor_idx:
        positive_idx = np.random.choice([i for i, label in enumerate(dataset.labels) if label == dataset.labels[anchor_idx]])

    negative_indices = np.random.choice([i for i, label in enumerate(dataset.labels) if label != dataset.labels[anchor_idx]], num_negatives, replace=False)
    negative_indices = [i for i in negative_indices if i != anchor_idx]

    return {
        'anchor_input_ids': dataset.samples[anchor_idx],
        'anchor_attention_mask': np.ones_like(dataset.samples[anchor_idx], dtype=np.long),
        'positive_input_ids': dataset.samples[positive_idx],
        'positive_attention_mask': np.ones_like(dataset.samples[positive_idx], dtype=np.long),
        'negative_input_ids': np.stack([dataset.samples[i] for i in negative_indices]),
        'negative_attention_mask': np.ones_like(np.stack([dataset.samples[i] for i in negative_indices]), dtype=np.long)
    }

def get_len(dataset):
    return len(dataset.samples)

class Dataset:
    def __init__(self, samples, labels, batch_size):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size

def train_step(state, inputs, model, layer_index, optimizer):
    grads = jax.grad(lambda params: compute_loss(params, inputs, model, layer_index))(state.params)
    state = state.apply_gradients(grads=grads)
    return state

def train(model, dataset, epochs, num_negatives, layer_index, optimizer):
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))['params'],
        tx=optimizer
    )
    for epoch in range(epochs):
        for i in range(get_len(dataset)):
            inputs = get_item(dataset, i, num_negatives)
            state = train_step(state, inputs, model, layer_index, optimizer)
        print(f'Epoch {epoch+1}, loss: {compute_loss(state.params, inputs, model, layer_index)}')
    return state

class TripletLossTrainer:
    def __init__(self, 
                 model, 
                 triplet_margin=1.0, 
                 layer_index=-1,
                 learning_rate=1e-4):
        self.model = model
        self.triplet_margin = triplet_margin
        self.layer_index = layer_index
        self.optimizer = create_optimizer(learning_rate)

    def train(self, dataset, epochs, num_negatives):
        state = train(self.model, dataset, epochs, num_negatives, self.layer_index, self.optimizer)
        return state

if __name__ == "__main__":
    # Initialize dataset
    dataset = Dataset(np.random.rand(100, 10), np.random.randint(0, 2, 100), 32)

    # Initialize model and trainer
    model = nn.Embed(100, 10)
    trainer = TripletLossTrainer(model, triplet_margin=1.0, layer_index=-1, learning_rate=1e-4)

    # Train model
    trainer.train(dataset, epochs=10, num_negatives=5)