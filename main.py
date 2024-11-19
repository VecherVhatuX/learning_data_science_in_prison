import os
import json
from dataclasses import dataclass
from typing import Dict
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import optax
from functools import partial

@dataclass
class Config:
    model_id: str = "t5-base"
    chat_format: str = "none"
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_rank: int = 64
    target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quantization: bool = False
    bit4_compute_type: str = "float16"
    bit4_quant_storage_type: str = "uint8"
    bit4_quant_type: str = "nf4"
    flash_attention: bool = False
    peft_lora: bool = False
    bit8_quantization: bool = False
    bit4_quantization: bool = False
    reentrant: bool = False
    unsloth: bool = False
    triplet_loss_training: bool = False
    dataset_name: str = "timdettmers/openassistant-guanaco"
    append_token: bool = False
    add_special_tokens: bool = False
    data_splits: str = "train,test"
    tokenized_data_path: str = None
    output_path: str = "./results"
    num_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    log_path: str = "./logs"
    save_steps: int = 500
    max_checkpoints: int = 2
    random_seed: int = 42
    resume_checkpoint: str = None

class DataLoader:
    def __init__(self, data, batch_size, negative_samples, triplet_mode):
        self.data = data
        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.triplet_mode = triplet_mode
        self.indices = np.arange(len(data["input_ids"]))

    def __iter__(self):
        np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            yield self._create_batch(self.indices[i:i + self.batch_size])

    def _create_batch(self, batch_indices):
        if not self.triplet_mode:
            return self._create_standard_batch(batch_indices)
        else:
            return self._create_triplet_batch(batch_indices)

    def _create_standard_batch(self, batch_indices):
        return (
            jnp.array([self.data["input_ids"][idx] for idx in batch_indices]),
            jnp.array([self.data["labels"][idx] for idx in batch_indices]),
        )

    def _create_triplet_batch(self, batch_indices):
        positive_indices = np.random.choice(batch_indices, size=self.batch_size)
        negative_indices = np.random.choice(self.indices, size=(self.batch_size, self.negative_samples), replace=False)
        return (
            jnp.array([self.data["input_ids"][idx] for idx in batch_indices]),
            jnp.array([self.data["labels"][idx] for idx in positive_indices]),
            jnp.array([[self.data["labels"][idx] for idx in sample] for sample in negative_indices]),
        )

class NeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(1000)(x)

class Trainer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optax.adam(learning_rate)

    def create_train_state(self, rng):
        params = self.model.init(rng, jnp.ones((1, 128)))['params']
        return optax.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer
        )

    def calculate_loss(self, params, batch):
        if len(batch) == 3:
            return self._calculate_triplet_loss(params, batch)
        else:
            return self._calculate_standard_loss(params, batch)

    def _calculate_standard_loss(self, params, batch):
        input_ids, labels = batch
        return jnp.mean((self.model.apply({'params': params}, input_ids) - labels)**2)

    def _calculate_triplet_loss(self, params, batch):
        input_ids, positive_labels, negative_labels = batch
        return jnp.mean(jnp.maximum((self.model.apply({'params': params}, input_ids) - positive_labels)**2 - (self.model.apply({'params': params}, input_ids) - negative_labels)**2, 0))

    def train_step(self, state, batch):
        loss, grads = jax.value_and_grad(self.calculate_loss, argnums=0)(state.params, batch)
        updates, new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def train_epoch(self, state, dataset):
        for batch in dataset:
            state, loss = self.train_step(state, batch)
        return state

def prepare_data(chat_format, data):
    return {
        "input_ids": [f"{chat_format} {example['input']}" for example in data],
        "labels": [f"{chat_format} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }

def load_data(chat_format):
    with open("train.json", 'r') as f:
        train_data = json.load(f)
    with open("test.json", 'r') as f:
        test_data = json.load(f)
    return prepare_data(chat_format, train_data), prepare_data(chat_format, test_data)

def train(config):
    train_data, _ = load_data(config.chat_format)
    model = NeuralNetwork()
    trainer = Trainer(model, 0.001)
    rng = jax.random.PRNGKey(42)
    state = trainer.create_train_state(rng)
    dataset = DataLoader(train_data, config.train_batch_size, 5, config.triplet_loss_training)
    for epoch in range(config.num_epochs):
        state = trainer.train_epoch(state, dataset)
        print(f"Epoch {epoch+1}")

if __name__ == "__main__":
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    train(config)