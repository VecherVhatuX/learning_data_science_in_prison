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

def data_loader(data, batch_size, negative_samples, triplet_mode):
    indices = np.arange(len(data["input_ids"]))
    def _create_batch(batch_indices):
        if not triplet_mode:
            return (
                jnp.array([data["input_ids"][idx] for idx in batch_indices]),
                jnp.array([data["labels"][idx] for idx in batch_indices]),
            )
        else:
            positive_indices = np.random.choice(batch_indices, size=batch_size)
            negative_indices = np.random.choice(indices, size=(batch_size, negative_samples), replace=False)
            return (
                jnp.array([data["input_ids"][idx] for idx in batch_indices]),
                jnp.array([data["labels"][idx] for idx in positive_indices]),
                jnp.array([[data["labels"][idx] for idx in sample] for sample in negative_indices]),
            )
    def __iter__():
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            yield _create_batch(indices[i:i + batch_size])
    return __iter__()

def neural_network():
    def __call__(x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(1000)(x)
    return __call__

def trainer(model, learning_rate):
    optimizer = optax.adam(learning_rate)
    def create_train_state(rng):
        params = model.init(rng, jnp.ones((1, 128)))['params']
        return optax.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
    def calculate_loss(params, batch):
        if len(batch) == 3:
            return _calculate_triplet_loss(params, batch)
        else:
            return _calculate_standard_loss(params, batch)
    def _calculate_standard_loss(params, batch):
        input_ids, labels = batch
        return jnp.mean((model.apply({'params': params}, input_ids) - labels)**2)
    def _calculate_triplet_loss(params, batch):
        input_ids, positive_labels, negative_labels = batch
        return jnp.mean(jnp.maximum((model.apply({'params': params}, input_ids) - positive_labels)**2 - (model.apply({'params': params}, input_ids) - negative_labels)**2, 0))
    def train_step(state, batch):
        loss, grads = jax.value_and_grad(calculate_loss, argnums=0)(state.params, batch)
        updates, new_state = state.apply_gradients(grads=grads)
        return new_state, loss
    def train_epoch(state, dataset):
        for batch in dataset:
            state, loss = train_step(state, batch)
        return state
    return create_train_state, train_epoch, train_step

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
    model = neural_network()
    create_train_state, train_epoch, _ = trainer(model, 0.001)
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng)
    dataset = data_loader(train_data, config.train_batch_size, 5, config.triplet_loss_training)
    for epoch in range(config.num_epochs):
        state = train_epoch(state, dataset)
        print(f"Epoch {epoch+1}")

if __name__ == "__main__":
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    train(config)