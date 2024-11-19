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

# Configuration dataclass for storing model and training hyperparameters
@dataclass
class Config:
    model_id: str = "t5-base"  # Model identifier
    chat_format: str = "none"  # Chat format string
    lora_alpha: int = 16  # Low-rank adaptation alpha value
    lora_dropout: float = 0.1  # Low-rank adaptation dropout rate
    lora_rank: int = 64  # Low-rank adaptation rank
    target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"  # Target layers for adaptation
    nested_quantization: bool = False  # Nested quantization flag
    bit4_compute_type: str = "float16"  # 4-bit compute type
    bit4_quant_storage_type: str = "uint8"  # 4-bit quantization storage type
    bit4_quant_type: str = "nf4"  # 4-bit quantization type
    flash_attention: bool = False  # Flash attention flag
    peft_lora: bool = False  # PEFT LORA flag
    bit8_quantization: bool = False  # 8-bit quantization flag
    bit4_quantization: bool = False  # 4-bit quantization flag
    reentrant: bool = False  # Reentrant flag
    unsloth: bool = False  # Unsloth flag
    triplet_loss_training: bool = False  # Triplet loss training flag
    dataset_name: str = "timdettmers/openassistant-guanaco"  # Dataset name
    append_token: bool = False  # Append token flag
    add_special_tokens: bool = False  # Add special tokens flag
    data_splits: str = "train,test"  # Data splits string
    tokenized_data_path: str = None  # Tokenized data path
    output_path: str = "./results"  # Output path
    num_epochs: int = 3  # Number of training epochs
    train_batch_size: int = 16  # Training batch size
    eval_batch_size: int = 64  # Evaluation batch size
    warmup_steps: int = 500  # Warmup steps
    weight_decay: float = 0.01  # Weight decay rate
    log_path: str = "./logs"  # Log path
    save_steps: int = 500  # Save steps
    max_checkpoints: int = 2  # Maximum checkpoints
    random_seed: int = 42  # Random seed
    resume_checkpoint: str = None  # Resume checkpoint path


def load_json_data(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def prepare_data(chat_format, data):
    return {
        "input_ids": [f"{chat_format} {example['input']}" for example in data],
        "labels": [f"{chat_format} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }


def load_data(chat_format):
    train_data = load_json_data("train.json")
    test_data = load_json_data("test.json")
    return prepare_data(chat_format, train_data), prepare_data(chat_format, test_data)


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


def create_train_state(model, rng, learning_rate):
    params = model.init(rng, jnp.ones((1, 128)))['params']
    optimizer = optax.adam(learning_rate)
    return optax.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )


def calculate_loss(model, params, batch):
    if len(batch) == 3:
        return _calculate_triplet_loss(model, params, batch)
    else:
        return _calculate_standard_loss(model, params, batch)


def _calculate_standard_loss(model, params, batch):
    input_ids, labels = batch
    return jnp.mean((model.apply({'params': params}, input_ids) - labels)**2)


def _calculate_triplet_loss(model, params, batch):
    input_ids, positive_labels, negative_labels = batch
    return jnp.mean(jnp.maximum((model.apply({'params': params}, input_ids) - positive_labels)**2 - (model.apply({'params': params}, input_ids) - negative_labels)**2, 0))


def train_step(model, state, batch):
    loss, grads = jax.value_and_grad(calculate_loss, argnums=1)(model, state.params, batch)
    updates, new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def train_epoch(model, state, dataset):
    for batch in dataset:
        state, loss = train_step(model, state, batch)
    return state


def train(config):
    train_data, _ = load_data(config.chat_format)
    model = neural_network()
    rng = jax.random.PRNGKey(42)
    state = create_train_state(model, rng, 0.001)
    dataset = data_loader(train_data, config.train_batch_size, 5, config.triplet_loss_training)
    for epoch in range(config.num_epochs):
        state = train_epoch(model, state, dataset)
        print(f"Epoch {epoch+1}")


if __name__ == "__main__":
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    train(config)