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

def prepare_data(chat_format, data):
    return {
        "input_ids": [f"{chat_format} {example['input']}" for example in data],
        "labels": [f"{chat_format} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }

def load_data(chat_format):
    train_data = prepare_data(chat_format, json.load(open("train.json", 'r')))
    test_data = prepare_data(chat_format, json.load(open("test.json", 'r')))
    return train_data, test_data

def create_dataset(data, batch_size, negative_samples, triplet_mode):
    indices = np.arange(len(data["input_ids"]))
    while True:
        np.random.shuffle(indices)
        for _ in range(len(indices) // batch_size):
            batch_indices = indices[_ * batch_size:(_ + 1) * batch_size]
            batch = {k: jnp.array([data[k][idx] for idx in batch_indices]) for k in ["input_ids"]}
            if triplet_mode:
                positive_indices = np.random.choice(batch_indices, size=batch_size)
                negative_indices = np.random.choice(indices, size=(batch_size, negative_samples), replace=False)
                batch["positive_labels"] = jnp.array([data["labels"][idx] for idx in positive_indices])
                batch["negative_labels"] = jnp.array([[data["labels"][idx] for idx in sample] for sample in negative_indices])
            else:
                batch["labels"] = jnp.array([data["labels"][idx] for idx in batch_indices])
            yield batch

def create_train_state(rng, model, learning_rate):
    return optax.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng, jnp.ones((1, 128)))['params'],
        tx=optax.adam(learning_rate)
    )

def calculate_loss(params, batch, triplet_mode, model):
    if triplet_mode:
        return jnp.mean(jnp.maximum((model.apply({'params': params}, batch["input_ids"]) - batch["positive_labels"])**2 - (model.apply({'params': params}, batch["input_ids"]) - batch["negative_labels"])**2, 0))
    else:
        return jnp.mean((model.apply({'params': params}, batch["input_ids"]) - batch["labels"])**2)

def execute_train_step(state, batch, triplet_mode, model):
    grads = jax.grad(calculate_loss, argnums=0)(state.params, batch, triplet_mode, model)
    return state.apply_gradients(grads=grads)

def train_epoch(model, state, dataset, triplet_mode):
    for batch in dataset:
        state = execute_train_step(state, batch, triplet_mode, model)
    return state

class NeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x

def train(config):
    train_data, test_data = load_data(config.chat_format)
    train_dataset = create_dataset(train_data, config.train_batch_size, 5, config.triplet_loss_training)
    test_dataset = create_dataset(test_data, config.eval_batch_size, 5, config.triplet_loss_training)
    model = NeuralNetwork()
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model, 0.001)
    for epoch in range(config.num_epochs):
        state = train_epoch(model, state, train_dataset, config.triplet_loss_training)
        print(f"Epoch {epoch+1}")

if __name__ == "__main__":
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    train(config)