import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import optax

# Data Classes

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

# Model Functions

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128, kernel_init=jax.nn.initializers.normal())(x))
        x = nn.relu(nn.Dense(128, kernel_init=jax.nn.initializers.normal())(x))
        x = nn.Dense(1000, kernel_init=jax.nn.initializers.normal())(x)
        return x

def initialize_model():
    model = Model()
    rng = jax.random.PRNGKey(42)
    input_shape = (1, 128)
    params = model.init(rng, jnp.ones(input_shape))
    return params

def calculate_loss(params, batch):
    model = Model()
    if len(batch) == 3:
        return jnp.mean(jnp.maximum((model.apply(params, batch[0]) - batch[1])**2 - (model.apply(params, batch[0]) - batch[2])**2, 0))
    else:
        return jnp.mean((model.apply(params, batch[0]) - batch[1])**2)

# Dataset

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

# Training Functions

@jax.jit
def train_step(params, batch, optimizer_state):
    loss, grads = jax.value_and_grad(lambda p: calculate_loss(p, batch))(params)
    updates, new_optimizer_state = optimizer_state.update(grads, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_optimizer_state, loss

def train_epoch(params, dataset, optimizer_state):
    batches = np.array_split(dataset, len(dataset) // 16)
    for batch in batches:
        params, optimizer_state, _ = train_step(params, batch, optimizer_state)
    return params, optimizer_state

def train(config):
    train_data, _ = load_data(config.chat_format)
    dataset = np.array(train_data["input_ids"]), np.array(train_data["labels"])
    params = initialize_model()
    optimizer_state = optax.adamw(0.001, b1=0.9, b2=0.999, eps=1e-8)
    for _ in range(config.num_epochs):
        params, optimizer_state = train_epoch(params, dataset, optimizer_state)
    return params

def main():
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    params = train(config)

if __name__ == "__main__":
    main()