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

def initialize_model():
    return {
        'layer1': jax.random.normal(jax.random.PRNGKey(42), (128, 128)),
        'layer2': jax.random.normal(jax.random.PRNGKey(43), (128, 128)),
        'layer3': jax.random.normal(jax.random.PRNGKey(44), (128, 1000)),
    }

def model(params, x):
    return jnp.matmul(jnp.matmul(jax.nn.relu(jnp.matmul(x, params['layer1'])), params['layer2']), params['layer3'])

def calculate_loss(params, batch):
    if len(batch) == 3:
        return jnp.mean(jnp.maximum((model(params, batch[0]) - batch[1])**2 - (model(params, batch[0]) - batch[2])**2, 0))
    else:
        return jnp.mean((model(params, batch[0]) - batch[1])**2)

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

def update_params(params, grads, learning_rate):
    return {k: v - learning_rate * g for k, v, g in zip(params.keys(), params.values(), grads.values())}

def train_step(params, batch):
    loss, grads = jax.value_and_grad(lambda p: calculate_loss(p, batch))(params)
    return update_params(params, grads, 0.001), loss

def train_epoch(params, dataset):
    return jax.jit(lambda params, dataset: params, static_argnums=(1,))(params, dataset)

def map_over_batches(params, dataset):
    def train_batch(params, batch):
        return train_step(params, batch)
    return jax.jit(lambda params, dataset: params, static_argnums=(1,))(params, dataset)

def map_over_epochs(params, dataset, num_epochs):
    def train_epoch(params, dataset):
        return map_over_batches(params, dataset)
    def train_loop(carry, _):
        params, dataset = carry
        params = train_epoch(params, dataset)
        return (params, dataset), None
    return jax.jit(lambda params, dataset, num_epochs: params, static_argnums=(2,))(params, dataset, num_epochs)

def train(config):
    train_data, _ = load_data(config.chat_format)
    dataset = np.array(train_data["input_ids"]), np.array(train_data["labels"])
    params = initialize_model()
    params = map_over_epochs(params, dataset, config.num_epochs)
    return params

def main():
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    params = train(config)

if __name__ == "__main__":
    main()