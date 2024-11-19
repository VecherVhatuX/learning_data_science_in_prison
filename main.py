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

# Data Loading Functions

def load_json_data(file_name: str) -> Dict:
    with open(file_name, 'r') as f:
        return json.load(f)

def prepare_data(chat_format: str, data: Dict) -> Dict:
    return {
        "input_ids": [f"{chat_format} {example['input']}" for example in data],
        "labels": [f"{chat_format} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }

def load_data(chat_format: str) -> Tuple[Dict, Dict]:
    return (
        prepare_data(chat_format, load_json_data("train.json")),
        prepare_data(chat_format, load_json_data("test.json"))
    )

# Dataset Creation Functions

def create_dataset(data: Dict, batch_size: int, negative_samples: int, triplet_mode: bool) -> callable:
    indices = np.arange(len(data["input_ids"]))
    def dataset():
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if not triplet_mode:
                yield (
                    jnp.array([data["input_ids"][idx] for idx in batch_indices]),
                    jnp.array([data["labels"][idx] for idx in batch_indices]),
                )
            else:
                positive_indices = np.random.choice(batch_indices, size=batch_size)
                negative_indices = np.random.choice(indices, size=(batch_size, negative_samples), replace=False)
                yield (
                    jnp.array([data["input_ids"][idx] for idx in batch_indices]),
                    jnp.array([data["labels"][idx] for idx in positive_indices]),
                    jnp.array([[data["labels"][idx] for idx in sample] for sample in negative_indices]),
                )
    return dataset

# Model Functions

def model(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(jnp.matmul(jax.nn.relu(jnp.matmul(x, params['layer1'])), params['layer2']), params['layer3'])

def calculate_loss(params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    if len(batch) == 3:
        return jnp.mean(jnp.maximum((model(params, batch[0]) - batch[1])**2 - (model(params, batch[0]) - batch[2])**2, 0))
    else:
        return jnp.mean((model(params, batch[0]) - batch[1])**2)

# Training Functions

def update_params(params: Dict, grads: Dict, learning_rate: float) -> Dict:
    return {k: v - learning_rate * g for k, v, g in zip(params.keys(), params.values(), grads.values())}

def train_step(params: Dict, batch: Tuple[jnp.ndarray, ...]) -> Tuple[Dict, jnp.ndarray]:
    loss, grads = jax.value_and_grad(lambda p: calculate_loss(p, batch))(params)
    return update_params(params, grads, 0.001), loss

def train_epoch(params: Dict, dataset: callable) -> Dict:
    return reduce(lambda params, batch: train_step(params, batch)[0], dataset(), params)

# Main Training Function

def train(config: Config):
    train_data, _ = load_data(config.chat_format)
    params = {
        'layer1': jax.random.normal(jax.random.PRNGKey(42), (128, 128)),
        'layer2': jax.random.normal(jax.random.PRNGKey(43), (128, 128)),
        'layer3': jax.random.normal(jax.random.PRNGKey(44), (128, 1000)),
    }
    dataset = create_dataset(train_data, config.train_batch_size, 5, config.triplet_loss_training)
    for epoch in range(config.num_epochs):
        params = train_epoch(params, dataset)
        print(f"Epoch {epoch+1}")

if __name__ == "__main__":
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    train(config)