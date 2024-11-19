import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import optax

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
    train_data = load_json_data("train.json")
    test_data = load_json_data("test.json")
    return prepare_data(chat_format, train_data), prepare_data(chat_format, test_data)

class Dataset:
    def __init__(self, data: Dict, batch_size: int, negative_samples: int, triplet_mode: bool):
        self.data = data
        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.triplet_mode = triplet_mode
        self.indices = np.arange(len(data["input_ids"]))

    def __iter__(self):
        np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if not self.triplet_mode:
                yield (
                    jnp.array([self.data["input_ids"][idx] for idx in batch_indices]),
                    jnp.array([self.data["labels"][idx] for idx in batch_indices]),
                )
            else:
                positive_indices = np.random.choice(batch_indices, size=self.batch_size)
                negative_indices = np.random.choice(self.indices, size=(self.batch_size, self.negative_samples), replace=False)
                yield (
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

def create_train_state(model: NeuralNetwork, rng: jax.random.PRNGKey, learning_rate: float) -> optax.TrainState:
    params = model.init(rng, jnp.ones((1, 128)))['params']
    optimizer = optax.adam(learning_rate)
    return optax.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

def calculate_loss(model: NeuralNetwork, params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    if len(batch) == 3:
        return _calculate_triplet_loss(model, params, batch)
    else:
        return _calculate_standard_loss(model, params, batch)

def _calculate_standard_loss(model: NeuralNetwork, params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    input_ids, labels = batch
    return jnp.mean((model.apply({'params': params}, input_ids) - labels)**2)

def _calculate_triplet_loss(model: NeuralNetwork, params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    input_ids, positive_labels, negative_labels = batch
    return jnp.mean(jnp.maximum((model.apply({'params': params}, input_ids) - positive_labels)**2 - (model.apply({'params': params}, input_ids) - negative_labels)**2, 0))

def train_step(model: NeuralNetwork, state: optax.TrainState, batch: Tuple[jnp.ndarray, ...]) -> Tuple[optax.TrainState, jnp.ndarray]:
    loss, grads = jax.value_and_grad(calculate_loss, argnums=1)(model, state.params, batch)
    updates, new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train_epoch(model: NeuralNetwork, state: optax.TrainState, dataset: Dataset) -> optax.TrainState:
    for batch in dataset:
        state, loss = train_step(model, state, batch)
    return state

def train(config: Config):
    train_data, _ = load_data(config.chat_format)
    model = NeuralNetwork()
    rng = jax.random.PRNGKey(42)
    state = create_train_state(model, rng, 0.001)
    dataset = Dataset(train_data, config.train_batch_size, 5, config.triplet_loss_training)
    for epoch in range(config.num_epochs):
        state = train_epoch(model, state, dataset)
        print(f"Epoch {epoch+1}")

if __name__ == "__main__":
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    train(config)