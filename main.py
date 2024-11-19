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

def calculate_loss(model: callable, params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    if len(batch) == 3:
        return _calculate_triplet_loss(model, params, batch)
    else:
        return _calculate_standard_loss(model, params, batch)

def _calculate_standard_loss(model: callable, params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    input_ids, labels = batch
    return jnp.mean((model(params, input_ids) - labels)**2)

def _calculate_triplet_loss(model: callable, params: Dict, batch: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    input_ids, positive_labels, negative_labels = batch
    return jnp.mean(jnp.maximum((model(params, input_ids) - positive_labels)**2 - (model(params, input_ids) - negative_labels)**2, 0))

def train_step(model: callable, state: optax.TrainState, batch: Tuple[jnp.ndarray, ...]) -> Tuple[optax.TrainState, jnp.ndarray]:
    loss, grads = jax.value_and_grad(lambda params: calculate_loss(model, params, batch))(state.params)
    updates, new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train_epoch(model: callable, state: optax.TrainState, dataset: callable) -> optax.TrainState:
    for batch in dataset():
        state, _ = train_step(model, state, batch)
    return state

def create_neural_network(key: jax.random.PRNGKey) -> callable:
    init_params = nn.initializers.zeros()
    @jax.jit
    def neural_network(params, x):
        x = jax.nn.relu(jnp.matmul(x, params['layer1']))
        x = jax.nn.relu(jnp.matmul(x, params['layer2']))
        return jnp.matmul(x, params['layer3'])
    key1, key2, key3 = jax.random.split(key, 3)
    layer1 = jax.random.normal(key1, (128, 128))
    layer2 = jax.random.normal(key2, (128, 128))
    layer3 = jax.random.normal(key3, (128, 1000))
    return neural_network, {'layer1': layer1, 'layer2': layer2, 'layer3': layer3}

def create_train_state(model: callable, params: Dict, rng: jax.random.PRNGKey, learning_rate: float) -> optax.TrainState:
    optimizer = optax.adam(learning_rate)
    return optax.TrainState.create(
        apply_fn=model,
        params=params,
        tx=optimizer
    )

def train(config: Config):
    train_data, _ = load_data(config.chat_format)
    rng = jax.random.PRNGKey(42)
    model, params = create_neural_network(rng)
    state = create_train_state(model, params, rng, 0.001)
    dataset = create_dataset(train_data, config.train_batch_size, 5, config.triplet_loss_training)
    for epoch in range(config.num_epochs):
        state = train_epoch(model, state, dataset)
        print(f"Epoch {epoch+1}")

if __name__ == "__main__":
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    train(config)