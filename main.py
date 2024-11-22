import os
import json
from dataclasses import dataclass
from typing import List
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, FanOut, FanInConcat
import optax

@dataclass
class Hyperparameters:
    base_model_identifier: str = "t5-base"
    conversation_format_identifier: str = "none"
    low_rank_approximation_alpha: int = 16
    low_rank_approximation_dropout_rate: float = 0.1
    low_rank_approximation_rank: int = 64
    target_model_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quantization_enabled: bool = False
    four_bit_computation_data_type: str = "float16"
    four_bit_quantization_storage_data_type: str = "uint8"
    four_bit_quantization_type: str = "nf4"
    flash_attention_enabled: bool = False
    peft_low_rank_approximation_enabled: bool = False
    eight_bit_quantization_enabled: bool = False
    four_bit_quantization_enabled: bool = False
    reentrant_training_enabled: bool = False
    unsloth_training_enabled: bool = False
    triplet_loss_training_enabled: bool = True
    dataset_identifier: str = "timdettmers/openassistant-guanaco"
    append_special_token: bool = False
    add_special_tokens: bool = False
    dataset_splits: str = "train,test"
    tokenized_data_path: str = None
    output_directory_path: str = "./results"
    number_of_epochs: int = 3
    training_batch_size: int = 16
    evaluation_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay_rate: float = 0.01
    log_directory_path: str = "./logs"
    save_steps: int = 500
    maximum_checkpoints: int = 2
    random_seed_value: int = 42
    resume_checkpoint_path: str = None
    negative_samples_per_positive_sample: int = 5

class CustomDataset:
    def __init__(self, data, conversation_format_identifier):
        self.data = data
        self.conversation_format_identifier = conversation_format_identifier

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_ids = jnp.array([0] + [ord(c) for c in f"{self.conversation_format_identifier} {example['input']}"] + [1], dtype=jnp.float32)
        labels = jnp.array([0] + [ord(c) for c in f"{self.conversation_format_identifier} {example['output']}"] + [1], dtype=jnp.float32)
        attention_mask = jnp.array([1] * len(input_ids), dtype=jnp.float32)
        return input_ids, labels, attention_mask

def neural_network_model():
    init_fn, apply_fn = stax.serial(
        Dense(128, W_init=jax.nn.initializers.normal(1.0)),
        Relu(),
        Dense(128, W_init=jax.nn.initializers.normal(1.0)),
        Relu(),
        Dense(1000, W_init=jax.nn.initializers.normal(1.0))
    )
    return init_fn, apply_fn

def train_model(model, dataset, hyperparameters):
    init_fn, apply_fn = model
    rng = jax.random.PRNGKey(hyperparameters.random_seed_value)
    params = init_fn(rng, (-1, 128))
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    loss_fn = lambda params, x, y: jnp.mean((apply_fn(params, x) - y) ** 2)
    for epoch in range(hyperparameters.number_of_epochs):
        total_loss = 0
        for i, (batch_inputs, batch_labels, _) in enumerate(dataset):
            grads = jax.grad(loss_fn)(params, batch_inputs, batch_labels)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            total_loss += loss_fn(params, batch_inputs, batch_labels)
        print(f"Epoch {epoch+1}, Loss: {total_loss / (i+1)}")
    jnp.save(os.path.join(hyperparameters.output_directory_path, "final_model.npy"), params)

def evaluate_model(model, dataset, hyperparameters):
    init_fn, apply_fn = model
    params = jnp.load(os.path.join(hyperparameters.output_directory_path, "final_model.npy"))
    total_loss = 0
    for batch_inputs, batch_labels, _ in dataset:
        total_loss += jnp.mean((apply_fn(params, batch_inputs) - batch_labels) ** 2)
    print(f"Test Loss: {total_loss / len(list(dataset))}")

def load_json_data(file_name):
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{file_name} not found.")
        return None

def load_hyperparameters(base_model_identifier, conversation_format_identifier, triplet_loss_training_enabled):
    return Hyperparameters(base_model_identifier=base_model_identifier, conversation_format_identifier=conversation_format_identifier, triplet_loss_training_enabled=triplet_loss_training_enabled)

def main():
    hyperparameters = load_hyperparameters("t5-base", "none", True)
    model = neural_network_model()
    training_data, testing_data = load_json_data("train.json"), load_json_data("test.json")
    if training_data is not None and testing_data is not None:
        batch_size = hyperparameters.training_batch_size
        train_data_loader = jax.tree_util.tree_leaves(CustomDataset(training_data, hyperparameters.conversation_format_identifier))
        train_data_loader = jax.tree_util.tree_map(lambda x: x.reshape(-1, batch_size, x.shape[-1]), train_data_loader)
        train_data_loader = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), train_data_loader)

        test_data_loader = jax.tree_util.tree_leaves(CustomDataset(testing_data, hyperparameters.conversation_format_identifier))
        test_data_loader = jax.tree_util.tree_map(lambda x: x.reshape(-1, batch_size, x.shape[-1]), test_data_loader)
        test_data_loader = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), test_data_loader)

        train_model(model, zip(*train_data_loader), hyperparameters)
        evaluate_model(model, zip(*test_data_loader), hyperparameters)

if __name__ == "__main__":
    main()