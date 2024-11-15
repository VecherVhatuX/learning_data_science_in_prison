import os
import json
from dataclasses import dataclass
from typing import Dict
import jax
import jax.numpy as jnp
from flax import linen as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import optax


@dataclass
class ModelConfig:
    """Model configuration class."""
    model_identifier: str = "t5-base"  # Identifier for the model
    chat_template: str = "none"  # Template for chat
    lora_alpha: int = 16  # Alpha value for LoRA
    lora_dropout: float = 0.1  # Dropout value for LoRA
    lora_rank: int = 64  # Rank value for LoRA
    lora_target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"  # Target layers for LoRA
    nested_quant: bool = False  # Flag for nested quantization
    bnb_4bit_compute_dtype: str = "float16"  # Compute data type for 4-bit BNB
    bnb_4bit_quant_storage_dtype: str = "uint8"  # Storage data type for 4-bit BNB quantization
    bnb_4bit_quant_type: str = "nf4"  # Type of 4-bit BNB quantization
    use_flash_attention: bool = False  # Flag for flash attention
    use_peft_lora: bool = False  # Flag for PEFT LoRA
    use_8bit_quantization: bool = False  # Flag for 8-bit quantization
    use_4bit_quantization: bool = False  # Flag for 4-bit quantization
    use_reentrant: bool = False  # Flag for reentrant
    use_unsloth: bool = False  # Flag for unsloth
    use_triplet_loss_trainer: bool = False  # Flag for triplet loss trainer


@dataclass
class TrainingDataConfig:
    """Training data configuration class."""
    dataset_name: str = "timdettmers/openassistant-guanaco"  # Name of the dataset
    append_concat_token: bool = False  # Flag for appending concat token
    add_special_tokens: bool = False  # Flag for adding special tokens
    splits: str = "train,test"  # Splits of the dataset
    tokenized_dataset_path: str = None  # Path to tokenized dataset


@dataclass
class TrainingConfig:
    """Training configuration class."""
    output_dir: str = "./results"  # Output directory
    num_train_epochs: int = 3  # Number of training epochs
    per_device_train_batch_size: int = 16  # Batch size per device for training
    per_device_eval_batch_size: int = 64  # Batch size per device for evaluation
    warmup_steps: int = 500  # Warmup steps
    weight_decay: float = 0.01  # Weight decay
    logging_dir: str = "./logs"  # Logging directory
    save_steps: int = 500  # Save steps
    save_total_limit: int = 2  # Save total limit
    seed: int = 42  # Seed for randomness
    resume_from_checkpoint: str = None  # Path to resume from checkpoint


def load_json_file(file_name):
    """Loads a JSON file."""
    with open(file_name, 'r') as f:
        return json.load(f)


def prepare_data(data_args, chat_template, data):
    """Prepares data for training."""
    return {
        "input_ids": [f"{chat_template} {example['input']}" for example in data],
        "labels": [f"{chat_template} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }


def prepare_triplet_data(data_args, chat_template, data, indices, use_triplet):
    """Prepares triplet data for training."""
    def _prepare_triplet_data(idx):
        if use_triplet:
            positive_labels = data["labels"][indices[idx]]
            negative_labels_idx = np.random.randint(0, len(data["labels"]))
            while negative_labels_idx == indices[idx]:
                negative_labels_idx = np.random.randint(0, len(data["labels"]))
            negative_labels = data["labels"][negative_labels_idx]
            return {"input_ids": data["input_ids"][indices[idx]], "positive_labels": positive_labels, "negative_labels": negative_labels}
        return {"input_ids": data["input_ids"][indices[idx]], "labels": data["labels"][indices[idx]]}
    return _prepare_triplet_data


def create_train_state(rng, model, learning_rate):
    """Creates a training state."""
    params = model.init(rng, jnp.ones((1, 128)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx,
    )


def calculate_loss(params, batch, use_triplet):
    """Calculates the loss."""
    model = Model()
    if use_triplet:
        outputs = model.apply({'params': params}, batch["input_ids"])
        loss = (outputs - batch["positive_labels"])**2 - (outputs - batch["negative_labels"])**2
    else:
        labels = batch["labels"]
        inputs = batch["input_ids"]
        outputs = model.apply({'params': params}, inputs)
        loss = (outputs - labels)**2
    return jnp.mean(loss)


def execute_train_step(state, batch, use_triplet):
    """Executes a training step."""
    grads = jax.grad(calculate_loss, argnums=0)(state.params, batch, use_triplet)
    state = state.apply_gradients(grads=grads)
    return state


class Model(nn.Module):
    """Model class."""
    @nn.compact
    def __call__(self, x):
        """Calls the model."""
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x


def train_epoch(model, state, dataset, use_triplet):
    """Trains the model for an epoch."""
    for batch in dataset:
        state = execute_train_step(state, batch, use_triplet)
    return state


def run_pipeline(model_args, data_args, training_args):
    """Runs the pipeline."""
    train_data = load_json_file("train.json")
    test_data = load_json_file("test.json")
    chat_template = data_args.chat_template if data_args.chat_template != "none" else ""
    train_dataset = prepare_data(data_args, chat_template, train_data)
    test_dataset = prepare_data(data_args, chat_template, test_data)
    
    train_dataset = Dataset(train_dataset, training_args.per_device_train_batch_size, model_args.use_triplet_loss_trainer)
    test_dataset = Dataset(test_dataset, training_args.per_device_eval_batch_size, model_args.use_triplet_loss_trainer)

    model = Model()
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model, 0.001)

    for epoch in range(training_args.num_train_epochs):
        state = train_epoch(model, state, train_dataset.get_triplet_data() if model_args.use_triplet_loss_trainer else train_dataset.get_data(), model_args.use_triplet_loss_trainer)
        print(f"Epoch {epoch+1}")


class Dataset:
    """Dataset class."""
    def __init__(self, data, batch_size, use_triplet):
        """Initializes the dataset."""
        self.data = data
        self.batch_size = batch_size
        self.use_triplet = use_triplet
        self.indices = list(range(len(self.data["input_ids"])))

    def shuffle(self):
        """Shuffles the dataset."""
        np.random.seed(42)
        self.indices = np.random.permutation(self.indices)

    def get_triplet_data(self):
        """Gets triplet data."""
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = {}
            for idx in batch_indices:
                if self.use_triplet:
                    positive_labels = self.data["labels"][idx]
                    negative_labels_idx = np.random.randint(0, len(self.data["labels"]))
                    while negative_labels_idx == idx:
                        negative_labels_idx = np.random.randint(0, len(self.data["labels"]))
                    negative_labels = self.data["labels"][negative_labels_idx]
                    batch_data[idx] = {"input_ids": self.data["input_ids"][idx], "positive_labels": positive_labels, "negative_labels": negative_labels}
                else:
                    batch_data[idx] = {"input_ids": self.data["input_ids"][idx], "labels": self.data["labels"][idx]}
            yield {k: np.array([v[k] for v in batch_data.values()]) for k in batch_data[batch_indices[0]].keys()}

    def get_data(self):
        """Gets data."""
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = {k: np.array([self.data[k][idx] for idx in batch_indices]) for k in self.data.keys()}
            yield batch_data


def main():
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none", use_triplet_loss_trainer=True)
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    run_pipeline(model_args, data_args, training_args)


if __name__ == "__main__":
    main()