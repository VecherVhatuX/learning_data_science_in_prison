import os
import json
from dataclasses import dataclass
from typing import Dict
import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
from jax.experimental.jax2tf import lower
from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

@dataclass
class ModelConfig:
    model_identifier: str = "t5-base"
    chat_template: str = "none"
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_rank: int = 64
    lora_target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quant: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_storage_dtype: str = "uint8"
    bnb_4bit_quant_type: str = "nf4"
    use_flash_attention: bool = False
    use_peft_lora: bool = False
    use_8bit_quantization: bool = False
    use_4bit_quantization: bool = False
    use_reentrant: bool = False
    use_unsloth: bool = False
    use_triplet_loss_trainer: bool = False

@dataclass
class TrainingDataConfig:
    dataset_name: str = "timdettmers/openassistant-guanaco"
    append_concat_token: bool = False
    add_special_tokens: bool = False
    splits: str = "train,test"
    tokenized_dataset_path: str = None

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_dir: str = "./logs"
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    resume_from_checkpoint: str = None

class Dataset:
    def __init__(self, data_args, dataset, use_triplet):
        self.data_args = data_args
        self.dataset = self._prepare_data(dataset)
        self.use_triplet = use_triplet

    def _prepare_data(self, data):
        return {
            "input_ids": np.array([f"{example['input']} " if self.data_args.chat_template != "none" else example["input"] for example in data]),
            "labels": np.array([f"{example['output']} " if self.data_args.chat_template != "none" else example["output"] for example in data]),
            "attention_mask": np.ones(len(data))
        }

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, idx):
        if self.use_triplet:
            positive_labels = self.dataset["labels"][idx]
            negative_labels = np.random.choice(self.dataset["labels"], 1, replace=False)[0]
            return {"input_ids": self.dataset["input_ids"][idx], "positive_labels": positive_labels, "negative_labels": negative_labels}
        return {"input_ids": self.dataset["input_ids"][idx], "labels": self.dataset["labels"][idx]}

    @staticmethod
    def load_json_file(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    @classmethod
    def prepare(cls, data_args):
        train_data = cls.load_json_file("train.json")
        test_data = cls.load_json_file("test.json")
        return cls(data_args, train_data, data_args.use_triplet_loss_trainer), cls(data_args, test_data, data_args.use_triplet_loss_trainer)

class T5Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x

    @staticmethod
    def get_loss_fn(use_triplet):
        def triplet_loss_fn(x, y, z):
            return (x - y)**2 - (x - z)**2
        def mse_loss_fn(x, y):
            return (x - y)**2
        return triplet_loss_fn if use_triplet else mse_loss_fn

    def train_step(self, batch, loss_fn, optimizer):
        if "positive_labels" in batch:
            outputs = self(batch["input_ids"])
            loss = loss_fn(outputs, batch["positive_labels"], batch["negative_labels"])
        else:
            labels = batch["labels"]
            outputs = self(batch["input_ids"])
            loss = loss_fn(outputs, labels)
        grads = jax.grad(loss, self)
        optimizer = optimizer.apply_gradient(optimizer, grads, self)
        return loss

    def train(self, data_loader, num_epochs, loss_fn, optimizer):
        for epoch in range(num_epochs):
            for batch in data_loader:
                loss = self.train_step(batch, loss_fn, optimizer)
            print(f"Epoch {epoch+1}, Loss: {loss}")

def run_pipeline(model_args, data_args, training_args):
    train_dataset, _ = Dataset.prepare(model_args)
    data_loader = keras.preprocessing.sequence.TimeseriesGenerator(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    model = T5Model()
    loss_fn = model.get_loss_fn(model_args.use_triplet_loss_trainer)
    optimizer = jax.experimental.optimizers.adam(0.001)
    model.train(data_loader, training_args.num_train_epochs, loss_fn, optimizer)

def resume_pipeline(model_args, data_args, training_args, checkpoint_path):
    train_dataset, _ = Dataset.prepare(model_args)
    data_loader = keras.preprocessing.sequence.TimeseriesGenerator(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    model = T5Model()
    model.load_state_dict(checkpoint_path)
    loss_fn = model.get_loss_fn(model_args.use_triplet_loss_trainer)
    optimizer = jax.experimental.optimizers.adam(0.001)
    model.train(data_loader, training_args.num_train_epochs, loss_fn, optimizer)

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none", use_triplet_loss_trainer=True)
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    run_pipeline(model_args, data_args, training_args)