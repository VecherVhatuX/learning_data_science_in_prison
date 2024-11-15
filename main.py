import os
import json
from dataclasses import dataclass
from typing import Dict
import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
from jax.experimental.jax2tf import convert as convert_tf
from flax import linen as nn
from flax.training import train_state
from flax.jax_utils import replicate
from flax.core import FrozenDict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

class CustomDataset:
    def __init__(self, data_args, dataset, use_triplet):
        self.data_args = data_args
        self.dataset = self._prepare_data(dataset)
        self.use_triplet = use_triplet
        self.indices = list(range(len(self.dataset["input_ids"])))

    def _prepare_data(self, data):
        chat_template = self.data_args.chat_template if self.data_args.chat_template != "none" else ""
        return {
            "input_ids": [f"{chat_template} {example['input']}" for example in data],
            "labels": [f"{chat_template} {example['output']}" for example in data],
            "attention_mask": [1] * len(data)
        }

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, idx):
        idx = self.indices[idx]
        if self.use_triplet:
            positive_labels = self.dataset["labels"][idx]
            negative_labels_idx = np.random.randint(0, len(self.dataset["labels"]))
            while negative_labels_idx == idx:
                negative_labels_idx = np.random.randint(0, len(self.dataset["labels"]))
            negative_labels = self.dataset["labels"][negative_labels_idx]
            return {"input_ids": self.dataset["input_ids"][idx], "positive_labels": positive_labels, "negative_labels": negative_labels}
        return {"input_ids": self.dataset["input_ids"][idx], "labels": self.dataset["labels"][idx]}

    def on_epoch_end(self):
        np.random.seed(42)
        self.indices = np.random.permutation(len(self.dataset["input_ids"])).tolist()

    @staticmethod
    def load_json_file(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    @classmethod
    def prepare(cls, data_args, use_triplet):
        train_data = cls.load_json_file("train.json")
        test_data = cls.load_json_file("test.json")
        return cls(data_args, train_data, use_triplet), cls(data_args, test_data, use_triplet)

class T5Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 128)))
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx,
    )
    return state

def train_step(state, batch, use_triplet):
    def loss_fn(params, batch, use_triplet):
        if use_triplet:
            outputs = state.apply_fn({'params': params}, batch["input_ids"])
            loss = (outputs - batch["positive_labels"])**2 - (outputs - batch["negative_labels"])**2
        else:
            labels = batch["labels"]
            inputs = batch["input_ids"]
            outputs = state.apply_fn({'params': params}, inputs)
            loss = (outputs - labels)**2
        return jnp.mean(loss)

    grads = jax.grad(loss_fn, argnums=0)(state.params, batch, use_triplet)
    state = state.apply_gradients(grads=grads)
    return state

def train(data_loader, num_epochs, state, use_triplet):
    for epoch in range(num_epochs):
        for batch in data_loader:
            state = train_step(state, batch, use_triplet)
        data_loader.dataset.on_epoch_end()
        print(f"Epoch {epoch+1}")

def run_pipeline(model_args, data_args, training_args):
    train_dataset, _ = CustomDataset.prepare(data_args, model_args.use_triplet_loss_trainer)
    data_loader = tf.data.Dataset.from_tensor_slices(train_dataset).batch(training_args.per_device_train_batch_size)
    model = T5Model()
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model, 0.001)
    train(data_loader, training_args.num_train_epochs, state, model_args.use_triplet_loss_trainer)

def resume_pipeline(model_args, data_args, training_args, checkpoint_path):
    train_dataset, _ = CustomDataset.prepare(data_args, model_args.use_triplet_loss_trainer)
    data_loader = tf.data.Dataset.from_tensor_slices(train_dataset).batch(training_args.per_device_train_batch_size)
    model = T5Model()
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=jnp.load(checkpoint_path),
    )
    train(data_loader, training_args.num_train_epochs, state, model_args.use_triplet_loss_trainer)

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none", use_triplet_loss_trainer=True)
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    run_pipeline(model_args, data_args, training_args)