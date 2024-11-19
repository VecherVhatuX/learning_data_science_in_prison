import os
import json
from dataclasses import make_dataclass
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import optax

Config = make_dataclass(
    "Config",
    [
        ("model_id", str, "t5-base"),
        ("chat_format", str, "none"),
        ("lora_alpha", int, 16),
        ("lora_dropout", float, 0.1),
        ("lora_rank", int, 64),
        ("target_layers", str, "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"),
        ("nested_quantization", bool, False),
        ("bit4_compute_type", str, "float16"),
        ("bit4_quant_storage_type", str, "uint8"),
        ("bit4_quant_type", str, "nf4"),
        ("flash_attention", bool, False),
        ("peft_lora", bool, False),
        ("bit8_quantization", bool, False),
        ("bit4_quantization", bool, False),
        ("reentrant", bool, False),
        ("unsloth", bool, False),
        ("triplet_loss_training", bool, False),
        ("dataset_name", str, "timdettmers/openassistant-guanaco"),
        ("append_token", bool, False),
        ("add_special_tokens", bool, False),
        ("data_splits", str, "train,test"),
        ("tokenized_data_path", str, None),
        ("output_path", str, "./results"),
        ("num_epochs", int, 3),
        ("train_batch_size", int, 16),
        ("eval_batch_size", int, 64),
        ("warmup_steps", int, 500),
        ("weight_decay", float, 0.01),
        ("log_path", str, "./logs"),
        ("save_steps", int, 500),
        ("max_checkpoints", int, 2),
        ("random_seed", int, 42),
        ("resume_checkpoint", str, None),
    ],
)

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128, kernel_init=jax.nn.initializers.normal())(x))
        x = nn.relu(nn.Dense(128, kernel_init=jax.nn.initializers.normal())(x))
        x = nn.Dense(1000, kernel_init=jax.nn.initializers.normal())(x)
        return x

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.model = Model()
        self.rng = jax.random.PRNGKey(self.config.random_seed)
        self.input_shape = (1, 128)
        self.params = self.model.init(self.rng, jnp.ones(self.input_shape))
        self.optimizer_state = optax.adamw(0.001, b1=0.9, b2=0.999, eps=1e-8)

    def prepare_dataset(self, chat_format, data):
        return {
            "input_ids": [f"{chat_format} {example['input']}" for example in data],
            "labels": [f"{chat_format} {example['output']}" for example in data],
            "attention_mask": [1] * len(data)
        }

    def load_dataset(self, chat_format):
        with open("train.json", 'r') as f:
            train_data = json.load(f)
        with open("test.json", 'r') as f:
            test_data = json.load(f)
        return self.prepare_dataset(chat_format, train_data), self.prepare_dataset(chat_format, test_data)

    @jax.jit
    def training_step(self, params, batch):
        def loss_fn(params):
            if len(batch) == 3:
                return jnp.mean(jnp.maximum((self.model.apply(params, batch[0]) - batch[1])**2 - (self.model.apply(params, batch[0]) - batch[2])**2, 0))
            else:
                return jnp.mean((self.model.apply(params, batch[0]) - batch[1])**2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, self.optimizer_state = self.optimizer_state.update(grads, self.optimizer_state)
        self.params = optax.apply_updates(params, updates)
        return self.params, loss

    def training_epoch(self, dataset):
        batches = np.array_split(dataset, len(dataset) // self.config.train_batch_size)
        for batch in batches:
            self.params, _ = self.training_step(self.params, batch)
        return self.params

    def train(self):
        train_data, _ = self.load_dataset(self.config.chat_format)
        dataset = np.array(train_data["input_ids"]), np.array(train_data["labels"])
        for _ in range(self.config.num_epochs):
            self.params = self.training_epoch(dataset)
        return self.params

def main():
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    pipeline = TrainingPipeline(config)
    params = pipeline.train()

if __name__ == "__main__":
    main()