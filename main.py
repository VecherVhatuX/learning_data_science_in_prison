import os
import json
from dataclasses import dataclass
from typing import Dict
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import optax

@dataclass
class Model:
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
class TrainingData:
    dataset_name: str = "timdettmers/openassistant-guanaco"
    append_concat_token: bool = False
    add_special_tokens: bool = False
    splits: str = "train,test"
    tokenized_dataset_path: str = None

@dataclass
class Training:
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

class DataProcessor:
    def __init__(self, data, batch_size, use_triplet):
        self.data = data
        self.batch_size = batch_size
        self.use_triplet = use_triplet
        self.indices = np.arange(len(data["input_ids"]))

    def __iter__(self):
        np.random.shuffle(self.indices)
        batches = np.array_split(self.indices, self.batch_size)
        for batch in batches:
            if self.use_triplet:
                yield {k: jnp.array([self.data[k][idx] for idx in batch]) for k in ["input_ids", "positive_labels", "negative_labels"]}
            else:
                yield {k: jnp.array([self.data[k][idx] for idx in batch]) for k in ["input_ids", "labels"]}

class NeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x

def prepare_data(chat_template, data):
    return {
        "input_ids": [f"{chat_template} {example['input']}" for example in data],
        "labels": [f"{chat_template} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }

def load_data(chat_template):
    train_data = json.load(open("train.json", 'r'))
    test_data = json.load(open("test.json", 'r'))
    return prepare_data(chat_template, train_data), prepare_data(chat_template, test_data)

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 128)))
    tx = optax.adam(learning_rate)
    return optax.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx,
    )

def calculate_loss(params, batch, use_triplet):
    model = NeuralNetwork()
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
    grads = jax.grad(calculate_loss, argnums=0)(state.params, batch, use_triplet)
    state = state.apply_gradients(grads=grads)
    return state

def train_epoch(model, state, dataset, use_triplet):
    return jax.jit(lambda state, dataset: jax.lax.fori_loop(
        0, len(dataset), lambda i, state: execute_train_step(state, next(dataset), use_triplet), state
    ))(state, dataset)

class Trainer:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def train(self):
        train_data, test_data = load_data(self.model_args.chat_template)
        train_dataset = DataProcessor(train_data, self.training_args.per_device_train_batch_size, self.model_args.use_triplet_loss_trainer)
        test_dataset = DataProcessor(test_data, self.training_args.per_device_eval_batch_size, self.model_args.use_triplet_loss_trainer)
        model = NeuralNetwork()
        rng = jax.random.PRNGKey(42)
        state = create_train_state(rng, model, 0.001)

        for epoch in range(self.training_args.num_train_epochs):
            state = train_epoch(model, state, train_dataset, self.model_args.use_triplet_loss_trainer)
            print(f"Epoch {epoch+1}")

if __name__ == "__main__":
    model_args = Model(model_identifier="t5-base", chat_template="none", use_triplet_loss_trainer=True)
    data_args = TrainingData(dataset_name="timdettmers/openassistant-guanaco")
    training_args = Training(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    trainer = Trainer(model_args, data_args, training_args)
    trainer.train()