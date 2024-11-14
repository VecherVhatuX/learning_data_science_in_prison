import os
import json
from dataclasses import dataclass
from typing import Dict
import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
from flax import linen as nn
from flax.training import train_state
from flax.training import common_utils
from tensorflow.io import gfile
from tqdm import tqdm
import optax

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

def load_json_file(file_name):
    with gfile.GFile(file_name, 'r') as f:
        return json.load(f)

def prepare_dataset(data_args):
    def load_and_prepare_data(file_name):
        data = load_json_file(file_name)
        return jax.tree_map(
            lambda x: jnp.array(x),
            {
                "input_ids": jnp.array([f"{example['input']} " if data_args.chat_template != "none" else example["input"] for example in data]),
                "labels": jnp.array([f"{example['output']} " if data_args.chat_template != "none" else example["output"] for example in data]),
                "attention_mask": jnp.ones(len(data))
            }
        )
    return load_and_prepare_data("train.json"), load_and_prepare_data("test.json")

def create_data_loader(dataset, batch_size):
    return jax.tree_map(
        lambda x: common_utils.get_iterator(x, batch_size, shuffle=True, rng=jax.random.PRNGKey(0)),
        dataset
    )

def get_loss_fn(use_triplet):
    def triplet_loss_fn(x, y, z):
        return jnp.mean((x - y)**2 - (x - z)**2)
    def mse_loss_fn(x, y):
        return jnp.mean((x - y)**2)
    return triplet_loss_fn if use_triplet else mse_loss_fn

def train_step(model, batch, loss_fn):
    if "positive_labels" in batch:
        outputs = model(batch["input_ids"])
        loss = loss_fn(outputs, batch["positive_labels"], batch["negative_labels"])
    else:
        labels = batch["labels"]
        outputs = model(batch["input_ids"])
        loss = loss_fn(outputs, labels)
    return loss

def train_model(model, data_loader, num_epochs, loss_fn):
    optimizer = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))["params"], tx=optimizer
    )
    return [
        (
            epoch,
            state,
            loss
        ) for epoch in tqdm(range(num_epochs))
        for batch in data_loader
        for loss in [train_step(model, batch, loss_fn)]
        for grads in [jax.grad(loss, state.params)]
        for state in [state.apply_gradients(grads=grads)]
    ]

class T5Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x

def run_pipeline(model_args, data_args, training_args):
    train_data, _ = prepare_dataset(model_args)
    train_loader = create_data_loader(train_data, training_args.per_device_train_batch_size)
    loss_fn = get_loss_fn(model_args.use_triplet_loss_trainer)
    for epoch, state, loss in train_model(T5Model(), train_loader, training_args.num_train_epochs, loss_fn):
        print(f"Epoch {epoch+1}, Loss: {loss}")

def resume_pipeline(model_args, data_args, training_args, checkpoint_path):
    train_data, _ = prepare_dataset(model_args)
    train_loader = create_data_loader(train_data, training_args.per_device_train_batch_size)
    model_state, _ = jax2tf.checkpoint.load_pytree(checkpoint_path, None)
    loss_fn = get_loss_fn(model_args.use_triplet_loss_trainer)
    for epoch, state, loss in train_model(T5Model(), train_loader, training_args.num_train_epochs, loss_fn):
        print(f"Epoch {epoch+1}, Loss: {loss}")

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none", use_triplet_loss_trainer=True)
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    run_pipeline(model_args, data_args, training_args)