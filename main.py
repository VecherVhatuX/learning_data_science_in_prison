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
    with open(file_name, 'r') as f:
        return json.load(f)


def prepare_data(data_args, chat_template, data):
    return {
        "input_ids": [f"{chat_template} {example['input']}" for example in data],
        "labels": [f"{chat_template} {example['output']}" for example in data],
        "attention_mask": [1] * len(data)
    }


def prepare_triplet_data(data_args, chat_template, data, indices, use_triplet):
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
    params = model.init(rng, jnp.ones((1, 128)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx,
    )


def calculate_loss(params, batch, use_triplet):
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
    grads = jax.grad(calculate_loss, argnums=0)(state.params, batch, use_triplet)
    state = state.apply_gradients(grads=grads)
    return state


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x


def train_epoch(model, state, dataset, use_triplet):
    for batch in dataset:
        state = execute_train_step(state, batch, use_triplet)
    return state


def run_pipeline(model_args, data_args, training_args):
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
    def __init__(self, data, batch_size, use_triplet):
        self.data = data
        self.batch_size = batch_size
        self.use_triplet = use_triplet
        self.indices = list(range(len(self.data["input_ids"])))

    def shuffle(self):
        np.random.seed(42)
        self.indices = np.random.permutation(self.indices)

    def get_triplet_data(self):
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
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = {k: np.array([self.data[k][idx] for idx in batch_indices]) for k in self.data.keys()}
            yield batch_data


def main(model_args, data_args, training_args):
    run_pipeline(model_args, data_args, training_args)


def pipeline(model_args, data_args, training_args):
    return lambda: main(model_args, data_args, training_args)


def compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def main_pipeline(model_args, data_args, training_args):
    return compose(
        lambda f: f(),
        lambda: pipeline(model_args, data_args, training_args)()
    )


def run(model_args, data_args, training_args):
    return main_pipeline(model_args, data_args, training_args)()


if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none", use_triplet_loss_trainer=True)
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    run(model_args, data_args, training_args)