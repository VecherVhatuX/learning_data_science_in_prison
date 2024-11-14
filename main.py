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

class DatasetBase:
    def __init__(self, data, model_args, num_negative_samples=0):
        self.data = data
        self.model_args = model_args
        self.num_negative_samples = num_negative_samples
        self.input_ids = jnp.array([self._process_input(example) for example in data])
        self.labels = jnp.array([self._process_output(example) for example in data])
        self.attention_mask = jnp.array([1]*len(self.input_ids))

    def _process_input(self, example):
        return example["input"] if self.model_args.chat_template == "none" else f"{example['input']} "

    def _process_output(self, example):
        return example["output"] if self.model_args.chat_template == "none" else f"{example['output']} "

    def __len__(self):
        return len(self.input_ids)

    @staticmethod
    def load_data(file_name):
        with gfile.GFile(file_name, 'r') as f:
            return json.load(f)

    @classmethod
    def prepare_datasets(cls, model_args, data_args, num_negative_samples=0):
        train_data = cls.load_data("train.json")
        test_data = cls.load_data("test.json")
        return cls(train_data, model_args, num_negative_samples), cls(test_data, model_args, num_negative_samples)

    @classmethod
    def create_data_loaders(cls, model_args, data_args, num_negative_samples=0):
        train_data, test_data = cls.prepare_datasets(model_args, data_args, num_negative_samples)
        train_loader = cls._create_data_loader(train_data, data_args.per_device_train_batch_size)
        test_loader = cls._create_data_loader(test_data, data_args.per_device_eval_batch_size)
        return train_loader, test_loader

    @classmethod
    def _create_data_loader(cls, dataset, batch_size):
        return common_utils.get_iterator(
            dataset, batch_size, shuffle=True, rng=jax.random.PRNGKey(0)
        )

class CustomDataset(DatasetBase):
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx], "attention_mask": self.attention_mask[idx]}

class TripletDataset(DatasetBase):
    def __init__(self, data, model_args):
        super().__init__(data, model_args, num_negative_samples=5)

    def __getitem__(self, idx):
        positive_example = self.data[idx]
        negative_examples = jax.random.choice(key=jax.random.PRNGKey(0), a=self.data, shape=(self.num_negative_samples,), replace=False)
        return {
            "input_ids": self.input_ids[idx],
            "positive_labels": positive_example["output"],
            "negative_labels": [example["output"] for example in negative_examples],
            "attention_mask": self.attention_mask[idx]
        }

    def on_epoch_end(self):
        self.data = jax.random.permutation(key=jax.random.PRNGKey(0), x=self.data)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        raise NotImplementedError

class T5Model(BaseModel):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(1000)(x)
        return x

    @staticmethod
    def train_step(model, batch, use_triplet):
        input_ids = batch["input_ids"]
        if use_triplet:
            positive_labels = batch["positive_labels"]
            negative_labels = batch["negative_labels"]
            outputs = model(input_ids)
            loss_fn = jax.jit(jnp.mean((outputs - positive_labels)**2 - (outputs - negative_labels)**2))
            loss = loss_fn
        else:
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids)
            loss_fn = jax.jit(jnp.mean((outputs - labels)**2))
            loss = loss_fn
        return loss

    @classmethod
    def train(cls, model, train_loader, num_epochs, use_triplet):
        optimizer = optax.adam(learning_rate=0.001)
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))["params"], tx=optimizer
        )
        for epoch in tqdm(range(num_epochs)):
            for batch in train_loader:
                loss = cls.train_step(model, batch, use_triplet)
                grads = jax.grad(loss, state.params)
                state = state.apply_gradients(grads=grads)
                print(f"Epoch {epoch+1}, Loss: {loss}")

    @classmethod
    def run_pipeline(cls, model_args, data_args, training_args):
        use_triplet = model_args.use_triplet_loss_trainer
        if use_triplet:
            train_loader, _ = TripletDataset.create_data_loaders(model_args, data_args)
        else:
            train_loader, _ = CustomDataset.create_data_loaders(model_args, data_args)
        cls.train(cls(), train_loader, training_args.num_train_epochs, use_triplet)

    @classmethod
    def resume_pipeline(cls, model_args, data_args, training_args, checkpoint_path):
        use_triplet = model_args.use_triplet_loss_trainer
        if use_triplet:
            train_loader, _ = TripletDataset.create_data_loaders(model_args, data_args)
        else:
            train_loader, _ = CustomDataset.create_data_loaders(model_args, data_args)
        model_state, _ = jax2tf.checkpoint.load_pytree(checkpoint_path, None)
        cls.train(cls(), train_loader, training_args.num_train_epochs, use_triplet)

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none", use_triplet_loss_trainer=True)
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)
    T5Model.run_pipeline(model_args, data_args, training_args)