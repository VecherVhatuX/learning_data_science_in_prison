import os
import json
import dataclasses
import typing
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

@dataclasses.dataclass
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

class Dataset:
    def __init__(self, config: Config, data: list):
        self.config = config
        self.data = self._prepare(data)

    def _prepare(self, data: list) -> list:
        prepared_data = []
        for example in data:
            prepared_example = {
                "input_ids": np.array([0] + [ord(c) for c in f"{self.config.chat_format} {example['input']}"] + [1], dtype=np.float32),
                "labels": np.array([0] + [ord(c) for c in f"{self.config.chat_format} {example['output']}"] + [1], dtype=np.float32),
                "attention_mask": np.ones(len(prepared_example["input_ids"]), dtype=np.float32)
            }
            prepared_data.append(prepared_example)
        return prepared_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.fc_layers = [layers.Dense(128, activation='relu') for _ in range(2)]
        self.fc_out = layers.Dense(1000)

    def call(self, x: np.ndarray) -> np.ndarray:
        for layer in self.fc_layers:
            x = layer(x)
        x = self.fc_out(x)
        return x

class Trainer:
    def __init__(self, config: Config, model: Model):
        self.config = config
        self.model = model
        self.criterion = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.AdamW(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.checkpoint_manager = self._create_checkpoint_manager()

    def _create_checkpoint_manager(self):
        checkpoint_dir = os.path.join(self.config.output_path, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt-{epoch:02d}")
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=False,
            save_freq=self.config.save_steps,
            max_to_keep=self.config.max_checkpoints
        )
        return cp_callback

    def training_step(self, batch: list) -> float:
        inputs = np.array([example['input_ids'] for example in batch])
        labels = np.array([example['labels'] for example in batch])
        with keras.GradientTape() as tape:
            outputs = self.model(inputs)
            loss = self.criterion(labels, outputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy()

    def training_epoch(self, dataset: Dataset, batch_size: int) -> float:
        batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
        total_loss = 0
        for batch in batches:
            loss = self.training_step(batch)
            total_loss += loss
        return total_loss / len(batches)

    def fit(self, dataset: Dataset, batch_size: int):
        for epoch in range(self.config.num_epochs):
            loss = self.training_epoch(dataset, batch_size)
            print(f"Epoch {epoch+1}, Loss: {loss}")
        self.model.save(os.path.join(self.config.output_path, "final_model"))

def load_dataset(config: Config) -> tuple:
    with open("train.json", 'r') as f:
        train_data = json.load(f)
    with open("test.json", 'r') as f:
        test_data = json.load(f)
    train_dataset = Dataset(config, train_data)
    test_dataset = Dataset(config, test_data)
    return train_dataset, test_dataset

def main():
    config = Config(model_id="t5-base", chat_format="none", triplet_loss_training=True)
    model = Model()
    trainer = Trainer(config, model)
    train_dataset, _ = load_dataset(config)
    trainer.fit(train_dataset, config.train_batch_size)

if __name__ == "__main__":
    main()