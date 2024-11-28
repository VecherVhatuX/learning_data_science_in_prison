import os
import json
import dataclasses
import typing
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np

@dataclasses.dataclass
class ModelConfig:
    model_base: str = "t5-base"
    conversation_format: str = "none"
    low_rank_alpha: int = 16
    low_rank_dropout: float = 0.1
    low_rank_rank: int = 64
    target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
    nested_quantization: bool = False
    four_bit_dtype: str = "float16"
    four_bit_storage_dtype: str = "uint8"
    four_bit_quantization: str = "nf4"
    flash_attention: bool = False
    peft_low_rank: bool = False
    eight_bit_quantization: bool = False
    four_bit_quantization_enabled: bool = False
    reentrant_training: bool = False
    unsloth_training: bool = False
    triplet_loss_training: bool = True
    dataset: str = "timdettmers/openassistant-guanaco"
    append_special_token: bool = False
    add_special_tokens: bool = False
    dataset_splits: str = "train,test"
    tokenized_data_path: str = None
    output_dir: str = "./results"
    num_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    log_dir: str = "./logs"
    save_steps: int = 500
    max_checkpoints: int = 2
    random_seed: int = 42
    resume_checkpoint: str = None
    negative_samples: int = 5

class TripletModel(tf.keras.Model):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = layers.LSTM(embedding_dim, return_sequences=True)
        self.dense = layers.Dense(embedding_dim, activation='relu')
        self.output_dense = layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        x = self.output_dense(x)
        return x

class Dataset(tf.data.Dataset):
    def __init__(self, data, config: ModelConfig):
        self.data = data
        self.config = config

    def _shuffle(self):
        np.random.shuffle(self.data)

    def _get_batch(self, batch_size):
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i+batch_size]
            yield batch

    def _map_fn(self, batch):
        input_ids = tf.strings.split(batch['input'], sep='').to_tensor(dtype=tf.string)
        labels = tf.strings.split(batch['output'], sep='').to_tensor(dtype=tf.string)
        attention_mask = tf.ones((self.config.train_batch_size, max(map(lambda example: len(example['input']), batch))))
        negative_examples = tf.map_fn(
            lambda example: tf.map_fn(
                lambda _: tf.strings.split(tf.strings.reduce_join(tf.random.shuffle(tf.strings.split(example['input'], sep='').to_tensor(dtype=tf.string))), sep='').to_tensor(dtype=tf.string),
                tf.range(self.config.negative_samples),
                dtype=tf.string
            ),
            batch,
            dtype=tf.string
        )
        return ({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "negative_examples": negative_examples
        }, tf.zeros((self.config.train_batch_size,)))

    def __new__(cls, data, config: ModelConfig):
        return super(Dataset, cls).__new__(cls)

    def __getitem__(self, idx):
        return self._map_fn(next(self._get_batch(self.config.train_batch_size)))

    def __len__(self):
        return len(self.data)

def load_data(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def calculate_loss(anchor, positive, negative, margin=2.0) -> tf.Tensor:
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    losses = tf.maximum(distance_positive - distance_negative + margin, 0)
    return tf.reduce_mean(losses)

def train_on_batch(model, optimizer, anchor, positive, negative) -> tf.Tensor:
    with tf.GradientTape() as tape:
        anchor_outputs = model(anchor, training=True)
        positive_outputs = model(positive, training=True)
        negative_outputs = model(negative, training=True)
        loss = calculate_loss(anchor_outputs, positive_outputs, negative_outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate(model, dataset) -> tf.Tensor:
    total_loss = 0
    for batch in dataset:
        input_ids = batch['input_ids']
        labels = batch['labels']
        outputs = model(input_ids)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, outputs)
        total_loss += loss
    return total_loss / len(list(dataset))

def save_model(model, config: ModelConfig, epoch: int) -> None:
    model.save_weights(f"{config.output_dir}/model_{epoch}.h5")

def train(model, optimizer, config: ModelConfig, train_dataset, test_dataset) -> None:
    for epoch in range(config.num_epochs):
        np.random.shuffle(train_dataset.data)
        total_loss = 0
        for batch in train_dataset:
            anchor = batch[0]['input_ids']
            positive = batch[0]['labels']
            negative = batch[0]['negative_examples']
            loss = train_on_batch(model, optimizer, anchor, positive, negative)
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataset)}")
        test_loss = evaluate(model, test_dataset)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss}")
        save_model(model, config, epoch+1)

def load_and_prepare_data(config: ModelConfig) -> tuple:
    train_data = load_data("train.json")
    test_data = load_data("test.json")
    train_dataset = Dataset(train_data, config)
    test_dataset = Dataset(test_data, config)
    return train_dataset, test_dataset

def build_and_compile_model(config: ModelConfig) -> tuple:
    model = TripletModel(embedding_dim=128, vocab_size=1000)
    optimizer = optimizers.Adam(learning_rate=0.001)
    return model, optimizer

def main() -> None:
    config = ModelConfig()
    train_dataset, test_dataset = load_and_prepare_data(config)
    model, optimizer = build_and_compile_model(config)
    train(model, optimizer, config, train_dataset, test_dataset)

if __name__ == "__main__":
    main()