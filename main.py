import os
import json
import dataclasses
import typing
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np

class ModelConfig:
    def __init__(
        self,
        model_base: str = "t5-base",
        conversation_format: str = "none",
        low_rank_alpha: int = 16,
        low_rank_dropout: float = 0.1,
        low_rank_rank: int = 64,
        target_layers: str = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        nested_quantization: bool = False,
        four_bit_dtype: str = "float16",
        four_bit_storage_dtype: str = "uint8",
        four_bit_quantization: str = "nf4",
        flash_attention: bool = False,
        peft_low_rank: bool = False,
        eight_bit_quantization: bool = False,
        four_bit_quantization_enabled: bool = False,
        reentrant_training: bool = False,
        unsloth_training: bool = False,
        triplet_loss_training: bool = True,
        dataset: str = "timdettmers/openassistant-guanaco",
        append_special_token: bool = False,
        add_special_tokens: bool = False,
        dataset_splits: str = "train,test",
        tokenized_data_path: str = None,
        output_dir: str = "./results",
        num_epochs: int = 3,
        train_batch_size: int = 16,
        eval_batch_size: int = 64,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        log_dir: str = "./logs",
        save_steps: int = 500,
        max_checkpoints: int = 2,
        random_seed: int = 42,
        resume_checkpoint: str = None,
        negative_samples: int = 5
    ):
        self.model_base = model_base
        self.conversation_format = conversation_format
        self.low_rank_alpha = low_rank_alpha
        self.low_rank_dropout = low_rank_dropout
        self.low_rank_rank = low_rank_rank
        self.target_layers = target_layers
        self.nested_quantization = nested_quantization
        self.four_bit_dtype = four_bit_dtype
        self.four_bit_storage_dtype = four_bit_storage_dtype
        self.four_bit_quantization = four_bit_quantization
        self.flash_attention = flash_attention
        self.peft_low_rank = peft_low_rank
        self.eight_bit_quantization = eight_bit_quantization
        self.four_bit_quantization_enabled = four_bit_quantization_enabled
        self.reentrant_training = reentrant_training
        self.unsloth_training = unsloth_training
        self.triplet_loss_training = triplet_loss_training
        self.dataset = dataset
        self.append_special_token = append_special_token
        self.add_special_tokens = add_special_tokens
        self.dataset_splits = dataset_splits
        self.tokenized_data_path = tokenized_data_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.log_dir = log_dir
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.random_seed = random_seed
        self.resume_checkpoint = resume_checkpoint
        self.negative_samples = negative_samples

def load_data(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def prepare_data(data, config: ModelConfig) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=len(data))
    if config.conversation_format == "none":
        dataset = dataset.batch(config.train_batch_size if config.train_batch_size > 0 else len(data))
    else:
        dataset = dataset.batch(config.train_batch_size)
    dataset = dataset.map(lambda x: ({
        "input_ids": tf.map_fn(lambda example: tf.strings.split(example['input'], sep='').to_tensor(dtype=tf.string), x, dtype=tf.string),
        "labels": tf.map_fn(lambda example: tf.strings.split(example['output'], sep='').to_tensor(dtype=tf.string), x, dtype=tf.string),
        "attention_mask": tf.ones((config.train_batch_size if config.train_batch_size > 0 else len(data), max(map(lambda example: len(example['input']), x)))),
        "negative_examples": tf.map_fn(lambda example: tf.map_fn(lambda _: tf.strings.split(tf.strings.reduce_join(tf.random.shuffle(tf.strings.split(example['input'], sep='').to_tensor(dtype=tf.string))), sep='').to_tensor(dtype=tf.string), tf.range(config.negative_samples), dtype=tf.string), x, dtype=tf.string)
    }, tf.zeros((config.train_batch_size if config.train_batch_size > 0 else len(data),))))
    return dataset

def build_model(config: ModelConfig) -> (tf.keras.Model, tf.keras.optimizers.Optimizer):
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=1000, output_dim=128),
        layers.LSTM(128),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1000)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    return model, optimizer

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
    return total_loss / len(dataset)

def save_model(model, config: ModelConfig, epoch: int) -> None:
    model.save_weights(f"{config.output_dir}/model_{epoch}.h5")

def train(model, optimizer, config: ModelConfig, train_dataset, test_dataset) -> None:
    for epoch in range(config.num_epochs):
        total_loss = 0
        for batch in train_dataset:
            anchor = batch['input_ids']
            positive = batch['labels']
            negative = batch['negative_examples']
            loss = train_on_batch(model, optimizer, anchor, positive, negative)
            total_loss += loss
        print(f"Loss: {total_loss / len(train_dataset)}")
        test_loss = evaluate(model, test_dataset)
        print(f"Test Loss: {test_loss}")
        save_model(model, config, epoch)

def main() -> None:
    model_config = ModelConfig()
    train_data = prepare_data(load_data("train.json"), model_config)
    test_data = prepare_data(load_data("test.json"), model_config)
    model, optimizer = build_model(model_config)
    train(model, optimizer, model_config, train_data, test_data)

if __name__ == "__main__":
    main()