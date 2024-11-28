import os
import json
from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np

@dataclass
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

def load_data(file_path: str) -> dict:
    """
    Loads a JSON file into a Python dictionary.
    
    Args:
        file_path (str): Path to the JSON file to be loaded.
    
    Returns:
        dict: A Python dictionary containing the data from the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def prepare_data(data, config: ModelConfig) -> tf.data.Dataset:
    """
    Prepares the input data for training by shuffling, batching, and mapping it.
    
    Args:
        data: Input data to be prepared.
        config (ModelConfig): Configuration for the model.
    
    Returns:
        tf.data.Dataset: Prepared dataset for training.
    """
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
    """
    Builds a neural network model with an embedding layer, two LSTM layers, two dense layers, and an output dense layer.
    
    Args:
        config (ModelConfig): Configuration for the model.
    
    Returns:
        tuple: A tuple containing the built model and the Adam optimizer.
    """
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
    """
    Calculates the triplet loss for a given anchor, positive, and negative example.
    
    Args:
        anchor: Anchor example.
        positive: Positive example.
        negative: Negative example.
        margin (float, optional): Margin for the triplet loss. Defaults to 2.0.
    
    Returns:
        tf.Tensor: Triplet loss for the given examples.
    """
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    losses = tf.maximum(distance_positive - distance_negative + margin, 0)
    return tf.reduce_mean(losses)

def train_on_batch(model, optimizer, anchor, positive, negative) -> tf.Tensor:
    """
    Trains the model on a single batch of data using the Adam optimizer and triplet loss.
    
    Args:
        model: Model to be trained.
        optimizer: Adam optimizer.
        anchor: Anchor example.
        positive: Positive example.
        negative: Negative example.
    
    Returns:
        tf.Tensor: Loss for the given batch.
    """
    with tf.GradientTape() as tape:
        anchor_outputs = model(anchor, training=True)
        positive_outputs = model(positive, training=True)
        negative_outputs = model(negative, training=True)
        loss = calculate_loss(anchor_outputs, positive_outputs, negative_outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def evaluate(model, dataset) -> tf.Tensor:
    """
    Evaluates the model on a given dataset using sparse categorical cross-entropy loss.
    
    Args:
        model: Model to be evaluated.
        dataset: Dataset to be evaluated on.
    
    Returns:
        tf.Tensor: Loss for the given dataset.
    """
    total_loss = 0
    for batch in dataset:
        input_ids = batch['input_ids']
        labels = batch['labels']
        outputs = model(input_ids)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, outputs)
        total_loss += loss
    return total_loss / len(dataset)

def save_model(model, config: ModelConfig, epoch: int) -> None:
    """
    Saves the model weights to a file after each epoch.
    
    Args:
        model: Model to be saved.
        config (ModelConfig): Configuration for the model.
        epoch (int): Current epoch number.
    """
    model.save_weights(f"{config.output_dir}/model_{epoch}.h5")

def train(model, optimizer, config: ModelConfig, train_dataset, test_dataset) -> None:
    """
    Trains the model on a given dataset for a specified number of epochs.
    
    Args:
        model: Model to be trained.
        optimizer: Adam optimizer.
        config (ModelConfig): Configuration for the model.
        train_dataset: Dataset to be trained on.
        test_dataset: Dataset to be evaluated on.
    """
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
    """
    Main function to run the training process.
    """
    model_config = ModelConfig()
    train_data = prepare_data(load_data("train.json"), model_config)
    test_data = prepare_data(load_data("test.json"), model_config)
    model, optimizer = build_model(model_config)
    train(model, optimizer, model_config, train_data, test_data)

if __name__ == "__main__":
    main()