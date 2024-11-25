import os
import json
from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np

@dataclass
class Hyperparameters:
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

def create_dataset(data, conversation_format, negative_samples, batch_size):
    def generator():
        for i in range(len(data) // batch_size):
            batch_data = data[i * batch_size:(i + 1) * batch_size]
            input_ids = []
            labels = []
            attention_mask = []
            negative_examples = []
            for example in batch_data:
                input_id = np.array([0] + [ord(c) for c in f"{conversation_format} {example['input']}"] + [1], dtype=np.int32)
                label = np.array([0] + [ord(c) for c in f"{conversation_format} {example['output']}"] + [1], dtype=np.int32)
                attention_mask_val = np.array([1] * len(input_id), dtype=np.int32)
                input_ids.append(input_id)
                labels.append(label)
                attention_mask.append(attention_mask_val)
                negative_example = []
                for _ in range(negative_samples):
                    negative_idx = np.random.randint(0, len(data))
                    while negative_idx == i:
                        negative_idx = np.random.randint(0, len(data))
                    negative_example_val = data[negative_idx]
                    negative_input_id = np.array([0] + [ord(c) for c in f"{conversation_format} {negative_example_val['input']}"] + [1], dtype=np.int32)
                    negative_label = np.array([0] + [ord(c) for c in f"{conversation_format} {negative_example_val['output']}"] + [1], dtype=np.int32)
                    negative_attention_mask_val = np.array([1] * len(negative_input_id), dtype=np.int32)
                    negative_example.append({"input_ids": negative_input_id, "labels": negative_label, "attention_mask": negative_attention_mask_val})
                negative_examples.append(negative_example)
            yield {"input_ids": np.array(input_ids), "labels": np.array(labels), "attention_mask": np.array(attention_mask), "negative_examples": negative_examples}

    def len_dataset():
        return len(data) // batch_size

    return generator, len_dataset

def create_t5_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1000)
    ])
    return model

def triplet_loss(anchor, positive, negative, margin=2.0):
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    losses = tf.maximum(distance_positive - distance_negative + margin, 0)
    return tf.reduce_mean(losses)

def load_data(file_name):
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{file_name} not found.")
        return None

def train(model, dataset_generator, dataset_len, hyperparameters):
    for epoch in range(hyperparameters.num_epochs):
        total_loss = 0
        for batch in dataset_generator():
            anchor_input_ids = batch["input_ids"]
            positive_input_ids = batch["labels"]
            negative_input_ids = batch["negative_examples"][0]["input_ids"]
            with tf.GradientTape() as tape:
                anchor_outputs = model(anchor_input_ids, training=True)
                positive_outputs = model(positive_input_ids, training=True)
                negative_outputs = model(negative_input_ids, training=True)
                loss = triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss / dataset_len()}")

def evaluate(model, dataset_generator, dataset_len):
    total_loss = 0
    for batch in dataset_generator():
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        outputs = model(input_ids)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, outputs)
        total_loss += loss
    print(f"Test Loss: {total_loss / dataset_len()}")

def save_model(model, epoch, output_dir):
    model.save_weights(f"{output_dir}/model_{epoch}.h5")

def main():
    hyperparameters = Hyperparameters(model_base="t5-base", conversation_format="none", triplet_loss_training=True)
    model = create_t5_model((224, 224, 3))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=triplet_loss, metrics=['accuracy'])
    training_data, testing_data = load_data("train.json"), load_data("test.json")
    if training_data is not None and testing_data is not None:
        train_dataset_generator, train_dataset_len = create_dataset(training_data, hyperparameters.conversation_format, hyperparameters.negative_samples, hyperparameters.train_batch_size)
        test_dataset_generator, test_dataset_len = create_dataset(testing_data, hyperparameters.conversation_format, hyperparameters.negative_samples, hyperparameters.eval_batch_size)
        for epoch in range(hyperparameters.num_epochs):
            train(model, train_dataset_generator, train_dataset_len, hyperparameters)
            evaluate(model, test_dataset_generator, test_dataset_len)
            save_model(model, epoch, hyperparameters.output_dir)

if __name__ == "__main__":
    main()