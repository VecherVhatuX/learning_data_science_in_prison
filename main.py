import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import T5Tokenizer

def get_model_config():
    return {
        "model_name": "t5-base", "alpha": 16, "dropout_rate": 0.1, "decomposition_rank": 64,
        "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        "quantization_config": {
            "nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
            "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
            "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
            "unsloth_training": False,
        },
        "use_triplet_loss": True, "data_source": "timdettmers/openassistant-guanaco",
        "token_flags": {"use_special_token": False, "include_special_tokens": False},
        "data_splits": ["train", "test"], "tokenized_file": None, "results_dir": "./results",
        "epochs": 3, "batch_sizes": {"train": 16, "eval": 64}, "warmup_steps": 500, "weight_decay": 0.01,
        "logging_dir": "./logs", "model_save_interval": 500, "max_checkpoints": 2, "seed": 42,
        "checkpoint_path": None, "negative_samples_per_batch": 5,
    }

def build_triplet_model(embedding_dim, vocab_size):
    inputs = layers.Input(shape=(None,))
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.LSTM(embedding_dim, return_sequences=True)(x)
    x = x[:, -1, :]
    x = layers.Dense(embedding_dim)(x)
    outputs = layers.Dense(vocab_size)(x)
    return Model(inputs, outputs)

def compute_triplet_loss(anchor, positive, negative):
    return tf.reduce_mean(tf.maximum(tf.reduce_mean(tf.square(anchor - positive), axis=-1) - 
                          tf.reduce_mean(tf.square(anchor - negative), axis=-1) + 2.0, 0.0))

class DataManager:
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.indices = list(range(len(data)))

    def shuffle(self):
        random.shuffle(self.indices, random.seed(self.config["seed"]))

    def get_sample(self, idx):
        input_ids = self.tokenizer.encode(self.data[self.indices[idx]]['input'], max_length=512, padding='max_length', truncation=True)
        labels = self.tokenizer.encode(self.data[self.indices[idx]]['output'], max_length=512, padding='max_length', truncation=True)
        neg_samples = self.get_negative_samples(idx)
        return input_ids, labels, neg_samples

    def get_negative_samples(self, idx):
        return [self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != self.indices[idx]])]['input'],
                                     max_length=512, padding='max_length', truncation=True) for _ in range(self.config["negative_samples_per_batch"])]

    def generate_samples(self):
        return [self.get_sample(i) for i in range(len(self.data))]

class BatchLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, config, tokenizer):
        self.dataset = DataManager(dataset, config, tokenizer)
        self.batch_size = config["batch_sizes"]['train']

    def __len__(self):
        return len(self.dataset.data) // self.batch_size

    def __getitem__(self, index):
        samples = self.dataset.generate_samples()[index * self.batch_size:(index + 1) * self.batch_size]
        return tuple(np.array(x) for x in zip(*samples))

def load_dataset(file_path):
    if os.path.exists(file_path):
        return json.load(open(file_path, 'r'))
    print(f"File not found: {file_path}")
    return None

def get_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")

def build_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.001)

def setup_training(model, optimizer):
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: compute_triplet_loss(y_true[0], y_true[1], y_pred))

def train(model, config, data_loader):
    model.fit(data_loader, epochs=config["epochs"], optimizer=build_optimizer(), setup_training(model, build_optimizer()))

def evaluate(model, data_loader):
    total_loss = sum(compute_triplet_loss(model(input_ids), model(labels), model(neg_samples)).numpy() for input_ids, labels, neg_samples in data_loader)
    print(f"Mean Evaluation Loss: {total_loss / len(data_loader):.4f}")

def save_weights(model, file_path):
    model.save_weights(file_path)

def save_history(history, file_path):
    json.dump(history.history, open(file_path, 'w'))

def run_training():
    config = get_model_config()
    train_data = load_dataset("train.json")
    test_data = load_dataset("test.json")
    tokenizer = get_tokenizer()

    if train_data and test_data:
        train_loader = BatchLoader(train_data, config, tokenizer)
        test_loader = BatchLoader(test_data, config, tokenizer)
        model = build_triplet_model(128, 30522)
        train(model, config, train_loader)
        evaluate(model, test_loader)
        save_weights(model, os.path.join(config["results_dir"], "triplet_model.h5"))

if __name__ == "__main__":
    run_training()