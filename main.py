import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import T5Tokenizer

def create_model_config():
    return {
        "model_name": "t5-base",
        "alpha": 16,
        "dropout_rate": 0.1,
        "decomposition_rank": 64,
        "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        "quantization_config": {
            "nested_quantization": False,
            "four_bit_dtype": "float16",
            "storage_dtype": "uint8",
            "quantization_strategy": "nf4",
            "flash_attention": False,
            "low_rank_peft": False,
            "eight_bit_quantization": False,
            "four_bit_quantization": False,
            "reentrant_training": False,
            "unsloth_training": False,
        },
        "use_triplet_loss": True,
        "data_source": "timdettmers/openassistant-guanaco",
        "token_flags": {"use_special_token": False, "include_special_tokens": False},
        "data_splits": ["train", "test"],
        "tokenized_file": None,
        "results_dir": "./results",
        "epochs": 3,
        "batch_sizes": {"train": 16, "eval": 64},
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "model_save_interval": 500,
        "max_checkpoints": 2,
        "seed": 42,
        "checkpoint_path": None,
        "negative_samples_per_batch": 5,
    }

class TripletModel(Model):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(embedding_dim, return_sequences=True)
        self.dense = layers.Dense(embedding_dim)
        self.output_layer = layers.Dense(vocab_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x[:, -1, :])
        return self.output_layer(x)

    def compute_triplet_loss(self, anchor, positive, negative):
        pos_dist = tf.reduce_mean(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_mean(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + 2.0, 0.0))

class DataHandler:
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.indices = list(range(len(data)))

    def shuffle_data(self):
        random.seed(self.config["seed"])
        random.shuffle(self.indices)

    def get_sample(self, idx):
        item = self.data[self.indices[idx]]
        input_ids = self.tokenizer.encode(item['input'], max_length=512, padding='max_length', truncation=True)
        labels = self.tokenizer.encode(item['output'], max_length=512, padding='max_length', truncation=True)
        return input_ids, labels, self.get_negative_samples(idx)

    def get_negative_samples(self, idx):
        return [self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != self.indices[idx]])]['input'],
                                  max_length=512, padding='max_length', truncation=True)
            for _ in range(self.config["negative_samples_per_batch"])]

    def generate_samples(self):
        self.shuffle_data()
        return [self.get_sample(i) for i in range(len(self.data))]

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, config, tokenizer):
        self.dataset = DataHandler(dataset, config, tokenizer)
        self.batch_size = config["batch_sizes"]['train']

    def __len__(self):
        return len(self.dataset.data) // self.batch_size

    def __getitem__(self, index):
        samples = self.dataset.generate_samples()[index * self.batch_size:(index + 1) * self.batch_size]
        return tuple(np.array(x) for x in zip(*samples))

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def initialize_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")

def create_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.001)

def configure_training(model, optimizer):
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: model.compute_triplet_loss(y_true[0], y_true[1], y_pred))

def train_model(model, config, data_loader):
    optimizer = create_optimizer()
    configure_training(model, optimizer)
    model.fit(data_loader, epochs=config["epochs"])

def evaluate_model(model, data_loader):
    total_loss = 0
    for input_ids, labels, neg_samples in data_loader:
        loss = model.compute_triplet_loss(model(input_ids), model(labels), model(neg_samples)).numpy()
        total_loss += loss
    print(f"Mean Evaluation Loss: {total_loss / len(data_loader):.4f}")

def save_model(model, file_path):
    model.save_weights(file_path)

def save_logs(history, file_path):
    with open(file_path, 'w') as f:
        json.dump(history.history, f)

def execute_training():
    config = create_model_config()
    train_data = load_data("train.json")
    test_data = load_data("test.json")
    tokenizer = initialize_tokenizer()

    if train_data is None or test_data is None:
        return

    train_loader = DataLoader(train_data, config, tokenizer)
    test_loader = DataLoader(test_data, config, tokenizer)

    model = TripletModel(128, 30522)

    train_model(model, config, train_loader)
    evaluate_model(model, test_loader)
    save_model(model, os.path.join(config["results_dir"], "triplet_model.h5"))

if __name__ == "__main__":
    execute_training()