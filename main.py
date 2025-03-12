import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import T5Tokenizer


def get_configuration():
    return {
        "model_name": "t5-base",
        "alpha": 16,
        "dropout_rate": 0.1,
        "decomposition_rank": 64,
        "layers_to_modify": [
            "q_proj", "k_proj", "v_proj",
            "o_proj", "down_proj", "up_proj", "gate_proj"
        ],
        "quantization_options": {
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
        "token_flags": {
            "use_special_token": False,
            "include_special_tokens": False,
        },
        "data_splits": ["train", "test"],
        "tokenized_file": None,
        "results_directory": "./results",
        "epochs": 3,
        "batch_sizes": {
            "train": 16,
            "eval": 64
        },
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_directory": "./logs",
        "model_save_interval": 500,
        "max_checkpoints": 2,
        "seed": 42,
        "checkpoint_path": None,
        "negative_samples_per_batch": 5,
    }


class TripletNetwork(models.Model):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = layers.LSTM(embedding_dim, return_sequences=True)
        self.dense_layer = layers.Dense(embedding_dim)
        self.output_layer = layers.Dense(vocab_size)

    def call(self, inputs):
        lstm_out = self.lstm_layer(self.embedding_layer(inputs))
        return self.output_layer(self.dense_layer(lstm_out[:, -1, :]))

    def triplet_loss_fn(self, anchor, positive, negative):
        pos_distance = tf.reduce_mean(tf.square(anchor - positive), axis=-1)
        neg_distance = tf.reduce_mean(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_distance - neg_distance + 2.0, 0.0))


class Dataset:
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.indices = list(range(len(data)))

    def shuffle_samples(self):
        random.seed(self.config["seed"])
        random.shuffle(self.indices)

    def get_sample(self, idx):
        data_item = self.data[self.indices[idx]]
        input_ids = self.tokenizer.encode(data_item['input'], max_length=512, padding='max_length', truncation=True)
        labels = self.tokenizer.encode(data_item['output'], max_length=512, padding='max_length', truncation=True)
        return input_ids, labels, self.get_negative_samples(idx)

    def get_negative_samples(self, idx):
        return [
            self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != self.indices[idx]])]['input'],
                                  max_length=512, padding='max_length', truncation=True)
            ) for _ in range(self.config["negative_samples_per_batch"])
        ]

    def generate_epoch_samples(self):
        self.shuffle_samples()
        return [self.get_sample(i) for i in range(len(self.data))]


class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, config, tokenizer):
        self.dataset = Dataset(dataset, config, tokenizer)
        self.batch_size = config["batch_sizes"]['train']

    def __len__(self):
        return len(self.dataset.data) // self.batch_size

    def __getitem__(self, index):
        batch_samples = self.dataset.generate_epoch_samples()[index * self.batch_size:(index + 1) * self.batch_size]
        return tuple(np.array(x) for x in zip(*batch_samples))


def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def train_model(model, config, data_loader):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=model.triplet_loss_fn)
    model.fit(data_loader, epochs=config["epochs"])


def evaluate_model(model, data_loader):
    total_loss = sum(model.triplet_loss_fn(
        model(input_ids),
        model(labels),
        model(neg_samples)
    ).numpy() for input_ids, labels, neg_samples in data_loader)

    print(f"Mean Evaluation Loss: {total_loss / len(data_loader):.4f}")


def load_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")


def save_model_weights(model, file_path):
    model.save_weights(file_path)


def run_training_pipeline():
    config = get_configuration()
    training_data = read_json("train.json")
    testing_data = read_json("test.json")
    tokenizer = load_tokenizer()

    if training_data is None or testing_data is None:
        return

    train_loader = CustomDataLoader(training_data, config, tokenizer)
    test_loader = CustomDataLoader(testing_data, config, tokenizer)

    model = TripletNetwork(128, 30522)

    train_model(model, config, train_loader)
    evaluate_model(model, test_loader)
    save_model_weights(model, os.path.join(config["results_directory"], "triplet_model.h5"))


if __name__ == "__main__":
    run_training_pipeline()