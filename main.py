import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import T5Tokenizer


class Configuration:
    def __init__(self):
        self.model_name = "t5-base"
        self.alpha = 16
        self.dropout_rate = 0.1
        self.decomposition_rank = 64
        self.layers_to_modify = [
            "q_proj", "k_proj", "v_proj", 
            "o_proj", "down_proj", "up_proj", "gate_proj"
        ]
        self.quantization_options = {
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
        }
        self.use_triplet_loss = True
        self.data_source = "timdettmers/openassistant-guanaco"
        self.token_flags = {
            "use_special_token": False,
            "include_special_tokens": False,
        }
        self.data_splits = ["train", "test"]
        self.tokenized_file = None
        self.results_directory = "./results"
        self.epochs = 3
        self.batch_sizes = {
            "train": 16,
            "eval": 64
        }
        self.warmup_steps = 500
        self.weight_decay = 0.01
        self.logging_directory = "./logs"
        self.model_save_interval = 500
        self.max_checkpoints = 2
        self.seed = 42
        self.checkpoint_path = None
        self.negative_samples_per_batch = 5


class TripletModel(models.Model):
    def __init__(self, embedding_dim, vocab_size):
        super(TripletModel, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = layers.LSTM(embedding_dim, return_sequences=True)
        self.dense_layer = layers.Dense(embedding_dim)
        self.output_layer = layers.Dense(vocab_size)

    def call(self, inputs):
        lstm_output = self.lstm_layer(self.embedding_layer(inputs))
        last_output = lstm_output[:, -1, :]
        return self.output_layer(self.dense_layer(last_output))

    def triplet_loss(self, anchor, positive, negative):
        positive_distance = tf.reduce_mean(tf.square(anchor - positive), axis=-1)
        negative_distance = tf.reduce_mean(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(positive_distance - negative_distance + 2.0, 0.0))


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, config, tokenizer):
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = config.batch_sizes['train']
        self.indices = list(range(len(dataset)))

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        return tuple(np.array(x) for x in zip(*[self.prepare_entry(index * self.batch_size + i) for i in range(self.batch_size)]))

    def prepare_entry(self, idx):
        entry = self.dataset[idx]
        input_ids = self.tokenizer.encode(entry['input'], max_length=512, padding='max_length', truncation=True)
        labels = self.tokenizer.encode(entry['output'], max_length=512, padding='max_length', truncation=True)
        negative_samples = [
            self.tokenizer.encode(self.dataset[random.choice([j for j in range(len(self.dataset)) if j != idx])]['input'], max_length=512, padding='max_length', truncation=True)
            ) for _ in range(self.config.negative_samples_per_batch)
        ]
        return input_ids, labels, negative_samples

    def shuffle_data(self):
        random.seed(self.config.seed)
        random.shuffle(self.indices)

    def epoch_shuffle(self):
        self.shuffle_data()


def load_json_file(path):
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None


def train_triplet_model(model, config, data_gen):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=model.triplet_loss)
    model.fit(data_gen, epochs=config.epochs)


def evaluate_triplet_model(model, data_gen):
    total_loss = 0
    for input_ids, labels, negative_samples in data_gen:
        total_loss += model.triplet_loss(
            model(input_ids),
            model(labels),
            model(negative_samples)
        ).numpy()
    
    print(f"Average Evaluation Loss: {total_loss / len(data_gen):.4f}")


def initialize_t5_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")


def save_model(model, path):
    model.save_weights(path)


def execute_training_pipeline():
    config = Configuration()
    train_data = load_json_file("train.json")
    test_data = load_json_file("test.json")
    tokenizer = initialize_t5_tokenizer()
    
    train_dataset = DataGenerator(train_data, config, tokenizer)
    test_dataset = DataGenerator(test_data, config, tokenizer)

    model = TripletModel(128, 30522)

    for _ in range(config.epochs):
        train_dataset.epoch_shuffle()

    train_triplet_model(model, config, train_dataset)
    evaluate_triplet_model(model, test_dataset)
    save_model(model, os.path.join(config.results_directory, "triplet_model.h5"))


if __name__ == "__main__":
    execute_training_pipeline()