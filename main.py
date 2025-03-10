import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import T5Tokenizer


class ModelConfig:
    def __init__(self):
        self.model_name = "t5-base"
        self.alpha = 16
        self.dropout_rate = 0.1
        self.decomposition_rank = 64
        self.layers_to_modify = [
            "q_proj", "k_proj", "v_proj", 
            "o_proj", "down_proj", "up_proj", "gate_proj"
        ]
        self.quantization_settings = {
            "enable_nested_quantization": False,
            "dtype_for_four_bit": "float16",
            "storage_dtype": "uint8",
            "quantization_method": "nf4",
            "use_flash_attention": False,
            "low_rank_peft": False,
            "enable_eight_bit_quantization": False,
            "enable_four_bit_quantization": False,
            "allow_reentrant_training": False,
            "allow_unsloth_training": False,
        }
        self.use_triplet_loss = True
        self.data_source = "timdettmers/openassistant-guanaco"
        self.special_token_flags = {
            "special_token_flag": False,
            "special_tokens_inclusion": False,
        }
        self.data_splits = ["train", "test"]
        self.tokenized_data_file = None
        self.results_directory = "./results"
        self.num_epochs = 3
        self.batch_sizes = {
            "train": 16,
            "eval": 64
        }
        self.warmup_steps_count = 500
        self.weight_decay_rate = 0.01
        self.logging_directory = "./logs"
        self.model_save_frequency = 500
        self.max_checkpoints_to_keep = 2
        self.seed = 42
        self.checkpoint_resume_path = None
        self.negative_samples_per_batch = 5


class TripletNetwork(models.Model):
    def __init__(self, embedding_dim, vocab_size):
        super(TripletNetwork, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(embedding_dim, return_sequences=True)
        self.fc = layers.Dense(embedding_dim)
        self.output = layers.Dense(vocab_size)

    def call(self, inputs):
        lstm_output = self.lstm(self.embedding(inputs))
        last_output = lstm_output[:, -1, :]
        return self.output(self.fc(last_output))

    def compute_triplet_loss(self, anchor, positive, negative):
        pos_dist = tf.reduce_mean(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_mean(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + 2.0, 0.0))


class TripletDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, config, tokenizer):
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer
        self.indices = list(range(len(dataset)))
        self.batch_size = config.batch_sizes['train']

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        return tuple(np.array(x) for x in zip(*[self.prepare_example(index * self.batch_size + i) for i in range(self.batch_size)]))

    def prepare_example(self, idx):
        entry = self.dataset[idx]
        input_ids = self.tokenizer.encode(entry['input'], max_length=512, padding='max_length', truncation=True)
        labels = self.tokenizer.encode(entry['output'], max_length=512, padding='max_length', truncation=True)
        negative_samples = [
            self.tokenizer.encode(self.dataset[random.choice([j for j in range(len(self.dataset)) if j != idx])]['input'], max_length=512, padding='max_length', truncation=True)
            ) for _ in range(self.config.negative_samples_per_batch)
        ]
        return input_ids, labels, negative_samples

    def shuffle_indices(self):
        random.seed(self.config.seed)
        random.shuffle(self.indices)

    def epoch_shuffle(self):
        self.shuffle_indices()


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def train_model(model, config, data_generator):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=model.compute_triplet_loss)
    model.fit(data_generator, epochs=config.num_epochs)


def evaluate_model(model, data_generator):
    total_loss = 0
    for input_ids, labels, negative_exs in data_generator:
        total_loss += model.compute_triplet_loss(
            model(input_ids),
            model(labels),
            model(negative_exs)
        ).numpy()
    
    print(f"Average Test Loss: {total_loss / len(data_generator):.4f}")


def initialize_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")


def save_model_weights(model, path):
    model.save_weights(path)


def run_training_pipeline():
    config = ModelConfig()
    train_data = load_json("train.json")
    test_data = load_json("test.json")
    tokenizer = initialize_tokenizer()
    
    train_dataset = TripletDataGenerator(train_data, config, tokenizer)
    test_dataset = TripletDataGenerator(test_data, config, tokenizer)

    model = TripletNetwork(128, 30522)

    for _ in range(config.num_epochs):
        train_dataset.epoch_shuffle()

    train_model(model, config, train_dataset)
    evaluate_model(model, test_dataset)
    save_model_weights(model, os.path.join(config.results_directory, "triplet_model.h5"))


if __name__ == "__main__":
    run_training_pipeline()