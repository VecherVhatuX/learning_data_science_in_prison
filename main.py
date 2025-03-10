import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import T5Tokenizer


class Config:
    def __init__(self):
        self.model_name = "t5-base"
        self.conversation_style = "none"
        self.alpha = 16
        self.dropout_rate = 0.1
        self.decomposition_rank = 64
        self.layers_to_modify = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
        self.enable_nested_quantization = False
        self.dtype_for_four_bit = "float16"
        self.storage_dtype = "uint8"
        self.quantization_method = "nf4"
        self.use_flash_attention = False
        self.low_rank_peft = False
        self.enable_eight_bit_quantization = False
        self.enable_four_bit_quantization = False
        self.allow_reentrant_training = False
        self.allow_unsloth_training = False
        self.use_triplet_loss = True
        self.data_source = "timdettmers/openassistant-guanaco"
        self.special_token_flag = False
        self.special_tokens_inclusion = False
        self.data_splits = ["train", "test"]
        self.tokenized_data_file = None
        self.results_directory = "./results"
        self.num_epochs = 3
        self.batch_size_train = 16
        self.batch_size_eval = 64
        self.warmup_steps_count = 500
        self.weight_decay_rate = 0.01
        self.logging_directory = "./logs"
        self.model_save_frequency = 500
        self.max_checkpoints_to_keep = 2
        self.seed = 42
        self.checkpoint_resume_path = None
        self.negative_samples_per_batch = 5


class TripletNet(models.Model):
    def __init__(self, embedding_dim, vocab_size):
        super(TripletNet, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = layers.LSTM(embedding_dim, return_sequences=True)
        self.fc_layer = layers.Dense(embedding_dim)
        self.output_layer = layers.Dense(vocab_size)

    def call(self, input_tensor):
        return self.output_layer(self.fc_layer(self.lstm_layer(self.embedding_layer(input_tensor))[:, -1, :]))

    def calculate_triplet_loss(self, anchor, positive, negative):
        return tf.reduce_mean(tf.maximum(tf.reduce_mean(tf.square(anchor - positive), axis=-1) - 
                                          tf.reduce_mean(tf.square(anchor - negative), axis=-1) + 2.0, 0.0))


class TripletData(tf.keras.utils.Sequence):
    def __init__(self, data, config, tokenizer):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.indices = list(range(len(data)))
        self.batch_size = config.batch_size_train

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        return tuple(np.array(x) for x in zip(*[self.prepare_example(index * self.batch_size + i) for i in range(self.batch_size)]))

    def prepare_example(self, idx):
        example = self.data[idx]
        input_id = self.tokenizer.encode(example['input'], max_length=512, padding='max_length', truncation=True)
        label = self.tokenizer.encode(example['output'], max_length=512, padding='max_length', truncation=True)
        negative_exs = [
            self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != idx])]['input'], max_length=512, padding='max_length', truncation=True)
            ) for _ in range(self.config.negative_samples_per_batch)
        ]
        return input_id, label, negative_exs

    def shuffle_indices(self):
        random.seed(self.config.seed)
        random.shuffle(self.indices)

    def epoch_shuffle(self):
        self.shuffle_indices()


def read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def train_triplet_model(model, config, train_loader):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=model.calculate_triplet_loss)
    model.fit(train_loader, epochs=config.num_epochs)


def evaluate_triplet_model(model, test_loader):
    total_loss = sum(model.calculate_triplet_loss(
        model(input_ids),
        model(labels),
        model(negative_exs)
    ).numpy() for input_ids, labels, negative_exs in test_loader)
    
    print(f"Test Loss Average: {total_loss / len(test_loader):.4f}")


def initialize_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")


def save_model(model, path):
    model.save_weights(path)


def main():
    config = Config()
    train_data = read_json("train.json")
    test_data = read_json("test.json")
    tokenizer = initialize_tokenizer()
    train_dataset = TripletData(train_data, config, tokenizer)
    test_dataset = TripletData(test_data, config, tokenizer)

    model = TripletNet(128, 30522)

    for _ in range(config.num_epochs):
        train_dataset.epoch_shuffle()

    train_triplet_model(model, config, train_dataset)
    evaluate_triplet_model(model, test_dataset)
    save_model(model, os.path.join(config.results_directory, "triplet_model.h5"))


if __name__ == "__main__":
    main()