import os
import json
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from transformers import T5Tokenizer

class ModelConfig:
    def __init__(self):
        self.settings = {
            "model_name": "t5-base", "alpha": 16, "dropout_rate": 0.1, "decomposition_rank": 64,
            "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            "quantization_config": {"nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
                                   "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
                                   "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
                                   "unsloth_training": False},
            "use_triplet_loss": True, "data_source": "timdettmers/openassistant-guanaco",
            "token_flags": {"use_special_token": False, "include_special_tokens": False}, "data_splits": ["train", "test"],
            "tokenized_file": None, "results_dir": "./results", "epochs": 3, "batch_sizes": {"train": 16, "eval": 64},
            "warmup_steps": 500, "weight_decay": 0.01, "logging_dir": "./logs", "model_save_interval": 500,
            "max_checkpoints": 2, "seed": 42, "checkpoint_path": None, "negative_samples_per_batch": 5
        }

class TextEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super(TextEncoder, self).__init__()
        self.embed = layers.Embedding(vocab_size, embed_dim)
        self.rnn = layers.LSTM(embed_dim, return_sequences=True, return_state=True)
        self.fc1 = layers.Dense(embed_dim)
        self.fc2 = layers.Dense(vocab_size)

    def call(self, x):
        x = self.embed(x)
        x, _, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.fc2(x)

class DataHandler:
    @staticmethod
    def load_data(file):
        if os.path.exists(file):
            return json.load(open(file, 'r'))
        print(f"File not found: {file}")
        return None

    @staticmethod
    def tokenize_data(data, tokenizer, max_len=512):
        return tf.convert_to_tensor(tokenizer.encode(data, max_length=max_len, padding='max_length', truncation=True))

class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, data, tokenizer, neg_samples=5):
        self.data = data
        self.tokenizer = tokenizer
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = DataHandler.tokenize_data(self.data[idx]['input'], self.tokenizer)
        labels = DataHandler.tokenize_data(self.data[idx]['output'], self.tokenizer)
        neg_samples = tf.stack([DataHandler.tokenize_data(self.data[random.choice([j for j in range(len(self.data)) if j != idx])]['input'], self.tokenizer) for _ in range(self.neg_samples)])
        return input_ids, labels, neg_samples

class TrainingManager:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_loss = float('inf')
        self.counter = 0

    def train(self, data_loader, epochs, patience=3):
        self.model.train()
        for epoch in range(epochs):
            for input_ids, labels, neg_samples in data_loader:
                with tf.GradientTape() as tape:
                    loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if self.early_stop(loss.numpy(), patience):
                print("Early stopping triggered.")
                break

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        for input_ids, labels, neg_samples in data_loader:
            total_loss += self.loss_fn(self.model(input_ids), labels, neg_samples).numpy()
        print(f"Mean Evaluation Loss: {total_loss / len(data_loader):.4f}")

    def early_stop(self, current_loss, patience):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= patience

class ModelSaver:
    @staticmethod
    def save_model(model, path):
        model.save_weights(path)

    @staticmethod
    def save_history(history, path):
        json.dump(history, open(path, 'w'))

def initialize_components():
    config = ModelConfig().settings
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = TextEncoder(30522, 128)
    optimizer = optimizers.Adam(learning_rate=0.001)
    return config, tokenizer, model, optimizer

def execute_training():
    config, tokenizer, model, optimizer = initialize_components()
    train_data = DataHandler.load_data("train.json")
    test_data = DataHandler.load_data("test.json")
    train_dataset = TripletDataset(train_data, tokenizer, config["negative_samples_per_batch"])
    test_dataset = TripletDataset(test_data, tokenizer, config["negative_samples_per_batch"])
    train_loader = tf.data.Dataset.from_generator(lambda: train_dataset, output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)
    )).batch(config["batch_sizes"]['train']).shuffle(buffer_size=len(train_dataset))
    test_loader = tf.data.Dataset.from_generator(lambda: test_dataset, output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)
    )).batch(config["batch_sizes"]['eval'])
    trainer = TrainingManager(model, optimizer, triplet_loss)
    trainer.train(train_loader, config["epochs"])
    trainer.evaluate(test_loader)
    ModelSaver.save_model(model, os.path.join(config["results_dir"], "triplet_model.h5"))

if __name__ == "__main__":
    execute_training()