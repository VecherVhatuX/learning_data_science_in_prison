import os
import json
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from transformers import T5Tokenizer

CONFIG = {
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

def create_language_model(vocab_size, embed_dim):
    return tf.keras.Sequential([
        layers.Embedding(vocab_size, embed_dim),
        layers.LSTM(embed_dim, return_sequences=True),
        layers.Dense(embed_dim),
        layers.Dense(vocab_size)
    ])

def load_json_data(file_path):
    return json.load(open(file_path, 'r')) if os.path.exists(file_path) else None

def tokenize_text(data, tokenizer, max_len=512):
    return tf.convert_to_tensor(tokenizer.encode(data, max_length=max_len, padding='max_length', truncation=True))

def create_text_dataset(data, tokenizer, neg_samples=5):
    return tf.data.Dataset.from_generator(
        lambda: ((tokenize_text(data[idx]['input'], tokenizer), 
                 tokenize_text(data[idx]['output'], tokenizer), 
                 tf.stack([tokenize_text(data[random.choice([j for j in range(len(data)) if j != idx])]['input'], tokenizer) for _ in range(neg_samples)])) 
        for idx in range(len(data))),
        output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))
    )

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_loss = float('inf')
        self.counter = 0

    def train_model(self, data_loader, epochs, patience=3):
        for epoch in range(epochs):
            for input_ids, labels, neg_samples in data_loader:
                with tf.GradientTape() as tape:
                    loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                if self.check_early_stopping(loss.numpy(), patience):
                    print("Training halted early.")
                    return

    def evaluate_model(self, data_loader):
        total_loss = 0
        for input_ids, labels, neg_samples in data_loader:
            loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
            total_loss += loss.numpy()
        print(f"Average Loss on Evaluation: {total_loss / len(data_loader):.4f}")

    def check_early_stopping(self, current_loss, patience):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= patience

def save_model_weights(model, path):
    model.save_weights(path)

def save_training_history(history, path):
    json.dump(history, open(path, 'w'))

def setup_environment():
    config = CONFIG
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = create_language_model(30522, 128)
    optimizer = optimizers.Adam(learning_rate=0.001)
    return config, tokenizer, model, optimizer

def run_pipeline():
    config, tokenizer, model, optimizer = setup_environment()
    train_data, test_data = load_json_data("train.json"), load_json_data("test.json")
    train_dataset = create_text_dataset(train_data, tokenizer, config["negative_samples_per_batch"])
    test_dataset = create_text_dataset(test_data, tokenizer, config["negative_samples_per_batch"])
    train_loader = tf.data.Dataset.from_generator(lambda: train_dataset, output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))))
    test_loader = tf.data.Dataset.from_generator(lambda: test_dataset, output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))))
    trainer = Trainer(model, optimizer, tf.keras.losses.TripletSemiHardLoss())
    trainer.train_model(train_loader, config["epochs"])
    trainer.evaluate_model(test_loader)
    save_model_weights(model, os.path.join(config["results_dir"], "triplet_model.h5"))

def add_learning_rate_scheduler(optimizer, initial_lr, decay_steps, decay_rate):
    return optimizers.schedules.ExponentialDecay(initial_lr, decay_steps, decay_rate)

def add_model_checkpoint(model, optimizer, checkpoint_dir, max_to_keep=2):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save(os.path.join(checkpoint_dir, f"checkpoint_{len(os.listdir(checkpoint_dir))}.ckpt"))

if __name__ == "__main__":
    run_pipeline()