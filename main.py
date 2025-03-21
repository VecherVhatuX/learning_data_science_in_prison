import os
import json
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from transformers import T5Tokenizer

SETTINGS = {
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

def LanguageModel(vocab_size, embed_dim):
    return tf.keras.Sequential([
        layers.Embedding(vocab_size, embed_dim),
        layers.LSTM(embed_dim, return_sequences=True),
        layers.Dense(embed_dim),
        layers.Dense(vocab_size)
    ])

def load_data(file_path):
    return json.load(open(file_path, 'r')) if os.path.exists(file_path) else None

def tokenize_data(data, tokenizer, max_len=512):
    return tf.convert_to_tensor(tokenizer.encode(data, max_length=max_len, padding='max_length', truncation=True))

def TextDataset(data, tokenizer, neg_samples=5):
    return tf.data.Dataset.from_generator(
        lambda: ((tokenize_data(data[idx]['input'], tokenizer), 
                 tokenize_data(data[idx]['output'], tokenizer), 
                 tf.stack([tokenize_data(data[random.choice([j for j in range(len(data)) if j != idx])]['input'], tokenizer) for _ in range(neg_samples)])) 
        for idx in range(len(data))),
        output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))
    )

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_loss = float('inf')
        self.counter = 0

    def train(self, data_loader, epochs, patience=3):
        for epoch in range(epochs):
            for input_ids, labels, neg_samples in data_loader:
                with tf.GradientTape() as tape:
                    loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                if self.early_stopping(loss.numpy(), patience):
                    print("Training halted early.")
                    return

    def evaluate(self, data_loader):
        total_loss = 0
        for input_ids, labels, neg_samples in data_loader:
            loss = self.loss_fn(self.model(input_ids), labels, neg_samples)
            total_loss += loss.numpy()
        print(f"Average Loss on Evaluation: {total_loss / len(data_loader):.4f}")

    def early_stopping(self, current_loss, patience):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= patience

def save_model(model, path):
    model.save_weights(path)

def save_history(history, path):
    json.dump(history, open(path, 'w'))

def initialize_environment():
    settings = SETTINGS
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = LanguageModel(30522, 128)
    optimizer = optimizers.Adam(learning_rate=0.001)
    return settings, tokenizer, model, optimizer

def execute_pipeline():
    settings, tokenizer, model, optimizer = initialize_environment()
    train_data, test_data = load_data("train.json"), load_data("test.json")
    train_dataset = TextDataset(train_data, tokenizer, settings["negative_samples_per_batch"])
    test_dataset = TextDataset(test_data, tokenizer, settings["negative_samples_per_batch"])
    train_loader = tf.data.Dataset.from_generator(lambda: train_dataset, output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))))
    test_loader = tf.data.Dataset.from_generator(lambda: test_dataset, output_signature=(tf.TensorSpec(shape=(None,), tf.TensorSpec(shape=(None,)), tf.TensorSpec(shape=(None,)))))
    trainer = ModelTrainer(model, optimizer, tf.keras.losses.TripletSemiHardLoss())
    trainer.train(train_loader, settings["epochs"])
    trainer.evaluate(test_loader)
    save_model(model, os.path.join(settings["results_dir"], "triplet_model.h5"))

def add_scheduler(optimizer, initial_lr, decay_steps, decay_rate):
    return optimizers.schedules.ExponentialDecay(initial_lr, decay_steps, decay_rate)

def add_checkpoint(model, optimizer, checkpoint_dir, max_to_keep=2):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save(os.path.join(checkpoint_dir, f"checkpoint_{len(os.listdir(checkpoint_dir))}.ckpt"))

if __name__ == "__main__":
    execute_pipeline()