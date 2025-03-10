import os
import json
from dataclasses import dataclass, field
from typing import Dict
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import backend as K
import numpy as np
import random
import tensorflow_text as tft

# Configuration settings for the model
class ModelConfig:
    def __init__(self):
        self.model_base = "t5-base"  # Base model to be utilized
        self.conversation_format = "none"  # Format for conversation handling
        self.low_rank_alpha = 16  # Alpha value for low-rank approximation
        self.low_rank_dropout = 0.1  # Dropout rate for low-rank layers
        self.low_rank_rank = 64  # Rank for low-rank matrices
        self.target_layers = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"  # Layers to target
        self.nested_quantization = False  # Indicates if nested quantization is used
        self.four_bit_dtype = "float16"  # Data type for four-bit representation
        self.four_bit_storage_dtype = "uint8"  # Storage data type for four-bit representation
        self.four_bit_quantization = "nf4"  # Method of four-bit quantization
        self.flash_attention = False  # Flag for flash attention optimization
        self.peft_low_rank = False  # Indicates if PEFT low-rank training is used
        self.eight_bit_quantization = False  # Flag for enabling eight-bit quantization
        self.four_bit_quantization_enabled = False  # Flag for enabling four-bit quantization
        self.reentrant_training = False  # Indicates if reentrant training is applied
        self.unsloth_training = False  # Flag for unsloth training
        self.triplet_loss_training = True  # Specifies if triplet loss is utilized
        self.dataset = "timdettmers/openassistant-guanaco"  # Dataset identifier
        self.append_special_token = False  # If special tokens should be appended
        self.add_special_tokens = False  # If special tokens should be added
        self.dataset_splits = "train,test"  # Identifies dataset splits
        self.tokenized_data_path = None  # Path for tokenized data storage
        self.output_dir = "./results"  # Directory for saving results
        self.num_epochs = 3  # Number of training epochs
        self.train_batch_size = 16  # Batch size for training
        self.eval_batch_size = 64  # Batch size for evaluation
        self.warmup_steps = 500  # Number of warm-up steps
        self.weight_decay = 0.01  # Weight decay factor
        self.log_dir = "./logs"  # Directory for log files
        self.save_steps = 500  # Steps interval for model saving
        self.max_checkpoints = 2  # Maximum number of checkpoints to retain
        self.random_seed = 42  # Seed for random number generation
        self.resume_checkpoint = None  # Checkpoint path to resume training
        self.negative_samples = 5  # Number of negative samples per training instance

# Triplet model class definition
class TripletModel(tf.keras.Model):
    def __init__(self, embedding_dim, vocab_size):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.lstm = layers.LSTM(embedding_dim, return_sequences=True)  # LSTM layer
        self.dense = layers.Dense(embedding_dim, activation='relu')  # Fully connected layer with ReLU
        self.output_dense = layers.Dense(vocab_size)  # Output layer for vocabulary size

    def call(self, x):
        x = self.embedding(x)  # Pass input through embedding
        x = self.lstm(x)  # Process through LSTM
        x = self.dense(x[:, -1, :])  # Apply dense layer to the last time step
        x = self.output_dense(x)  # Final output layer
        return x

    def compute_triplet_loss(self, anchor, positive, negative):
        # Calculate the triplet loss based on anchor, positive, and negative samples
        return tf.reduce_mean(tf.maximum(tf.reduce_mean((anchor - positive) ** 2) - tf.reduce_mean((anchor - negative) ** 2) + 2.0, 0.0))

# Dataset class for triplet samples
class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, data, config, tokenizer):
        self.data = data  # Dataset
        self.config = config  # Model configuration
        self.tokenizer = tokenizer  # Tokenizer for processing text
        self.indices = list(range(len(data)))  # List of indices for data access
        self.batch_size = config.train_batch_size  # Batch size from configuration

    def __len__(self):
        return len(self.data) // self.batch_size  # Number of batches

    def __getitem__(self, index):
        input_ids = []  # List to hold input IDs
        labels = []  # List to hold labels
        negative_examples = []  # List for negative examples
        for i in range(self.batch_size):  # Iterate over batch size
            example_index = self.indices[index * self.batch_size + i]  # Get example index
            example = self.data[example_index]  # Fetch example data
            input_ids.append(self.tokenizer.encode(example['input'], max_length=512, padding='max_length', truncation=True))  # Tokenize input
            labels.append(self.tokenizer.encode(example['output'], max_length=512, padding='max_length', truncation=True))  # Tokenize output
            for _ in range(self.config.negative_samples):  # Generate negative samples
                negative_index = random.randint(0, len(self.data) - 1)  # Randomly select negative index
                if negative_index == example_index:  # Prevent selecting the same index
                    negative_index = (negative_index + 1) % len(self.data)  # Adjust index
                negative_example = self.tokenizer.encode(self.data[negative_index]['input'], max_length=512, padding='max_length', truncation=True)  # Tokenize negative example
                negative_examples.append(negative_example)  # Store negative example
        input_ids = np.array(input_ids)  # Convert input IDs to numpy array
        labels = np.array(labels)  # Convert labels to numpy array
        negative_examples = np.array(negative_examples)  # Convert negative examples to numpy array
        return input_ids, labels, negative_examples  # Return the batch

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        random.seed(self.config.random_seed)  # Set random seed for consistency
        random.seed(random.randint(0, 2**32))  # Set a different random seed for shuffling
        self.indices = random.sample(range(len(self.data)), len(self.data))  # Shuffle dataset indices

# Function to load data from a JSON file
def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)  # Return loaded JSON data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")  # Handle file not found error
        return None  # Return None if file is not found

# Function to train the model
def train_model(model, config, train_dataset, test_dataset):
    optimizer = optimizers.Adam(learning_rate=0.001)  # Instantiate Adam optimizer
    model.compile(optimizer=optimizer, loss=model.compute_triplet_loss)  # Compile model with loss function
    model.fit(train_dataset, epochs=config.num_epochs, validation_data=test_dataset)  # Train the model

# Function to load the tokenizer
def load_tokenizer():
    return tft.BertTokenizer("bert-base-uncased-vocab.txt", return_special_tokens_mask=True)  # Initialize tokenizer

# Main execution function
def main():
    config = ModelConfig()  # Create model configuration
    train_data = load_data("train.json")  # Load training data
    test_data = load_data("test.json")  # Load testing data
    tokenizer = load_tokenizer()  # Load tokenizer
    train_dataset = TripletDataset(train_data, config, tokenizer)  # Create training dataset
    test_dataset = TripletDataset(test_data, config, tokenizer)  # Create testing dataset
    model = TripletModel(128, 30522)  # Initialize the triplet model
    train_model(model, config, train_dataset, test_dataset)  # Train the model

if __name__ == "__main__":
    main()  # Run the main function