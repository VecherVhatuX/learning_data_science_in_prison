import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import T5Tokenizer

class Config:
    def __init__(self):
        self.model_name = "t5-base"  # Base model identifier
        self.conversation_style = "none"  # Style of conversation to be used
        self.alpha = 16  # Hyperparameter for model adjustments
        self.dropout_rate = 0.1  # Rate for dropout regularization
        self.decomposition_rank = 64  # Rank for decomposition of tensors
        self.layers_to_modify = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]  # List of layers for modification
        self.enable_nested_quantization = False  # Flag to allow nested quantization
        self.dtype_for_four_bit = "float16"  # Data type for 4-bit quantization
        self.storage_dtype = "uint8"  # Data type for storage
        self.quantization_method = "nf4"  # Method used for quantization
        self.use_flash_attention = False  # Use of flash attention mechanism
        self.low_rank_peft = False  # Low-rank parameter-efficient fine-tuning flag
        self.enable_eight_bit_quantization = False  # Enable 8-bit quantization
        self.enable_four_bit_quantization = False  # Enable 4-bit quantization
        self.allow_reentrant_training = False  # Flag to allow reentrant training
        self.allow_unsloth_training = False  # Allow unsloth training
        self.use_triplet_loss = True  # Enable triplet loss in model training
        self.data_source = "timdettmers/openassistant-guanaco"  # Source of the training data
        self.special_token_flag = False  # Indicator for usage of special tokens
        self.special_tokens_inclusion = False  # Whether to include special tokens
        self.data_splits = ["train", "test"]  # Data splits for training and testing
        self.tokenized_data_file = None  # File to store tokenized data
        self.results_directory = "./results"  # Directory to save results
        self.num_epochs = 3  # Number of epochs for training
        self.batch_size_train = 16  # Batch size for training
        self.batch_size_eval = 64  # Batch size for evaluation
        self.warmup_steps_count = 500  # Number of warmup steps
        self.weight_decay_rate = 0.01  # Weight decay rate for regularization
        self.logging_directory = "./logs"  # Directory for logging training progress
        self.model_save_frequency = 500  # Frequency of saving model checkpoints
        self.max_checkpoints_to_keep = 2  # Maximum number of checkpoints to retain
        self.seed = 42  # Random seed for reproducibility
        self.checkpoint_resume_path = None  # Path to resume from a checkpoint
        self.negative_samples_per_batch = 5  # Number of negative samples in each batch

class TripletNet(models.Model):
    def __init__(self, embedding_dim, vocab_size):
        super(TripletNet, self).__init__()  # Initialize the parent class
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim)  # Embedding layer for input tokens
        self.lstm_layer = layers.LSTM(embedding_dim, return_sequences=True)  # LSTM layer for sequence processing
        self.fc_layer = layers.Dense(embedding_dim)  # Fully connected layer to process LSTM output
        self.output_layer = layers.Dense(vocab_size)  # Output layer to generate predictions

    def call(self, input_tensor):
        embedded = self.embedding_layer(input_tensor)  # Apply embedding layer
        lstm_out = self.lstm_layer(embedded)  # Process through LSTM
        dense_out = self.fc_layer(lstm_out[:, -1, :])  # Get output from the last LSTM cell
        output = self.output_layer(dense_out)  # Generate final output predictions
        return output

    def calculate_triplet_loss(self, anchor, positive, negative):
        # Calculate triplet loss based on anchor, positive, and negative examples
        return tf.reduce_mean(tf.maximum(tf.reduce_mean(tf.square(anchor - positive), axis=-1) - 
                                          tf.reduce_mean(tf.square(anchor - negative), axis=-1) + 2.0, 0.0))

class TripletData(tf.keras.utils.Sequence):
    def __init__(self, data, config, tokenizer):
        self.data = data  # Dataset to be used
        self.config = config  # Configuration settings
        self.tokenizer = tokenizer  # Tokenizer for processing text
        self.indices = list(range(len(data)))  # Indices for data samples
        self.batch_size = config.batch_size_train  # Batch size for training

    def __len__(self):
        return len(self.data) // self.batch_size  # Number of batches

    def __getitem__(self, index):
        # Prepare a batch of data for training
        input_ids, labels, negative_exs = zip(*[
            self.prepare_example(index * self.batch_size + i) for i in range(self.batch_size)
        ])
        return (np.array(input_ids), np.array(labels), np.array(negative_exs))

    def prepare_example(self, idx):
        # Prepare a single example with input, label, and negative samples
        example = self.data[idx]
        input_id = self.tokenizer.encode(example['input'], max_length=512, padding='max_length', truncation=True)  # Tokenize input
        label = self.tokenizer.encode(example['output'], max_length=512, padding='max_length', truncation=True)  # Tokenize label
        negative_exs = [
            self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != idx])]['input'], max_length=512, padding='max_length', truncation=True)
            ) for _ in range(self.config.negative_samples_per_batch)  # Generate negative examples
        ]
        return input_id, label, negative_exs

    def shuffle_indices(self):
        random.seed(self.config.seed)  # Set random seed for shuffling
        random.shuffle(self.indices)  # Shuffle indices for data

    def epoch_shuffle(self):
        self.shuffle_indices()  # Shuffle indices at the start of each epoch

def read_json(file_path):
    # Read and parse a JSON file
    try:
        with open(file_path, 'r') as f:
            return json.load(f)  # Load JSON data
    except FileNotFoundError:
        print(f"File not found: {file_path}")  # Handle file not found error
        return None

def train_triplet_model(model, config, train_loader):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam optimizer for training
    model.compile(optimizer=optimizer, loss=model.calculate_triplet_loss)  # Compile model with loss function
    model.fit(train_loader, epochs=config.num_epochs)  # Train model

def evaluate_triplet_model(model, test_loader):
    total_loss = 0  # Initialize total loss
    for input_ids, labels, negative_exs in test_loader:
        loss = model.calculate_triplet_loss(
            model(input_ids),
            model(labels),
            model(negative_exs)
        )  # Calculate loss for the batch
        total_loss += loss.numpy()  # Accumulate loss
    print(f"Test Loss Average: {total_loss / len(test_loader):.4f}")  # Print average test loss

def initialize_tokenizer():
    return T5Tokenizer.from_pretrained("t5-base")  # Initialize T5 tokenizer

def save_model(model, path):
    model.save_weights(path)  # Save model weights to specified path

def main():
    config = Config()  # Create configuration instance
    train_data = read_json("train.json")  # Load training data
    test_data = read_json("test.json")  # Load testing data
    tokenizer = initialize_tokenizer()  # Initialize tokenizer
    train_dataset = TripletData(train_data, config, tokenizer)  # Create training dataset
    test_dataset = TripletData(test_data, config, tokenizer)  # Create testing dataset

    model = TripletNet(128, 30522)  # Initialize triplet network model
    
    for _ in range(config.num_epochs):
        train_dataset.epoch_shuffle()  # Shuffle training data for each epoch
    
    train_triplet_model(model, config, train_dataset)  # Train the model
    evaluate_triplet_model(model, test_dataset)  # Evaluate the model
    save_model(model, os.path.join(config.results_directory, "triplet_model.h5"))  # Save the trained model

if __name__ == "__main__":
    main()  # Entry point of the script