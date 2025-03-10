import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class ModelConfig:
    def __init__(self):
        self.model_base = "t5-base"  # Base model architecture
        self.conversation_format = "none"  # Format for conversation input
        self.low_rank_alpha = 16  # Alpha for low-rank approximations
        self.low_rank_dropout = 0.1  # Dropout rate for low-rank layers
        self.low_rank_rank = 64  # Rank for low-rank decomposition
        self.target_layers = "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"  # Layers to apply low-rank methods
        self.nested_quantization = False  # Flag for nested quantization
        self.four_bit_dtype = "float16"  # Data type for four-bit quantization
        self.four_bit_storage_dtype = "uint8"  # Storage type for four-bit quantization
        self.four_bit_quantization = "nf4"  # Method for four-bit quantization
        self.flash_attention = False  # Enable flash attention mechanism
        self.peft_low_rank = False  # Enable low-rank PEFT
        self.eight_bit_quantization = False  # Flag for eight-bit quantization
        self.four_bit_quantization_enabled = False  # Enable four-bit quantization
        self.reentrant_training = False  # Flag for reentrant training
        self.unsloth_training = False  # Flag for unsloth training
        self.triplet_loss_training = True  # Enable triplet loss during training
        self.dataset = "timdettmers/openassistant-guanaco"  # Dataset source
        self.append_special_token = False  # Append special tokens flag
        self.add_special_tokens = False  # Add special tokens flag
        self.dataset_splits = "train,test"  # Split identifiers for dataset
        self.tokenized_data_path = None  # Path for tokenized data
        self.output_dir = "./results"  # Directory for saving results
        self.num_epochs = 3  # Number of training epochs
        self.train_batch_size = 16  # Batch size for training
        self.eval_batch_size = 64  # Batch size for evaluation
        self.warmup_steps = 500  # Number of warmup steps for learning rate
        self.weight_decay = 0.01  # Weight decay factor for optimization
        self.log_dir = "./logs"  # Directory for logging
        self.save_steps = 500  # Frequency of model saving
        self.max_checkpoints = 2  # Maximum number of checkpoints to keep
        self.random_seed = 42  # Seed for randomness
        self.resume_checkpoint = None  # Path to resume training from checkpoint
        self.negative_samples = 5  # Number of negative samples per batch


class TripletModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.lstm = nn.LSTM(embedding_dim, batch_first=True)  # LSTM layer for sequence processing
        self.dense = nn.Linear(embedding_dim, embedding_dim)  # Dense layer for transformation
        self.output_dense = nn.Linear(embedding_dim, vocab_size)  # Output layer for generating predictions

    def forward(self, x):
        x = self.embedding(x)  # Convert input to embeddings
        x, _ = self.lstm(x)  # Process embeddings through LSTM
        x = self.dense(x[:, -1, :])  # Apply dense layer to the last LSTM output
        x = self.output_dense(x)  # Generate final output
        return x

    def compute_triplet_loss(self, anchor, positive, negative):
        # Compute the triplet loss for anchor-positive-negative triplets
        return torch.mean(torch.clamp(torch.mean((anchor - positive) ** 2) - torch.mean((anchor - negative) ** 2) + 2.0, min=0.0))


class TripletDataset(Dataset):
    def __init__(self, data, config, tokenizer):
        self.data = data  # Initialize dataset with data
        self.config = config  # Store model configuration
        self.tokenizer = tokenizer  # Tokenizer for encoding text
        self.indices = list(range(len(data)))  # List of indices for data
        self.batch_size = config.train_batch_size  # Batch size from configuration

    def __len__(self):
        # Return the number of batches in the dataset
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        # Retrieve a batch of data for training
        input_ids = []
        labels = []
        negative_examples = []
        for i in range(self.batch_size):
            example_index = self.indices[index * self.batch_size + i]  # Get example index
            example = self.data[example_index]  # Fetch example data
            input_ids.append(self.tokenizer.encode(example['input'], max_length=512, padding='max_length', truncation=True))  # Encode input
            labels.append(self.tokenizer.encode(example['output'], max_length=512, padding='max_length', truncation=True))  # Encode output
            for _ in range(self.config.negative_samples):
                negative_index = random.choice([j for j in range(len(self.data)) if j != example_index])  # Choose a random negative example
                negative_example = self.tokenizer.encode(self.data[negative_index]['input'], max_length=512, padding='max_length', truncation=True)  # Encode negative example
                negative_examples.append(negative_example)  # Add to negative examples list
        return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(negative_examples))  # Return tensors

    def on_epoch_end(self):
        # Shuffle indices after each epoch
        random.seed(self.config.random_seed)  # Set random seed for reproducibility
        self.indices = random.sample(range(len(self.data)), len(self.data))  # Shuffle indices


def load_data(file_path):
    # Load data from a JSON file
    try:
        with open(file_path, 'r') as file:
            return json.load(file)  # Return loaded data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")  # Handle file not found error
        return None


def train_model(model, config, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Initialize Adam optimizer
    model.train()  # Set model to training mode
    for epoch in range(config.num_epochs):
        for input_ids, labels, negative_examples in train_loader:
            optimizer.zero_grad()  # Reset gradients
            anchor = model(input_ids)  # Forward pass for anchors
            positive = model(labels)  # Forward pass for positive samples
            negative = model(negative_examples)  # Forward pass for negative samples
            loss = model.compute_triplet_loss(anchor, positive, negative)  # Calculate triplet loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
        print(f"Epoch {epoch + 1}/{config.num_epochs} completed.")  # Log epoch completion


def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss
    with torch.no_grad():  # Disable gradient calculations
        for input_ids, labels, negative_examples in test_loader:
            anchor = model(input_ids)  # Forward pass for anchors
            positive = model(labels)  # Forward pass for positive samples
            negative = model(negative_examples)  # Forward pass for negative samples
            loss = model.compute_triplet_loss(anchor, positive, negative)  # Calculate triplet loss
            total_loss += loss.item()  # Accumulate loss
    average_loss = total_loss / len(test_loader)  # Compute average loss
    print(f"Average Test Loss: {average_loss:.4f}")  # Log average loss


def load_tokenizer():
    # Load the BERT tokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")  # Return the tokenizer


def main():
    config = ModelConfig()  # Instantiate model configuration
    train_data = load_data("train.json")  # Load training data
    test_data = load_data("test.json")  # Load testing data
    tokenizer = load_tokenizer()  # Load tokenizer
    train_dataset = TripletDataset(train_data, config, tokenizer)  # Prepare training dataset
    test_dataset = TripletDataset(test_data, config, tokenizer)  # Prepare testing dataset
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)  # DataLoader for training
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)  # DataLoader for testing
    model = TripletModel(128, 30522)  # Initialize the triplet model
    train_model(model, config, train_loader)  # Train the model
    evaluate_model(model, test_loader)  # Evaluate the model


if __name__ == "__main__":
    main()  # Execute main function