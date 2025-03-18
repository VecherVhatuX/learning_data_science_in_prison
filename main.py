import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

# Configuration for the model, training, and data handling
ModelConfiguration = lambda: {
    "model_name": "t5-base",  # Name of the pre-trained model to use
    "alpha": 16,  # Alpha parameter for model configuration
    "dropout_rate": 0.1,  # Dropout rate for regularization
    "decomposition_rank": 64,  # Rank for decomposition techniques
    "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],  # Layers to apply modifications
    "quantization_config": {  # Configuration for quantization techniques
        "nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
        "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
        "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
        "unsloth_training": False,
    },
    "use_triplet_loss": True,  # Whether to use triplet loss for training
    "data_source": "timdettmers/openassistant-guanaco",  # Source of the dataset
    "token_flags": {"use_special_token": False, "include_special_tokens": False},  # Tokenization flags
    "data_splits": ["train", "test"],  # Data splits for training and evaluation
    "tokenized_file": None,  # Path to the tokenized file (if any)
    "results_dir": "./results",  # Directory to save results
    "epochs": 3,  # Number of training epochs
    "batch_sizes": {"train": 16, "eval": 64},  # Batch sizes for training and evaluation
    "warmup_steps": 500,  # Number of warmup steps for learning rate scheduling
    "weight_decay": 0.01,  # Weight decay for regularization
    "logging_dir": "./logs",  # Directory to save logs
    "model_save_interval": 500,  # Interval to save model checkpoints
    "max_checkpoints": 2,  # Maximum number of checkpoints to keep
    "seed": 42,  # Random seed for reproducibility
    "checkpoint_path": None,  # Path to load a checkpoint (if any)
    "negative_samples_per_batch": 5  # Number of negative samples per batch for triplet loss
}

# Neural network model for sentence embedding
class SentenceEmbedder(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(SentenceEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)  # LSTM layer
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)  # Fully connected layer
        self.fc2 = nn.Linear(embedding_dim, vocab_size)  # Output layer

    def forward(self, x):
        x = self.embedding(x)  # Convert input tokens to embeddings
        x, _ = self.lstm(x)  # Process embeddings through LSTM
        x = x[:, -1, :]  # Take the last hidden state
        x = self.fc1(x)  # Apply first fully connected layer
        x = self.fc2(x)  # Apply second fully connected layer
        return x  # Return the final output

# Function to calculate triplet loss
calculate_triplet_loss = lambda anchor, positive, negative: torch.mean(
    torch.maximum(torch.mean(torch.square(anchor - positive), dim=-1) - 
    torch.mean(torch.square(anchor - negative), dim=-1) + 2.0, torch.tensor(0.0)
)

# Dataset class for handling sentence data
class SentenceDataset(Dataset):
    def __init__(self, data, config, tokenizer):
        self.data = data  # Dataset
        self.config = config  # Configuration
        self.tokenizer = tokenizer  # Tokenizer for encoding
        self.indices = list(range(len(data)))  # List of indices for data access

    def __len__(self):
        return len(self.data)  # Return the size of the dataset

    def __getitem__(self, idx):
        input_ids = self.tokenizer.encode(self.data[self.indices[idx]]['input'], max_length=512, padding='max_length', truncation=True)  # Encode input
        labels = self.tokenizer.encode(self.data[self.indices[idx]]['output'], max_length=512, padding='max_length', truncation=True)  # Encode output
        neg_samples = [self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != self.indices[idx]])]['input'],
                       max_length=512, padding='max_length', truncation=True) for _ in range(self.config["negative_samples_per_batch"])]  # Generate negative samples
        return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(neg_samples)  # Return tensors

# Function to load data from a JSON file
def load_data(file_path):
    if os.path.exists(file_path):
        return json.load(open(file_path, 'r'))  # Load and return data
    else:
        print(f"File not found: {file_path}")  # Print error if file not found
        return None

# Function to initialize T5 tokenizer
initialize_t5_tokenizer = lambda: T5Tokenizer.from_pretrained("t5-base")

# Function to configure the optimizer
configure_optimizer = lambda model: optim.Adam(model.parameters(), lr=0.001)

# Function to set up training components
def setup_training(model, optimizer):
    return model, optimizer, calculate_triplet_loss

# Function to run the training loop
def run_training(model, config, data_loader, optimizer, loss_function):
    model.train()  # Set model to training mode
    for epoch in range(config["epochs"]):  # Iterate over epochs
        for batch in data_loader:  # Iterate over batches
            input_ids, labels, neg_samples = batch  # Unpack batch
            optimizer.zero_grad()  # Zero gradients
            outputs = model(input_ids)  # Forward pass
            loss = loss_function(outputs, labels, neg_samples)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
        if early_stopping(loss.item()):  # Check for early stopping
            print("Early stopping triggered.")
            break

# Function to run evaluation
def run_evaluation(model, data_loader, loss_function):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:  # Iterate over batches
            input_ids, labels, neg_samples = batch  # Unpack batch
            outputs = model(input_ids)  # Forward pass
            total_loss += loss_function(outputs, labels, neg_samples).item()  # Accumulate loss
    print(f"Mean Evaluation Loss: {total_loss / len(data_loader):.4f}")  # Print mean loss

# Function to save the trained model
def save_trained_model(model, file_path):
    torch.save(model.state_dict(), file_path)  # Save model state

# Function to save training history
def save_training_history(history, file_path):
    json.dump(history, open(file_path, 'w'))  # Save history as JSON

# Function to add a learning rate scheduler
def add_learning_rate_scheduler(optimizer, config):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # Exponential LR scheduler

# Function to configure early stopping
def configure_early_stopping(patience=3):
    best_loss = float('inf')
    counter = 0
    def early_stopping(current_loss):
        nonlocal best_loss, counter
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
        else:
            counter += 1
        return counter >= patience
    return early_stopping

# Main function to execute the training process
def execute_training():
    config = ModelConfiguration()  # Load configuration
    train_data = load_data("train.json")  # Load training data
    test_data = load_data("test.json")  # Load test data
    tokenizer = initialize_t5_tokenizer()  # Initialize tokenizer
    train_dataset = SentenceDataset(train_data, config, tokenizer)  # Create training dataset
    test_dataset = SentenceDataset(test_data, config, tokenizer)  # Create test dataset
    train_loader = DataLoader(train_dataset, batch_size=config["batch_sizes"]['train'], shuffle=True)  # Create training DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config["batch_sizes"]['eval'], shuffle=False)  # Create test DataLoader
    model = SentenceEmbedder(128, 30522)  # Initialize model
    optimizer = configure_optimizer(model)  # Configure optimizer
    model, optimizer, loss_function = setup_training(model, optimizer)  # Set up training
    scheduler = add_learning_rate_scheduler(optimizer, config)  # Add learning rate scheduler
    run_training(model, config, train_loader, optimizer, loss_function)  # Run training
    run_evaluation(model, test_loader, loss_function)  # Run evaluation
    save_trained_model(model, os.path.join(config["results_dir"], "triplet_model.pth"))  # Save trained model

if __name__ == "__main__":
    execute_training()  # Execute training process