import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os

# Custom dataset class to handle the triplets data
class CustomDataset(Dataset):
    # Initialize the dataset with triplets, max sequence length, tokenizer, and batch size
    def __init__(self, triplets, max_sequence_length, tokenizer, batch_size=32):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    # Get the length of the dataset
    def __len__(self):
        return len(self.triplets) // self.batch_size + 1

    # Get a batch of data from the dataset
    def __getitem__(self, idx):
        batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs = []
        attention_masks = []
        for triplet in batch_triplets:
            # Encode the anchor, positive, and negative texts using the tokenizer
            anchor = self.tokenizer.encode_plus(
                triplet['anchor'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            positive = self.tokenizer.encode_plus(
                triplet['positive'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            negative = self.tokenizer.encode_plus(
                triplet['negative'],
                max_length=self.max_sequence_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            # Append the encoded inputs and attention masks to the lists
            inputs.extend([anchor['input_ids'].squeeze(0), positive['input_ids'].squeeze(0), negative['input_ids'].squeeze(0)])
            attention_masks.extend([anchor['attention_mask'].squeeze(0), positive['attention_mask'].squeeze(0), negative['attention_mask'].squeeze(0)])
        # Return the batch of data as a dictionary
        return {'input_ids': torch.stack(inputs), 'attention_mask': torch.stack(attention_masks)}

# Function to build the model
def build_model(embedding_size, fully_connected_size, dropout_rate, max_sequence_length):
    # Define the model class
    class Model(nn.Module):
        # Initialize the model
        def __init__(self):
            super(Model, self).__init__()
            # Embedding layer
            self.embedding = nn.Embedding(10000, embedding_size)
            # Adaptive average pooling layer
            self.pooling = nn.AdaptiveAvgPool1d(1)
            # Dropout layer
            self.dropout = nn.Dropout(dropout_rate)
            # Fully connected layer 1
            self.fc1 = nn.Linear(embedding_size, fully_connected_size)
            # Fully connected layer 2
            self.fc2 = nn.Linear(fully_connected_size, embedding_size)

        # Forward pass
        def forward(self, x):
            # Embed the input
            x = self.embedding(x)
            # Apply adaptive average pooling
            x = self.pooling(x.permute(0, 2, 1)).squeeze(2)
            # Apply dropout
            x = self.dropout(x)
            # Apply ReLU activation to the first fully connected layer
            x = torch.relu(self.fc1(x))
            # Apply the second fully connected layer
            x = self.fc2(x)
            return x
    # Return an instance of the model
    return Model()

# Function to train the model
def train(model, device, dataset, epochs, learning_rate_value):
    # Define the mean squared error loss function
    criterion = nn.MSELoss()
    # Define the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_value)
    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        total_loss = 0
        # Iterate over the batches in the dataset
        for batch in dataset:
            # Move the input and attention mask to the device
            inputs = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            # Split the inputs and attention masks into anchor, positive, and negative
            inputs = torch.split(inputs, inputs.size(0) // 3, dim=0)
            attention_masks = torch.split(attention_masks, attention_masks.size(0) // 3, dim=0)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            anchor_embeddings = model(inputs[0])
            positive_embeddings = model(inputs[1])
            negative_embeddings = model(inputs[2])
            # Calculate the loss
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            # Backward pass
            loss.backward()
            # Update the model parameters
            optimizer.step()
            # Add the loss to the total loss
            total_loss += loss.item()
        # Print the average loss for the epoch
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataset)}')

# Function to evaluate the model
def evaluate(model, device, dataset):
    total_correct = 0
    # Evaluate the model on the test dataset
    with torch.no_grad():
        # Iterate over the batches in the dataset
        for batch in dataset:
            # Move the input and attention mask to the device
            inputs = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            # Split the inputs and attention masks into anchor, positive, and negative
            inputs = torch.split(inputs, inputs.size(0) // 3, dim=0)
            attention_masks = torch.split(attention_masks, attention_masks.size(0) // 3, dim=0)
            # Forward pass
            anchor_embeddings = model(inputs[0])
            positive_embeddings = model(inputs[1])
            negative_embeddings = model(inputs[2])
            # Iterate over the anchor embeddings
            for i in range(len(anchor_embeddings)):
                # Calculate the similarity between the anchor and positive embeddings
                similarity_positive = torch.dot(anchor_embeddings[i], positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                # Calculate the similarity between the anchor and negative embeddings
                similarity_negative = torch.dot(anchor_embeddings[i], negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                # Check if the similarity between the anchor and positive is greater than the similarity between the anchor and negative
                total_correct += similarity_positive > similarity_negative
    # Calculate the accuracy
    accuracy = total_correct / (len(dataset) * 32)
    # Print the accuracy
    print(f'Test Accuracy: {accuracy}')

# Function to calculate the loss
def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    # Calculate the squared distance between the anchor and positive embeddings
    positive_distance = torch.mean(torch.pow(anchor_embeddings - positive_embeddings, 2))
    # Calculate the squared distance between the anchor and negative embeddings
    negative_distance = torch.mean(torch.pow(anchor_embeddings - negative_embeddings, 2))
    # Return the sum of the positive distance and the maximum of the negative distance and zero
    return positive_distance + torch.max(negative_distance - positive_distance, torch.tensor(0.0))

# Function to load data from a file
def load_data(file_path):
    # Check if the file is a numpy file
    if file_path.endswith('.npy'):
        # Load the data from the numpy file
        return np.load(file_path, allow_pickle=True)
    # Check if the file is a JSON file
    else:
        # Load the data from the JSON file
        return json.load(open(file_path, 'r', encoding='utf-8'))

# Function to load snippets from a folder
def load_snippets(folder_path):
    # Initialize an empty list to store the snippet paths
    snippet_paths = []
    # Iterate over the subfolders in the folder
    for folder in os.listdir(folder_path):
        # Check if the subfolder is a directory
        if os.path.isdir(os.path.join(folder_path, folder)):
            # Append the snippet path to the list
            snippet_paths.append((os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')))
    # Return the list of snippet paths
    return snippet_paths

# Function to separate the snippets into bug and non-bug snippets
def separate_snippets(snippets):
    # Initialize empty lists to store the bug and non-bug snippets
    bug_snippets = []
    non_bug_snippets = []
    # Iterate over the snippet paths
    for folder_path, snippet_file_path in snippets:
        # Load the snippet data from the JSON file
        snippet_data = load_data(snippet_file_path)
        # Check if the snippet is a bug snippet
        if snippet_data.get('is_bug', False):
            # Append the snippet to the bug snippets list
            bug_snippets.append((snippet_data['snippet'], True))
        else:
            # Append the snippet to the non-bug snippets list
            non_bug_snippets.append((snippet_data['snippet'], False))
    # Return the bug and non-bug snippets as tuples
    return tuple(map(list, zip(*[(bug_snippets, non_bug_snippets)])))

# Function to create triplets from the problem statement, positive snippets, negative snippets, and number of negatives per positive
def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    # Initialize an empty list to store the triplets
    triplets = []
    # Iterate over the positive snippets
    for positive_doc in positive_snippets:
        # Iterate over the range of the number of negatives per positive
        for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
            # Append the triplet to the list
            triplets.append({'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)})
    # Return the list of triplets
    return triplets

# Main function
def main():
    # Set the dataset path
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    # Set the snippet folder path
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    # Set the number of negatives per positive
    num_negatives_per_positive = 1
    # Set the embedding size
    embedding_size = 128
    # Set the fully connected size
    fully_connected_size = 64
    # Set the dropout rate
    dropout_rate = 0.2
    # Set the maximum sequence length
    max_sequence_length = 512
    # Set the learning rate value
    learning_rate_value = 1e-5
    # Set the number of epochs
    epochs = 5
    # Set the batch size
    batch_size = 32

    # Set the device to the GPU if available, otherwise the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the instance ID map from the dataset
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    # Load the snippets from the snippet folder
    snippets = load_snippets(snippet_folder_path)
    # Initialize an empty list to store the triplets
    triplets = []
    # Iterate over the snippet paths
    for folder_path, _ in snippets:
        # Separate the snippets into bug and non-bug snippets
        bug_snippets, non_bug_snippets = separate_snippets([(folder_path, os.path.join(folder_path, 'snippet.json'))])
        # Get the problem statement from the instance ID map
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        # Iterate over the bug and non-bug snippets
        for bug_snippet, non_bug_snippet in [(bug, non_bug) for bug in bug_snippets for non_bug in non_bug_snippets]:
            # Create triplets from the problem statement, bug snippet, non-bug snippet, and number of negatives per positive
            triplets.extend(create_triplets(problem_statement, [bug_snippet], non_bug_snippets, num_negatives_per_positive))

    # Split the triplets into training and testing sets
    train_triplets, test_triplets = torch.utils.data.random_split(triplets, [int(len(triplets)*0.8), len(triplets)-int(len(triplets)*0.8)])
    # Load the BERT tokenizer
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
    # Create custom datasets for the training and testing sets
    train_data = CustomDataset(train_triplets, max_sequence_length, tokenizer, batch_size=batch_size)
    test_data = CustomDataset(test_triplets, max_sequence_length, tokenizer, batch_size=batch_size)
    # Create data loaders for the training and testing sets
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    # Build the model
    model = build_model(embedding_size, fully_connected_size, dropout_rate, max_sequence_length)
    # Move the model to the device
    model.to(device)
    # Train the model
    train(model, device, train_loader, epochs, learning_rate_value)
    # Evaluate the model
    evaluate(model, device, test_loader)

# Run the main function
if __name__ == "__main__":
    main()