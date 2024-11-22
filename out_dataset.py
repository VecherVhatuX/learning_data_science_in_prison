import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os

# Custom Dataset class to handle data loading and preprocessing
class CustomDataset(Dataset):
    def __init__(self, triplets, max_sequence_length, tokenizer, batch_size=32):
        """
        Initializes the CustomDataset class.

        Args:
            triplets (list): List of triplets containing anchor, positive, and negative snippets.
            max_sequence_length (int): Maximum sequence length for tokenization.
            tokenizer (object): Tokenizer object for text preprocessing.
            batch_size (int): Batch size for data loading. Defaults to 32.
        """
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.triplets) // self.batch_size + 1

    def __getitem__(self, idx):
        """
        Returns a batch of preprocessed data.

        Args:
            idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing input IDs and attention masks.
        """
        batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs = []
        attention_masks = []
        for triplet in batch_triplets:
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
            inputs.extend([anchor['input_ids'].squeeze(0), positive['input_ids'].squeeze(0), negative['input_ids'].squeeze(0)])
            attention_masks.extend([anchor['attention_mask'].squeeze(0), positive['attention_mask'].squeeze(0), negative['attention_mask'].squeeze(0)])
        return {'input_ids': torch.stack(inputs), 'attention_mask': torch.stack(attention_masks)}

# Model class for training and evaluation
class Model(nn.Module):
    def __init__(self, embedding_size, fully_connected_size, dropout_rate):
        """
        Initializes the Model class.

        Args:
            embedding_size (int): Size of the embedding layer.
            fully_connected_size (int): Size of the fully connected layer.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(Model, self).__init__()
        self.embedding = nn.Embedding(30522, embedding_size) 
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_size, fully_connected_size)
        self.fc2 = nn.Linear(fully_connected_size, embedding_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor.
        """
        x = self.embedding(x)
        x = self.pooling(x.permute(0, 2, 1)).squeeze(2)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# TripletDataset class for handling triplet data
class TripletDataset(Dataset):
    def __init__(self, triplets, num_negatives_per_positive, batch_size):
        """
        Initializes the TripletDataset class.

        Args:
            triplets (list): List of triplets containing anchor, positive, and negative snippets.
            num_negatives_per_positive (int): Number of negative samples per positive sample.
            batch_size (int): Batch size for data loading.
        """
        self.triplets = triplets
        self.num_negatives_per_positive = num_negatives_per_positive
        self.batch_size = batch_size
        self.positive_snippets = [triplet['positive'] for triplet in triplets]
        self.negative_snippets = [triplet['negative'] for triplet in triplets]
        self.anchor = [triplet['anchor'] for triplet in triplets]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.anchor) // self.batch_size + 1

    def __getitem__(self, idx):
        """
        Returns a batch of triplets.

        Args:
            idx (int): Index of the batch.

        Returns:
            list: List of triplets.
        """
        batch_anchor = self.anchor[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_positive_snippets = self.positive_snippets[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_negative_snippets = self.negative_snippets[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_triplets = []
        for i in range(len(batch_anchor)):
            for _ in range(self.num_negatives_per_positive):
                batch_triplets.append({'anchor': batch_anchor[i], 'positive': batch_positive_snippets[i], 'negative': random.choice(self.negative_snippets)})
        return batch_triplets

# Function to train the model
def train(model, device, dataset, epochs, learning_rate_value, max_sequence_length, tokenizer, batch_size):
    """
    Trains the model.

    Args:
        model (object): Model object.
        device (object): Device object.
        dataset (object): Dataset object.
        epochs (int): Number of epochs.
        learning_rate_value (float): Learning rate value.
        max_sequence_length (int): Maximum sequence length.
        tokenizer (object): Tokenizer object.
        batch_size (int): Batch size.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_value)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch_triplets in enumerate(dataset):
            batch_triplets = batch_triplets
            inputs = []
            attention_masks = []
            for triplet in batch_triplets:
                anchor = tokenizer.encode_plus(
                    triplet['anchor'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                positive = tokenizer.encode_plus(
                    triplet['positive'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                negative = tokenizer.encode_plus(
                    triplet['negative'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                inputs.extend([anchor['input_ids'].squeeze(0), positive['input_ids'].squeeze(0), negative['input_ids'].squeeze(0)])
                attention_masks.extend([anchor['attention_mask'].squeeze(0), positive['attention_mask'].squeeze(0), negative['attention_mask'].squeeze(0)])
            inputs = torch.stack(inputs)
            attention_masks = torch.stack(attention_masks)
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            inputs = torch.split(inputs, inputs.size(0) // 3, dim=0)
            attention_masks = torch.split(attention_masks, attention_masks.size(0) // 3, dim=0)
            optimizer.zero_grad()
            anchor_embeddings = model(inputs[0])
            positive_embeddings = model(inputs[1])
            negative_embeddings = model(inputs[2])
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataset)}')

# Function to evaluate the model
def evaluate(model, device, dataset, max_sequence_length, tokenizer, batch_size):
    """
    Evaluates the model.

    Args:
        model (object): Model object.
        device (object): Device object.
        dataset (object): Dataset object.
        max_sequence_length (int): Maximum sequence length.
        tokenizer (object): Tokenizer object.
        batch_size (int): Batch size.
    """
    total_correct = 0
    with torch.no_grad():
        for batch_idx, batch_triplets in enumerate(dataset):
            batch_triplets = batch_triplets
            inputs = []
            attention_masks = []
            for triplet in batch_triplets:
                anchor = tokenizer.encode_plus(
                    triplet['anchor'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                positive = tokenizer.encode_plus(
                    triplet['positive'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                negative = tokenizer.encode_plus(
                    triplet['negative'],
                    max_length=max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                inputs.extend([anchor['input_ids'].squeeze(0), positive['input_ids'].squeeze(0), negative['input_ids'].squeeze(0)])
                attention_masks.extend([anchor['attention_mask'].squeeze(0), positive['attention_mask'].squeeze(0), negative['attention_mask'].squeeze(0)])
            inputs = torch.stack(inputs)
            attention_masks = torch.stack(attention_masks)
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            inputs = torch.split(inputs, inputs.size(0) // 3, dim=0)
            attention_masks = torch.split(attention_masks, attention_masks.size(0) // 3, dim=0)
            anchor_embeddings = model(inputs[0])
            positive_embeddings = model(inputs[1])
            negative_embeddings = model(inputs[2])
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.dot(anchor_embeddings[i], positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.dot(anchor_embeddings[i], negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
    accuracy = total_correct / ((len(dataset) * batch_size) // 3)
    print(f'Test Accuracy: {accuracy}')

# Function to calculate loss
def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    """
    Calculates the loss.

    Args:
        anchor_embeddings (tensor): Anchor embeddings.
        positive_embeddings (tensor): Positive embeddings.
        negative_embeddings (tensor): Negative embeddings.

    Returns:
        tensor: Loss tensor.
    """
    positive_distance = torch.mean(torch.pow(anchor_embeddings - positive_embeddings, 2))
    negative_distance = torch.mean(torch.pow(anchor_embeddings - negative_embeddings, 2))
    return positive_distance + torch.max(negative_distance - positive_distance, torch.tensor(0.0))

# Function to load data
def load_data(file_path):
    """
    Loads data from a file.

    Args:
        file_path (str): File path.

    Returns:
        object: Loaded data.
    """
    if file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    else:
        return json.load(open(file_path, 'r', encoding='utf-8'))

# Function to load snippets
def load_snippets(folder_path):
    """
    Loads snippets from a folder.

    Args:
        folder_path (str): Folder path.

    Returns:
        list: List of snippet paths.
    """
    snippet_paths = []
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)):
            snippet_paths.append((os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')))
    return snippet_paths

# Function to separate snippets
def separate_snippets(snippets):
    """
    Separates snippets into bug and non-bug snippets.

    Args:
        snippets (list): List of snippet paths.

    Returns:
        tuple: Tuple of bug and non-bug snippets.
    """
    bug_snippets = []
    non_bug_snippets = []
    for folder_path, snippet_file_path in snippets:
        snippet_data = load_data(snippet_file_path)
        if snippet_data.get('is_bug', False):
            bug_snippets.append((snippet_data['snippet'], True))
        else:
            non_bug_snippets.append((snippet_data['snippet'], False))
    return tuple(map(list, zip(*[(bug_snippets, non_bug_snippets)])))

# Function to create triplets
def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    """
    Creates triplets from positive and negative snippets.

    Args:
        problem_statement (str): Problem statement.
        positive_snippets (list): List of positive snippets.
        negative_snippets (list): List of negative snippets.
        num_negatives_per_positive (int): Number of negative samples per positive sample.

    Returns:
        list: List of triplets.
    """
    triplets = []
    for positive_doc in positive_snippets:
        for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
            triplets.append({'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)})
    return triplets

# Main function
def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    num_negatives_per_positive = 1
    embedding_size = 128
    fully_connected_size = 64
    dropout_rate = 0.2
    max_sequence_length = 512
    learning_rate_value = 1e-5
    epochs = 5
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    snippets = load_snippets(snippet_folder_path)
    triplets = []
    for folder_path, _ in snippets:
        bug_snippets, non_bug_snippets = separate_snippets([(folder_path, os.path.join(folder_path, 'snippet.json'))])
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        for bug_snippet, non_bug_snippet in [(bug, non_bug) for bug in bug_snippets for non_bug in non_bug_snippets]:
            triplets.extend(create_triplets(problem_statement, [bug_snippet], non_bug_snippets, num_negatives_per_positive))
    train_triplets, test_triplets = torch.utils.data.random_split(triplets, [int(len(triplets)*0.8), len(triplets)-int(len(triplets)*0.8)])
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
    train_data = TripletDataset(train_triplets, num_negatives_per_positive, batch_size)
    test_data = TripletDataset(test_triplets, num_negatives_per_positive, batch_size)
    model = Model(embedding_size, fully_connected_size, dropout_rate)
    model.to(device)
    train(model, device, train_data, epochs, learning_rate_value, max_sequence_length, tokenizer, batch_size)
    evaluate(model, device, test_data, max_sequence_length, tokenizer, batch_size)

if __name__ == "__main__":
    main()