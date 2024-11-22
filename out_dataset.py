import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os

class CustomDataset(Dataset):
    """
    Handles the triplets data.

    Args:
        triplets (list): List of triplets.
        max_sequence_length (int): Maximum sequence length.
        tokenizer (object): Tokenizer object.
        batch_size (int, optional): Batch size. Defaults to 32.
    """
    def __init__(self, triplets, max_sequence_length, tokenizer, batch_size=32):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    """
    Returns the length of the dataset.

    Returns:
        int: Length of the dataset.
    """
    def __len__(self):
        return len(self.triplets) // self.batch_size + 1

    """
    Returns a batch of data from the dataset.

    Args:
        idx (int): Index of the batch.

    Returns:
        dict: Batch of data.
    """
    def __getitem__(self, idx):
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

def build_model(embedding_size, fully_connected_size, dropout_rate, max_sequence_length):
    """
    Builds the model.

    Args:
        embedding_size (int): Embedding size.
        fully_connected_size (int): Fully connected size.
        dropout_rate (float): Dropout rate.
        max_sequence_length (int): Maximum sequence length.

    Returns:
        object: Model object.
    """
    class Model(nn.Module):
        """
        Model class.

        Args:
            embedding_size (int): Embedding size.
            fully_connected_size (int): Fully connected size.
            dropout_rate (float): Dropout rate.
            max_sequence_length (int): Maximum sequence length.
        """
        def __init__(self):
            super(Model, self).__init__()
            self.embedding = nn.Embedding(10000, embedding_size)
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(embedding_size, fully_connected_size)
            self.fc2 = nn.Linear(fully_connected_size, embedding_size)

        """
        Forward pass.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor.
        """
        def forward(self, x):
            x = self.embedding(x)
            x = self.pooling(x.permute(0, 2, 1)).squeeze(2)
            x = self.dropout(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return Model()

def train(model, device, dataset, epochs, learning_rate_value):
    """
    Trains the model.

    Args:
        model (object): Model object.
        device (str): Device to use.
        dataset (object): Dataset object.
        epochs (int): Number of epochs.
        learning_rate_value (float): Learning rate value.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_value)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            inputs = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
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

def evaluate(model, device, dataset):
    """
    Evaluates the model.

    Args:
        model (object): Model object.
        device (str): Device to use.
        dataset (object): Dataset object.
    """
    total_correct = 0
    with torch.no_grad():
        for batch in dataset:
            inputs = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            inputs = torch.split(inputs, inputs.size(0) // 3, dim=0)
            attention_masks = torch.split(attention_masks, attention_masks.size(0) // 3, dim=0)
            anchor_embeddings = model(inputs[0])
            positive_embeddings = model(inputs[1])
            negative_embeddings = model(inputs[2])
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.dot(anchor_embeddings[i], positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.dot(anchor_embeddings[i], negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += similarity_positive > similarity_negative
    accuracy = total_correct / (len(dataset) * 32)
    print(f'Test Accuracy: {accuracy}')

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

def separate_snippets(snippets):
    """
    Separates snippets into bug and non-bug snippets.

    Args:
        snippets (list): List of snippets.

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

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    """
    Creates triplets from the problem statement, positive snippets, negative snippets, and number of negatives per positive.

    Args:
        problem_statement (str): Problem statement.
        positive_snippets (list): List of positive snippets.
        negative_snippets (list): List of negative snippets.
        num_negatives_per_positive (int): Number of negatives per positive.

    Returns:
        list: List of triplets.
    """
    triplets = []
    for positive_doc in positive_snippets:
        for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
            triplets.append({'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)})
    return triplets

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
    train_data = CustomDataset(train_triplets, max_sequence_length, tokenizer, batch_size=batch_size)
    test_data = CustomDataset(test_triplets, max_sequence_length, tokenizer, batch_size=batch_size)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = build_model(embedding_size, fully_connected_size, dropout_rate, max_sequence_length)
    model.to(device)
    train(model, device, train_loader, epochs, learning_rate_value)
    evaluate(model, device, test_loader)

if __name__ == "__main__":
    main()