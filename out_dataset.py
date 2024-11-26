import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

class DataUtil:
    """
    A utility class for loading and processing data.
    """
    @staticmethod
    def load_data(file_path: str) -> dict or np.ndarray:
        """
        Load data from a file.

        Args:
        file_path (str): The path to the file.

        Returns:
        dict or np.ndarray: The loaded data.
        """
        # Load data from a JSON or NumPy file
        if file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True)
        else:
            return json.load(open(file_path, 'r', encoding='utf-8'))

    @staticmethod
    def load_snippets(snippet_folder_path: str) -> list:
        """
        Load snippets from a folder.

        Args:
        snippet_folder_path (str): The path to the folder.

        Returns:
        list: A list of tuples containing the folder path and snippet file path.
        """
        # Load snippets from a folder
        return [(os.path.join(snippet_folder_path, folder), os.path.join(snippet_folder_path, folder, 'snippet.json')) 
                for folder in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, folder))]

    @staticmethod
    def separate_snippets(snippets: list) -> tuple:
        """
        Separate snippets into bug and non-bug snippets.

        Args:
        snippets (list): A list of tuples containing the folder path and snippet file path.

        Returns:
        tuple: A tuple containing two lists of bug and non-bug snippets.
        """
        # Separate snippets into bug and non-bug snippets
        bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                        for snippet_data in [DataUtil.load_data(snippet_file_path)] if snippet_data.get('is_bug', False)]
        non_bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                            for snippet_data in [DataUtil.load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)]
        return bug_snippets, non_bug_snippets


class TripletUtil:
    """
    A utility class for creating triplets.
    """
    @staticmethod
    def create_triplets(num_negatives_per_positive: int, instance_id_map: dict, snippets: list) -> list:
        """
        Create triplets from snippets.

        Args:
        num_negatives_per_positive (int): The number of negatives per positive.
        instance_id_map (dict): A dictionary mapping instance IDs to problem statements.
        snippets (list): A list of tuples containing the folder path and snippet file path.

        Returns:
        list: A list of triplets.
        """
        # Create triplets from snippets
        bug_snippets, non_bug_snippets = DataUtil.separate_snippets(snippets)
        return [{'anchor': instance_id_map[os.path.basename(folder_path)], 'positive': positive_doc, 'negative': random.choice(non_bug_snippets)} 
                for folder_path, _ in snippets 
                for positive_doc in bug_snippets 
                for _ in range(min(num_negatives_per_positive, len(non_bug_snippets)))]


class BugTripletDataset(Dataset):
    """
    A dataset class for bug triplets.
    """
    def __init__(self, triplets: list, max_sequence_length: int):
        """
        Initialize the dataset.

        Args:
        triplets (list): A list of triplets.
        max_sequence_length (int): The maximum sequence length.
        """
        # Initialize the dataset
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        int: The length of the dataset.
        """
        # Get the length of the dataset
        return len(self.triplets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a triplet from the dataset.

        Args:
        idx (int): The index of the triplet.

        Returns:
        torch.Tensor: The triplet.
        """
        # Get a triplet from the dataset
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']

        anchor_sequence = self.tokenizer.encode(anchor, max_length=self.max_sequence_length, padding='max_length', truncation=True, return_tensors='pt')
        positive_sequence = self.tokenizer.encode(positive, max_length=self.max_sequence_length, padding='max_length', truncation=True, return_tensors='pt')
        negative_sequence = self.tokenizer.encode(negative, max_length=self.max_sequence_length, padding='max_length', truncation=True, return_tensors='pt')

        return torch.cat([anchor_sequence, positive_sequence, negative_sequence], dim=0)

    def shuffle(self) -> None:
        """
        Shuffle the dataset.
        """
        # Shuffle the dataset
        random.shuffle(self.triplets)


class BugTripletModel(nn.Module):
    """
    A model class for bug triplets.
    """
    def __init__(self, embedding_size: int, fully_connected_size: int, dropout_rate: float):
        """
        Initialize the model.

        Args:
        embedding_size (int): The embedding size.
        fully_connected_size (int): The fully connected size.
        dropout_rate (float): The dropout rate.
        """
        # Initialize the model
        super(BugTripletModel, self).__init__()
        self.model = nn.Sequential(
            AutoModel.from_pretrained('bert-base-uncased'),
            nn.Linear(768, fully_connected_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fully_connected_size, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
        x (torch.Tensor): The input.

        Returns:
        torch.Tensor: The output.
        """
        # Forward pass
        outputs = self.model(x)
        return outputs.last_hidden_state[:, 0, :]


def calculate_loss(anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Calculate the loss.

    Args:
    anchor_embeddings (torch.Tensor): The anchor embeddings.
    positive_embeddings (torch.Tensor): The positive embeddings.
    negative_embeddings (torch.Tensor): The negative embeddings.
    device (torch.device): The device.

    Returns:
    torch.Tensor: The loss.
    """
    # Calculate the loss
    return torch.mean((anchor_embeddings - positive_embeddings) ** 2) + torch.max(torch.mean((anchor_embeddings - negative_embeddings) ** 2) - torch.mean((anchor_embeddings - positive_embeddings) ** 2), torch.tensor(0.0).to(device))


def train(model: nn.Module, dataset: Dataset, learning_rate_value: float, epochs: int, batch_size: int, device: torch.device) -> float:
    """
    Train the model.

    Args:
    model (nn.Module): The model.
    dataset (Dataset): The dataset.
    learning_rate_value (float): The learning rate.
    epochs (int): The number of epochs.
    batch_size (int): The batch size.
    device (torch.device): The device.

    Returns:
    float: The total loss.
    """
    # Train the model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_value)
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            batch = batch.to(device)
            anchor_embeddings = model(batch[:, 0, :])
            positive_embeddings = model(batch[:, 1, :])
            negative_embeddings = model(batch[:, 2, :])
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(dataset)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataset)}')
    return total_loss / epochs


def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> float:
    """
    Evaluate the model.

    Args:
    model (nn.Module): The model.
    dataset (Dataset): The dataset.
    batch_size (int): The batch size.
    device (torch.device): The device.

    Returns:
    float: The accuracy.
    """
    # Evaluate the model
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            anchor_embeddings = model(batch[:, 0, :])
            positive_embeddings = model(batch[:, 1, :])
            negative_embeddings = model(batch[:, 2, :])
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
    return total_correct / len(dataset) / batch_size


def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    instance_id_map = {item['instance_id']: item['problem_statement'] for item in DataUtil.load_data(dataset_path)}
    snippets = DataUtil.load_snippets(snippet_folder_path)
    triplets = TripletUtil.create_triplets(1, instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    train_dataset = BugTripletDataset(train_triplets, 512)
    test_dataset = BugTripletDataset(test_triplets, 512)
    model = BugTripletModel(128, 64, 0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(5):
        loss = train(model, train_dataset, 1e-5, 1, 32, device)
        print(f'Epoch {epoch+1}, Loss: {loss}')
    print(f'Test Accuracy: {evaluate(model, test_dataset, 32, device)}')


if __name__ == "__main__":
    main()