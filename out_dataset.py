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

class DataProcessor:
    @staticmethod
    def load_data(file_path):
        """Retrieves data from a specified file path."""
        if file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True)
        else:
            return json.load(open(file_path, 'r', encoding='utf-8'))

    @staticmethod
    def load_snippets(snippet_folder_path):
        """Loads snippets from a folder and returns their paths."""
        return [(os.path.join(snippet_folder_path, folder), os.path.join(snippet_folder_path, folder, 'snippet.json')) 
                for folder in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, folder))]

    @staticmethod
    def separate_snippets(snippets):
        """Separates snippets into bug and non-bug categories."""
        bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                        for snippet_data in [DataProcessor.load_data(snippet_file_path)] if snippet_data.get('is_bug', False)]
        non_bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                            for snippet_data in [DataProcessor.load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)]
        return bug_snippets, non_bug_snippets


class TripletCreator:
    @staticmethod
    def create_triplets(num_negatives_per_positive, instance_id_map, snippets):
        """Generates triplets based on the provided snippets and instance id map."""
        bug_snippets, non_bug_snippets = DataProcessor.separate_snippets(snippets)
        return [{'anchor': instance_id_map[os.path.basename(folder_path)], 'positive': positive_doc, 'negative': random.choice(non_bug_snippets)} 
                for folder_path, _ in snippets 
                for positive_doc in bug_snippets 
                for _ in range(min(num_negatives_per_positive, len(non_bug_snippets)))]


class BugTripletDataset(Dataset):
    def __init__(self, triplets, max_sequence_length):
        """Initializes the BugTripletDataset with the provided triplets and max sequence length."""
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        """Returns the number of triplets in the dataset."""
        return len(self.triplets)

    def __getitem__(self, idx):
        """Retrieves a triplet from the dataset and preprocesses it for training."""
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']

        anchor_sequence = self.tokenizer.encode_plus(anchor, 
                                                      max_length=self.max_sequence_length, 
                                                      padding='max_length', 
                                                      truncation=True, 
                                                      return_tensors='pt',
                                                      return_attention_mask=True)

        positive_sequence = self.tokenizer.encode_plus(positive, 
                                                        max_length=self.max_sequence_length, 
                                                        padding='max_length', 
                                                        truncation=True, 
                                                        return_tensors='pt',
                                                        return_attention_mask=True)

        negative_sequence = self.tokenizer.encode_plus(negative, 
                                                        max_length=self.max_sequence_length, 
                                                        padding='max_length', 
                                                        truncation=True, 
                                                        return_tensors='pt',
                                                        return_attention_mask=True)

        return {'anchor': {'input_ids': anchor_sequence['input_ids'].flatten(), 
                           'attention_mask': anchor_sequence['attention_mask'].flatten()}, 
                'positive': {'input_ids': positive_sequence['input_ids'].flatten(), 
                             'attention_mask': positive_sequence['attention_mask'].flatten()}, 
                'negative': {'input_ids': negative_sequence['input_ids'].flatten(), 
                             'attention_mask': negative_sequence['attention_mask'].flatten()}}

    def shuffle(self):
        """Shuffles the triplets in the dataset."""
        random.shuffle(self.triplets)


class BugTripletModel(nn.Module):
    def __init__(self, embedding_size, fully_connected_size, dropout_rate):
        """Initializes the BugTripletModel with the specified hyperparameters."""
        super(BugTripletModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, fully_connected_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fully_connected_size, embedding_size)

    def forward(self, x):
        """Defines the forward pass of the BugTripletModel."""
        anchor_input_ids = x['anchor']['input_ids']
        anchor_attention_mask = x['anchor']['attention_mask']
        positive_input_ids = x['positive']['input_ids']
        positive_attention_mask = x['positive']['attention_mask']
        negative_input_ids = x['negative']['input_ids']
        negative_attention_mask = x['negative']['attention_mask']

        anchor_output = self.bert(anchor_input_ids, attention_mask=anchor_attention_mask)
        positive_output = self.bert(positive_input_ids, attention_mask=positive_attention_mask)
        negative_output = self.bert(negative_input_ids, attention_mask=negative_attention_mask)

        anchor_embedding = self.fc2(self.relu(self.fc1(self.dropout(anchor_output.pooler_output))))
        positive_embedding = self.fc2(self.relu(self.fc1(self.dropout(positive_output.pooler_output))))
        negative_embedding = self.fc2(self.relu(self.fc1(self.dropout(negative_output.pooler_output))))

        return anchor_embedding, positive_embedding, negative_embedding


def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings, device):
    """Calculates the loss based on the anchor, positive, and negative embeddings."""
    return torch.mean((anchor_embeddings - positive_embeddings) ** 2) + torch.max(torch.mean((anchor_embeddings - negative_embeddings) ** 2) - torch.mean((anchor_embeddings - positive_embeddings) ** 2), torch.tensor(0.0).to(device))


def train(model, device, train_loader, optimizer, epochs):
    """Trains the model on the provided training data."""
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            anchor_embeddings, positive_embeddings, negative_embeddings = model(batch)
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}')
    return total_loss / epochs


def evaluate(model, device, test_loader):
    """Evaluates the model on the provided test data."""
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            anchor_embeddings, positive_embeddings, negative_embeddings = model(batch)
            for i in range(len(anchor_embeddings)):
                similarity_positive = torch.sum(anchor_embeddings[i] * positive_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(positive_embeddings[i]))
                similarity_negative = torch.sum(anchor_embeddings[i] * negative_embeddings[i]) / (torch.norm(anchor_embeddings[i]) * torch.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
    return total_correct / len(test_loader.dataset)


def main():
    """Executes the main training and evaluation loop."""
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    instance_id_map = {item['instance_id']: item['problem_statement'] for item in DataProcessor.load_data(dataset_path)}
    snippets = DataProcessor.load_snippets(snippet_folder_path)
    triplets = TripletCreator.create_triplets(1, instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    train_dataset = BugTripletDataset(train_triplets, 512)
    test_dataset = BugTripletDataset(test_triplets, 512)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BugTripletModel(128, 64, 0.2)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        loss = train(model, device, train_loader, optimizer, 1)
        print(f'Epoch {epoch+1}, Loss: {loss}')
    print(f'Test Accuracy: {evaluate(model, device, test_loader)}')


if __name__ == "__main__":
    main()