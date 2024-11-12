import json
import os
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from pytorch_lightning import LightningModule, Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

nltk.download('punkt')

# Constants
INSTANCE_ID_FIELD = 'instance_id'
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_NEGATIVES_PER_POSITIVE = 3
EMBEDDING_DIM = 128
FC_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 1e-5
MAX_EPOCHS = 5

# File operations
def load_dataset(dataset_path):
    try:
        return np.load(dataset_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return []

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

# Folder operations
def get_subfolder_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

# Data processing
def separate_snippets(snippets):
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
            for positive_doc in positive_snippets
            for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

# Encoding
def encode_triplet(triplet, tokenizer, max_length):
    encoded_triplet = {}
    for key, text in triplet.items():
        encoded_text = tokenizer.encode_plus(
            text=text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        encoded_triplet[f"{key}_input_ids"] = encoded_text['input_ids'].flatten()
        encoded_triplet[f"{key}_attention_mask"] = encoded_text['attention_mask'].flatten()
    return encoded_triplet

# Dataset
class TripletDataset(Dataset):
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        dataset = load_dataset(self.dataset_path)
        return len(dataset) * NUM_NEGATIVES_PER_POSITIVE

    def __getitem__(self, index):
        dataset = load_dataset(self.dataset_path)
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
        folder = os.listdir(self.snippet_folder_path)[index // NUM_NEGATIVES_PER_POSITIVE]
        folder_path = os.path.join(self.snippet_folder_path, folder)
        snippet_file = os.path.join(folder_path, 'snippet.json')
        snippets = load_json_file(snippet_file)
        bug_snippets, non_bug_snippets = separate_snippets(snippets)
        problem_statement = instance_id_map.get(folder)
        triplets = create_triplets(problem_statement, bug_snippets, non_bug_snippets, NUM_NEGATIVES_PER_POSITIVE)
        triplet = triplets[index % NUM_NEGATIVES_PER_POSITIVE]
        return encode_triplet(triplet, self.tokenizer, MAX_LENGTH)

# Model
class TripletModel(LightningModule):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(30522, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(EMBEDDING_DIM, FC_DIM)
        self.relu = nn.ReLU()

    def forward(self, anchor_input_ids, _, positive_input_ids, _, negative_input_ids, _):
        anchor_outputs = self._get_outputs(anchor_input_ids)
        positive_outputs = self._get_outputs(positive_input_ids)
        negative_outputs = self._get_outputs(negative_input_ids)
        return torch.cat([anchor_outputs, positive_outputs, negative_outputs], dim=0)

    def _get_outputs(self, input_ids):
        outputs = self.embedding(input_ids)
        outputs = self.dropout(outputs)
        outputs = torch.mean(outputs, dim=1)
        outputs = self.fc(outputs)
        outputs = self.relu(outputs)
        return outputs

    def training_step(self, batch, batch_idx):
        anchor_input_ids, _, positive_input_ids, _, negative_input_ids, _ = batch
        outputs = self(anchor_input_ids, _, positive_input_ids, _, negative_input_ids, _)
        loss = torch.mean(torch.norm(outputs[:BATCH_SIZE] - outputs[BATCH_SIZE:2*BATCH_SIZE], dim=1) - torch.norm(outputs[:BATCH_SIZE] - outputs[2*BATCH_SIZE:], dim=1) + 1)
        return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)

# Training
def train(dataset_path, snippet_folder_path):
    dataset = TripletDataset(dataset_path, snippet_folder_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model = TripletModel()
    trainer = Trainer(max_epochs=MAX_EPOCHS)
    trainer.fit(model, dataloader)

# Main
def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    train(dataset_path, snippet_folder_path)

if __name__ == "__main__":
    main()