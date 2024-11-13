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

class TripletDataset(Dataset):
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dataset = self._load_dataset(dataset_path)
        self.instance_id_map = {item['instance_id']: item['problem_statement'] for item in self.dataset}
        self.folder_paths = self._get_subfolder_paths(snippet_folder_path)
        self.snippets = [self._load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in self.folder_paths]
        self.bug_snippets, self.non_bug_snippets = zip(*[self._separate_snippets(snippet) for snippet in self.snippets])
        self.problem_statements = [self.instance_id_map.get(os.path.basename(folder_path)) for folder_path in self.folder_paths]
        self.triplets = [self._create_triplets(problem_statement, bug_snippets, non_bug_snippets, NUM_NEGATIVES_PER_POSITIVE) for problem_statement, bug_snippets, non_bug_snippets in zip(self.problem_statements, self.bug_snippets, self.non_bug_snippets)]

    def _load_dataset(self, dataset_path):
        try:
            return np.load(dataset_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"File not found: {dataset_path}")
            return []

    def _load_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def _get_subfolder_paths(self, folder_path):
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def _separate_snippets(self, snippets):
        return (
            [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
            [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
        )

    def _create_triplets(self, problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
                for positive_doc in positive_snippets
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

    def _encode_triplet(self, triplet):
        encoded_triplet = {}
        for key, text in triplet.items():
            encoded_text = self.tokenizer.encode_plus(
                text=text,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            encoded_triplet[f"{key}_input_ids"] = encoded_text['input_ids'].flatten()
            encoded_triplet[f"{key}_attention_mask"] = encoded_text['attention_mask'].flatten()
        return encoded_triplet

    def __len__(self):
        return len(self.dataset) * NUM_NEGATIVES_PER_POSITIVE

    def __getitem__(self, index):
        folder_index = index // NUM_NEGATIVES_PER_POSITIVE
        triplet_index = index % NUM_NEGATIVES_PER_POSITIVE
        return self._encode_triplet(self.triplets[folder_index][triplet_index])

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

    def validation_step(self, batch, batch_idx):
        anchor_input_ids, _, positive_input_ids, _, negative_input_ids, _ = batch
        outputs = self(anchor_input_ids, _, positive_input_ids, _, negative_input_ids, _)
        loss = torch.mean(torch.norm(outputs[:BATCH_SIZE] - outputs[BATCH_SIZE:2*BATCH_SIZE], dim=1) - torch.norm(outputs[:BATCH_SIZE] - outputs[2*BATCH_SIZE:], dim=1) + 1)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        anchor_input_ids, _, positive_input_ids, _, negative_input_ids, _ = batch
        outputs = self(anchor_input_ids, _, positive_input_ids, _, negative_input_ids, _)
        loss = torch.mean(torch.norm(outputs[:BATCH_SIZE] - outputs[BATCH_SIZE:2*BATCH_SIZE], dim=1) - torch.norm(outputs[:BATCH_SIZE] - outputs[2*BATCH_SIZE:], dim=1) + 1)
        return {'test_loss': loss}

def train(model, dataloader):
    trainer = Trainer(max_epochs=MAX_EPOCHS, 
                      log_every_n_steps=10, 
                      flush_logs_every_n_steps=100, 
                      checkpoint_callback=True)
    trainer.fit(model, dataloader)

def evaluate(model, dataloader):
    trainer = Trainer()
    return trainer.test(model, dataloader)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    dataset = TripletDataset(dataset_path, snippet_folder_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = TripletModel()
    train(model, dataloader)

if __name__ == "__main__":
    main()