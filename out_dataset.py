import json
import os
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

nltk.download('punkt')

class TripletDataset(Dataset):
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.instance_id_field = 'instance_id'
        self.num_negatives_per_positive = 3
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.batch_size = 16

    def load_dataset(self):
        try:
            return np.load(self.dataset_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"File not found: {self.dataset_path}")
            return []

    def load_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def get_subfolder_paths(self, folder_path):
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def separate_snippets(self, snippets):
        return (
            [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
            [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
        )

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
                for positive_doc in positive_snippets
                for _ in range(min(self.num_negatives_per_positive, len(negative_snippets)))]

    def encode_triplet(self, triplet):
        encoded_triplet = {}
        for key, text in triplet.items():
            encoded_text = self.tokenizer.encode_plus(
                text=text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            encoded_triplet[f"{key}_input_ids"] = encoded_text['input_ids'].flatten()
            encoded_triplet[f"{key}_attention_mask"] = encoded_text['attention_mask'].flatten()
        return encoded_triplet

    def __len__(self):
        dataset = self.load_dataset()
        return len(dataset) * self.num_negatives_per_positive

    def __getitem__(self, index):
        dataset = self.load_dataset()
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
        folder = os.listdir(self.snippet_folder_path)[index // self.num_negatives_per_positive]
        folder_path = os.path.join(self.snippet_folder_path, folder)
        snippet_file = os.path.join(folder_path, 'snippet.json')
        snippets = self.load_json_file(snippet_file)
        bug_snippets, non_bug_snippets = self.separate_snippets(snippets)
        problem_statement = instance_id_map.get(folder)
        triplets = self.create_triplets(problem_statement, bug_snippets, non_bug_snippets)
        triplet = triplets[index % self.num_negatives_per_positive]
        return self.encode_triplet(triplet)


class TripletModel(LightningModule):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.embedding = nn.Embedding(30522, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 64)
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
        loss = torch.mean(torch.norm(outputs[:64] - outputs[64:128], dim=1) - torch.norm(outputs[:64] - outputs[128:], dim=1) + 1)
        return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5)


def main(dataset_path, snippet_folder_path):
    dataset = TripletDataset(dataset_path, snippet_folder_path)
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size)
    model = TripletModel()
    trainer = torch.trainer.Trainer(max_epochs=5)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    main(dataset_path, snippet_folder_path)