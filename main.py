**dataset.py**
```python
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import random
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, dataset_list, tokenizer, max_length=75, negative_sample_size=3):
        self.dataset_list = dataset_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        anchor = self.dataset_list[idx]['anchor']
        positive = self.dataset_list[idx]['positive']
        negative_samples = random.sample(self.dataset_list[idx]['negative'], self.negative_sample_size)

        anchor_encoding = self.tokenizer(anchor, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        positive_encoding = self.tokenizer(positive, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        negative_encodings = [self.tokenizer(negative, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length') for negative in negative_samples]

        return {
            'anchor_input_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_input_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_input_ids': [negative_encoding['input_ids'].flatten() for negative_encoding in negative_encodings],
            'negative_attention_mask': [negative_encoding['attention_mask'].flatten() for negative_encoding in negative_encodings],
        }

    def get_epoch_dataset(self):
        random.shuffle(self.dataset_list)
        return self.dataset_list

class EpochDataset(Dataset):
    def __init__(self, dataset_list, tokenizer, max_length=75, negative_sample_size=3):
        self.dataset_list = dataset_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        anchor = self.dataset_list[idx]['anchor']
        positive = self.dataset_list[idx]['positive']
        negative_samples = self.dataset_list[idx]['negative']

        anchor_encoding = self.tokenizer(anchor, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        positive_encoding = self.tokenizer(positive, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        negative_encodings = [self.tokenizer(negative, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length') for negative in negative_samples]

        return {
            'anchor_input_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_input_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_input_ids': [negative_encoding['input_ids'].flatten() for negative_encoding in negative_encodings],
            'negative_attention_mask': [negative_encoding['attention_mask'].flatten() for negative_encoding in negative_encodings],
        }

def create_epoch_dataset(dataset, tokenizer, max_length=75, negative_sample_size=3):
    dataset_list = dataset.get_epoch_dataset()
    return EpochDataset(dataset_list, tokenizer, max_length, negative_sample_size)
```

**data_loader.py**
```python
import pickle
import os
from pathlib import Path
import random

def load_data(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def prepare_dataset(data, negative_sample_size):
    dataset_list = []
    for item in data:
        query = item['query']
        relevant_doc = item['relevant_doc']
        non_relevant_docs = item['irrelevant_docs']
        negative_samples = random.sample(non_relevant_docs, min(len(non_relevant_docs), negative_sample_size))

        dataset_list.append({
            "anchor": query,
            "positive": relevant_doc,
            "negative": negative_samples
        })
    return dataset_list
```

**training.py**
```python
from model import create_model
from dataset import CustomDataset, EpochDataset, create_epoch_dataset
from data_loader import load_data, prepare_dataset
from training_utils import train_model, evaluate_model
import torch
import os
import datetime

def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _get_optimizer(model, lr):
    return AdamW(model.parameters(), lr=lr)

def _get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

def _get_data(data_path):
    return load_data(data_path)

def _get_dataset(data, negative_sample_size, tokenizer):
    dataset_list = prepare_dataset(data, negative_sample_size)
    return CustomDataset(dataset_list, tokenizer)

def _get_dataloaders(dataset, batch_size):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, eval_dataloader

def train(model_path, output_dir, train_batch_size, negative_sample_size):
    device = _get_device()
    data_path = os.environ.get('DATA_PATH', '/home/ma-user/data/data.pkl')
    data = _get_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = _get_dataset(data, negative_sample_size, tokenizer)

    model = create_model(model_path)
    model.to(device)

    optimizer = _get_optimizer(model, 1e-5)
    scheduler = _get_scheduler(optimizer, num_warmup_steps=len(dataset) * 20 * 0.1, num_training_steps=len(dataset) * 20)

    for epoch in range(20):
        epoch_dataset = create_epoch_dataset(dataset, tokenizer)
        epoch_dataloader = DataLoader(epoch_dataset, batch_size=train_batch_size, shuffle=True)
        total_loss = train_model(model, device, epoch_dataloader, optimizer, scheduler)
        accuracy = evaluate_model(model, device, epoch_dataloader)
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(epoch_dataloader)}, Accuracy: {accuracy}')
    save_model(model, output_dir)
```

**training_utils.py**
```python
from model import create_model
from dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

def train_model(model, device, train_dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch['anchor_input_ids'].to(device)
        attention_mask = batch['anchor_attention_mask'].to(device)
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        negative_input_ids = torch.cat([nids.unsqueeze(1) for nids in batch['negative_input_ids']], dim=1).to(device)
        negative_attention_mask = torch.cat([nids.unsqueeze(1) for nids in batch['negative_attention_mask']], dim=1).to(device)

        optimizer.zero_grad()

        anchor_outputs = model(input_ids, attention_mask=attention_mask)
        positive_outputs = model(positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = model(negative_input_ids, attention_mask=negative_attention_mask)

        anchor_embeddings = anchor_outputs.last_hidden_state[:, 0, :]
        positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]
        negative_embeddings = negative_outputs.last_hidden_state[:, 0, :]

        similarity = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), positive_embeddings.detach().cpu().numpy())
        similarity_negative = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), negative_embeddings.detach().cpu().numpy().reshape(-1, negative_embeddings.shape[-1]))

        labels = torch.ones(similarity.shape[0])

        loss = nn.MSELoss()(torch.tensor(similarity), labels) + nn.MSELoss()(torch.tensor(1-similarity_negative.max(axis=1)[0]), labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss

def evaluate_model(model, device, eval_dataloader):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch in eval_dataloader:
            input_ids = batch['anchor_input_ids'].to(device)
            attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = torch.cat([nids.unsqueeze(1) for nids in batch['negative_input_ids']], dim=1).to(device)
            negative_attention_mask = torch.cat([nids.unsqueeze(1) for nids in batch['negative_attention_mask']], dim=1).to(device)

            anchor_outputs = model(input_ids, attention_mask=attention_mask)
            positive_outputs = model(positive_input_ids, attention_mask=positive_attention_mask)
            negative_outputs = model(negative_input_ids, attention_mask=negative_attention_mask)

            anchor_embeddings = anchor_outputs.last_hidden_state[:, 0, :]
            positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]
            negative_embeddings = negative_outputs.last_hidden_state[:, 0, :]

            similarity = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), positive_embeddings.detach().cpu().numpy())
            similarity_negative = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), negative_embeddings.detach().cpu().numpy().reshape(-1, negative_embeddings.shape[-1]))

            labels = torch.ones(similarity.shape[0])

            predicted = torch.argmax(torch.cat((torch.tensor(similarity).unsqueeze(1), torch.tensor(similarity_negative.max(axis=1)[0]).unsqueeze(1)), dim=1), dim=1)

            total_correct += (predicted == 0).sum().item()

        accuracy = total_correct / len(eval_dataloader.dataset)
        return accuracy
```
Remember to replace all instances of `dataset.py`, `data_loader.py` and `training_utils.py` with the updated code in the respective files.