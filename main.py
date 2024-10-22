import logging
import sys
import traceback
from datetime import datetime
import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import Repository
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from tqdm import tqdm
from random import sample
from pathlib import Path

# Logging
def setup_logging():
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def log_info(message):
    logging.getLogger().info(message)

# Model
def load_model(model_path):
    return AutoModel.from_pretrained(model_path)

def save_model(model, output_dir):
    model.save_pretrained(output_dir)

def create_model(model_path):
    return nn.Module(AutoModel.from_pretrained(model_path))

# Data
def load_data(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def prepare_dataset(data, negative_sample_size):
    dataset_list = []
    for item in data:
        query = item['query']
        relevant_doc = item['relevant_doc']
        non_relevant_docs = sample(item['irrelevant_docs'], min(len(item['irrelevant_docs']), negative_sample_size))
        for item in non_relevant_docs:
            dataset_list.append({
                "anchor": query,
                "positive": relevant_doc,
                "negative": item
            })
    return dataset_list

# Dataset
class CustomDataset(Dataset):
    def __init__(self, dataset_list, tokenizer):
        self.dataset_list = dataset_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        anchor = self.dataset_list[idx]['anchor']
        positive = self.dataset_list[idx]['positive']
        negative = self.dataset_list[idx]['negative']
        anchor_encoding = self.tokenizer(anchor, return_tensors='pt', max_length=75, truncation=True, padding='max_length')
        positive_encoding = self.tokenizer(positive, return_tensors='pt', max_length=75, truncation=True, padding='max_length')
        negative_encoding = self.tokenizer(negative, return_tensors='pt', max_length=75, truncation=True, padding='max_length')
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_input_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_input_ids': negative_encoding['input_ids'].flatten(),
            'negative_attention_mask': negative_encoding['attention_mask'].flatten(),
        }

# Training
def train_model(model, device, train_dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch['anchor_input_ids'].to(device)
        attention_mask = batch['anchor_attention_mask'].to(device)
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        negative_input_ids = batch['negative_input_ids'].to(device)
        negative_attention_mask = batch['negative_attention_mask'].to(device)
        optimizer.zero_grad()
        anchor_outputs = model(input_ids, attention_mask=attention_mask)
        positive_outputs = model(positive_input_ids, attention_mask=positive_attention_mask)
        negative_outputs = model(negative_input_ids, attention_mask=negative_attention_mask)
        anchor_embeddings = anchor_outputs.last_hidden_state[:, 0, :]
        positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]
        negative_embeddings = negative_outputs.last_hidden_state[:, 0, :]
        similarity = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), positive_embeddings.detach().cpu().numpy())
        similarity_negative = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), negative_embeddings.detach().cpu().numpy())
        labels = torch.ones(similarity.shape[0])
        loss = nn.MSELoss()(torch.tensor(similarity), labels) + nn.MSELoss()(torch.tensor(similarity_negative), 1-labels)
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
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            anchor_outputs = model(input_ids, attention_mask=attention_mask)
            positive_outputs = model(positive_input_ids, attention_mask=positive_attention_mask)
            negative_outputs = model(negative_input_ids, attention_mask=negative_attention_mask)
            anchor_embeddings = anchor_outputs.last_hidden_state[:, 0, :]
            positive_embeddings = positive_outputs.last_hidden_state[:, 0, :]
            negative_embeddings = negative_outputs.last_hidden_state[:, 0, :]
            similarity = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), positive_embeddings.detach().cpu().numpy())
            similarity_negative = cosine_similarity(anchor_embeddings.detach().cpu().numpy(), negative_embeddings.detach().cpu().numpy())
            labels = torch.ones(similarity.shape[0])
            predicted = torch.argmax(torch.cat((torch.tensor(similarity).unsqueeze(1), torch.tensor(similarity_negative).unsqueeze(1)), dim=1), dim=1)
            total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(eval_dataloader.dataset)
        return accuracy

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
    log_info(f"There are {torch.cuda.device_count()} GPUs available for torch.")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        log_info(f"GPU {i}: {name}")

    data_path = os.environ.get('DATA_PATH', '/home/ma-user/data/data.pkl')
    data = _get_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = _get_dataset(data, negative_sample_size, tokenizer)
    train_dataloader, eval_dataloader = _get_dataloaders(dataset, train_batch_size)

    model = create_model(model_path)
    model.to(device)

    optimizer = _get_optimizer(model, 1e-5)
    scheduler = _get_scheduler(optimizer, num_warmup_steps=len(train_dataloader) * 20 * 0.1, num_training_steps=len(train_dataloader) * 20)

    training_args = {
        'output_dir': output_dir,
        'num_train_epochs': 20,
        'per_device_train_batch_size': train_batch_size,
        'per_device_eval_batch_size': train_batch_size,
        'warmup_ratio': 0.1,
        'fp16': True,
        'bf16': False,
        'batch_sampler': 'no_duplicates',
        'eval_strategy': 'steps',
        'eval_steps': 1000,
        'save_strategy': 'steps',
        'save_steps': 1000,
        'save_total_limit': 2,
        'logging_steps': 100,
        'run_name': "nli-v2",
    }

    for epoch in range(training_args['num_train_epochs']):
        total_loss = train_model(model, device, train_dataloader, optimizer, scheduler)
        log_info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')
        accuracy = evaluate_model(model, device, eval_dataloader)
        log_info(f'Epoch {epoch+1}, Accuracy: {accuracy}')
    save_model(model, output_dir)

if __name__ == "__main__":
    setup_logging()
    model_path = os.environ.get('MODEL_PATH', '/home/ma-user/bert-base-uncased')
    model_path = "bert-base-uncased" if not model_path else model_path
    output_path = os.environ.get('OUTPUT_PATH', '/tmp/output')
    model_name = Path(model_path).stem
    output_dir = output_path + "/output/training_nli_v2_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_batch_size = 64
    negative_sample_size = 3
    train(model_path, output_dir, train_batch_size, negative_sample_size)