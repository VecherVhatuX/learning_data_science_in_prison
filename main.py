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

class SentenceTransformerTrainer:
    def __init__(self, model_path, output_dir, train_batch_size, negative_sample_size=3):
        self.model_path = model_path
        self.output_dir = output_dir
        self.train_batch_size = train_batch_size
        self.negative_sample_size = negative_sample_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _setup_logging(self):
        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    def _load_model(self):
        return AutoModel.from_pretrained(self.model_path)

    def _load_data(self, data_path):
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    def _prepare_dataset(self, data):
        dataset_list = []
        for item in data:
            query = item['query']
            relevant_doc = item['relevant_doc']
            non_relevant_docs = sample(item['irrelevant_docs'], min(len(item['irrelevant_docs']), self.negative_sample_size))
            for item in non_relevant_docs:
                dataset_list.append({
                    "anchor": query,
                    "positive": relevant_doc,
                    "negative": item
                })
        return dataset_list

    def _create_dataset(self, data):
        dataset_list = self._prepare_dataset(data)
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
        return CustomDataset(dataset_list, AutoTokenizer.from_pretrained('bert-base-uncased'))

    def _get_training_args(self):
        return {
            'output_dir': self.output_dir,
            'num_train_epochs': 20,
            'per_device_train_batch_size': self.train_batch_size,
            'per_device_eval_batch_size': self.train_batch_size,
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

    def train(self):
        self._setup_logging()
        print(f"There are {torch.cuda.device_count()} GPUs available for torch.")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {name}")

        model = self._load_model()
        model.to(self.device)
        data_path = os.environ.get('DATA_PATH', '/home/ma-user/data/data.pkl')
        data = self._load_data(data_path)
        train_dataset = self._create_dataset(data)
        eval_dataset = self._create_dataset(data)

        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.train_batch_size, shuffle=False)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader) * 20 * 0.1, num_training_steps=len(train_dataloader) * 20)

        args = self._get_training_args()
        for epoch in range(args['num_train_epochs']):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch['anchor_input_ids'].to(self.device)
                attention_mask = batch['anchor_attention_mask'].to(self.device)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                negative_input_ids = batch['negative_input_ids'].to(self.device)
                negative_attention_mask = batch['negative_attention_mask'].to(self.device)
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
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')
            model.eval()
            with torch.no_grad():
                total_correct = 0
                for batch in eval_dataloader:
                    input_ids = batch['anchor_input_ids'].to(self.device)
                    attention_mask = batch['anchor_attention_mask'].to(self.device)
                    positive_input_ids = batch['positive_input_ids'].to(self.device)
                    positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                    negative_input_ids = batch['negative_input_ids'].to(self.device)
                    negative_attention_mask = batch['negative_attention_mask'].to(self.device)
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
                print(f'Epoch {epoch+1}, Accuracy: {accuracy}')
        model.save_pretrained(self.output_dir)

if __name__ == "__main__":
    model_path = os.environ.get('MODEL_PATH', '/home/ma-user/bert-base-uncased')
    model_path = "bert-base-uncased" if not model_path else model_path
    output_path = os.environ.get('OUTPUT_PATH', '/tmp/output')
    model_name = Path(model_path).stem
    output_dir = output_path + "/output/training_nli_v2_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_batch_size = 64
    SentenceTransformerTrainer(model_path, output_dir, train_batch_size).train()