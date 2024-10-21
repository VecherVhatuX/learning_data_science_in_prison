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

class ModelTrainer:
    """A class to train a sentence transformer model."""

    def __init__(self):
        """Initialize the model trainer."""
        # Initialize the device info
        self.device_info()

    def _disable_ssl_warnings(self):
        """Disable SSL warnings for requests."""
        # Disable SSL warnings
        import requests
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
        original_request = requests.Session.request
        def patched_request(self, *args, **kwargs):
            # Set verify to False
            kwargs['verify'] = False
            return original_request(self, *args, **kwargs)
        requests.Session.request = patched_request

    def _setup_logging(self):
        """Set up logging for the model trainer."""
        # Set up logging
        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    def _device_info(self):
        """Print information about the available devices."""
        # Print device info
        n = torch.cuda.device_count()
        print(f"There are {n} GPUs available for torch.")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {name}")

    def _load_model(self, model_path):
        """Load a sentence transformer model.

        Args:
            model_path (str): The path to the model.

        Returns:
            AutoModel: The loaded model.
        """
        # Load the model
        return AutoModel.from_pretrained(model_path)

    def load_pretrained_model(self, model_path):
        """Load a pretrained sentence transformer model.

        Args:
            model_path (str): The path to the model.

        Returns:
            AutoModel: The loaded model.
        """
        # Load the pretrained model
        model = self._load_model(model_path)
        return model

    def _load_data(self, data_path):
        """Load data from a file.

        Args:
            data_path (str): The path to the data file.

        Returns:
            Any: The loaded data.
        """
        # Load the data
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    def _prepare_dataset(self, data, negative_sample_size=3):
        """Prepare a dataset for training.

        Args:
            data (list): The data to prepare.
            negative_sample_size (int): The number of negative samples to use. Defaults to 3.

        Returns:
            list: The prepared dataset.
        """
        # Prepare the dataset
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

    def _create_dataset(self, data):
        """Create a dataset from a list of data.

        Args:
            data (list): The data to create a dataset from.

        Returns:
            Dataset: The created dataset.
        """
        # Create the dataset
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

    def _load_datasets(self):
        """Load the training and evaluation datasets.

        Returns:
            tuple: The training and evaluation datasets.
        """
        # Load the datasets
        train_dataset = self._create_dataset(self._load_data('data.pkl'))
        eval_dataset = self._create_dataset(self._load_data('data.pkl'))
        return train_dataset, eval_dataset

    def _get_training_args(self, output_dir, train_batch_size):
        """Get the training arguments.

        Args:
            output_dir (str): The output directory.
            train_batch_size (int): The batch size for training.

        Returns:
            dict: The training arguments.
        """
        # Get the training arguments
        return {
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

    def train_model(self, model, train_dataset, eval_dataset, args):
        """Train a sentence transformer model.

        Args:
            model (AutoModel): The model to train.
            train_dataset (Dataset): The training dataset.
            eval_dataset (Dataset): The evaluation dataset.
            args (dict): The training arguments.
        """
        # Train the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        train_dataloader = DataLoader(train_dataset, batch_size=args['per_device_train_batch_size'], shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args['per_device_eval_batch_size'], shuffle=False)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader) * args['num_train_epochs'] * args['warmup_ratio'], num_training_steps=len(train_dataloader) * args['num_train_epochs'])
        for epoch in range(args['num_train_epochs']):
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
                loss = nn.MSELoss()(similarity, labels) + nn.MSELoss()(similarity_negative, 1-labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')
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
                    predicted = torch.argmax(torch.cat((similarity.unsqueeze(1), similarity_negative.unsqueeze(1)), dim=1), dim=1)
                    total_correct += (predicted == labels).sum().item()
                accuracy = total_correct / len(eval_dataloader.dataset)
                print(f'Epoch {epoch+1}, Accuracy: {accuracy}')

    def _save_model(self, model, output_path):
        """Save a sentence transformer model.

        Args:
            model (AutoModel): The model to save.
            output_path (str): The output path.
        """
        # Save the model
        model.save_pretrained(output_path)

    def run(self):
        """Run the model trainer."""
        # Run the model trainer
        self._disable_ssl_warnings()
        self._setup_logging()
        self._device_info()

        data_path = os.environ.get('DATA_PATH', '/home/ma-user/data/data.pkl')
        output_path = os.environ.get('OUTPUT_PATH', '/tmp/output')
        model_path = os.environ.get('MODEL_PATH', '/home/ma-user/bert-base-uncased')
        model_path = "bert-base-uncased" if not model_path else model_path

        train_batch_size = 64
        max_seq_length = 75
        num_epochs = 1

        model = self.load_pretrained_model(model_path)
        train_dataset, eval_dataset = self._load_datasets()

        model_name = Path(model_path).stem
        output_dir = output_path + "/output/training_nli_v2_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        args = self._get_training_args(output_dir, train_batch_size)

        self.train_model(model, train_dataset, eval_dataset, args)
        self._save_model(model, output_dir)

if __name__ == "__main__":
    ModelTrainer().run()