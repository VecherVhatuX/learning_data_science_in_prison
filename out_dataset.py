import os
import random
import json
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn import CosineSimilarity

class TripletDataset(Dataset):
    """
    Dataset class for creating triplets of anchor, positive and negative samples.
    """
    def __init__(self, triplets, max_sequence_length, tokenizer):
        """
        Initialize the dataset with triplets, max sequence length and tokenizer.

        Args:
            triplets (list): List of dictionaries containing anchor, positive and negative samples.
            max_sequence_length (int): Maximum sequence length for the input samples.
            tokenizer (object): Tokenizer object for encoding the input samples.
        """
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the number of triplets in the dataset.

        Returns:
            int: Number of triplets.
        """
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Returns the encoded input sample and attention mask for the given index.

        Args:
            idx (int): Index of the triplet.

        Returns:
            dict: Dictionary containing encoded input sample and attention mask for anchor, positive and negative samples.
        """
        triplet = self.triplets[idx]
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
        return {
            'anchor': {'input_ids': anchor['input_ids'], 'attention_mask': anchor['attention_mask']},
            'positive': {'input_ids': positive['input_ids'], 'attention_mask': positive['attention_mask']},
            'negative': {'input_ids': negative['input_ids'], 'attention_mask': negative['attention_mask']}
        }

class TripletModel(LightningModule):
    """
    Model class for training a triplet model using PyTorch Lightning.
    """
    def __init__(self, embedding_size=128, fully_connected_size=64, dropout_rate=0.2, max_sequence_length=512, learning_rate_value=1e-5):
        """
        Initialize the model with hyperparameters.

        Args:
            embedding_size (int, optional): Size of the embedding layer. Defaults to 128.
            fully_connected_size (int, optional): Size of the fully connected layer. Defaults to 64.
            dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.2.
            max_sequence_length (int, optional): Maximum sequence length for the input samples. Defaults to 512.
            learning_rate_value (float, optional): Learning rate value for the optimizer. Defaults to 1e-5.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.learning_rate_value = learning_rate_value
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.distilbert.config.hidden_size, self.fully_connected_size)
        self.fc2 = nn.Linear(self.fully_connected_size, self.embedding_size)
        self.cosine_similarity = CosineSimilarity()

    def load_json_data(self, file_path):
        """
        Loads data from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            list: List of data loaded from the JSON file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def load_dataset(self, file_path):
        """
        Loads a dataset from a file.

        Args:
            file_path (str): Path to the dataset file.

        Returns:
            np.ndarray: Dataset loaded from the file.
        """
        return np.load(file_path, allow_pickle=True)

    def load_snippets(self, folder_path):
        """
        Loads snippets from a folder.

        Args:
            folder_path (str): Path to the folder containing snippets.

        Returns:
            list: List of snippets loaded from the folder.
        """
        return [(os.path.join(folder_path, f), os.path.join(folder_path, f, 'snippet.json')) 
                for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def separate_code_snippets(self, snippets):
        """
        Separates code snippets into positive and negative snippets.

        Args:
            snippets (list): List of snippets.

        Returns:
            tuple: Tuple containing lists of positive and negative snippets.
        """
        return tuple(map(list, zip(*[
            ((snippet_data['snippet'], True) if snippet_data.get('is_bug', False) else (snippet_data['snippet'], False)) 
            for folder_path, snippet_file_path in snippets 
            for snippet_data in [self.load_json_data(snippet_file_path)]
            if snippet_data.get('snippet')
        ])))

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        """
        Creates triplets from problem statements, positive snippets and negative snippets.

        Args:
            problem_statement (str): Problem statement.
            positive_snippets (list): List of positive snippets.
            negative_snippets (list): List of negative snippets.
            num_negatives_per_positive (int): Number of negative snippets per positive snippet.

        Returns:
            list: List of triplets.
        """
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)} 
                for positive_doc in positive_snippets 
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

    def create_triplet_dataset(self, dataset_path, snippet_folder_path):
        """
        Creates a triplet dataset from a dataset and a folder of snippets.

        Args:
            dataset_path (str): Path to the dataset.
            snippet_folder_path (str): Path to the folder of snippets.

        Returns:
            list: List of triplets.
        """
        dataset = self.load_dataset(dataset_path)
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
        snippets = self.load_snippets(snippet_folder_path)
        return [
            (problem_statement, bug_snippet, non_bug_snippet) 
            for folder_path, _ in snippets 
            for bug_snippets, non_bug_snippets in [self.separate_code_snippets([(folder_path, os.path.join(folder_path, 'snippet.json'))])]
            for problem_statement in [instance_id_map.get(os.path.basename(folder_path))] 
            for bug_snippet, non_bug_snippet in [(bug, non_bug) for bug in bug_snippets for non_bug in non_bug_snippets]
        ]

    def create_model(self):
        """
        Creates a DistilBERT model.

        Returns:
            object: DistilBERT model.
        """
        return self.distilbert

    def forward(self, input_ids, attention_masks):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_masks (torch.Tensor): Attention masks.

        Returns:
            torch.Tensor: Output of the model.
        """
        outputs = self.distilbert(input_ids, attention_mask=attention_masks)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        fc1_output = torch.relu(self.fc1(dropout_output))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss of the model.
        """
        loss = self.calculate_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss of the model.
        """
        loss = self.calculate_loss(batch)
        self.log('val_loss', loss)
        return loss

    def calculate_loss(self, batch):
        """
        Calculates the loss of the model.

        Args:
            batch (dict): Batch of data.

        Returns:
            torch.Tensor: Loss of the model.
        """
        anchor_input_ids = batch['anchor']['input_ids'].squeeze(1)
        anchor_attention_masks = batch['anchor']['attention_mask'].squeeze(1)
        positive_input_ids = batch['positive']['input_ids'].squeeze(1)
        positive_attention_masks = batch['positive']['attention_mask'].squeeze(1)
        negative_input_ids = batch['negative']['input_ids'].squeeze(1)
        negative_attention_masks = batch['negative']['attention_mask'].squeeze(1)
        anchor_embeddings = self.forward(anchor_input_ids, anchor_attention_masks)
        positive_embeddings = self.forward(positive_input_ids, positive_attention_masks)
        negative_embeddings = self.forward(negative_input_ids, negative_attention_masks)
        positive_distance = torch.mean((anchor_embeddings - positive_embeddings) ** 2)
        negative_distance = torch.mean((anchor_embeddings - negative_embeddings) ** 2)
        return positive_distance + torch.clamp(negative_distance - positive_distance, min=0)

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate_value)

    def evaluate_model(self, test_data):
        """
        Evaluates the model on test data.

        Args:
            test_data (tuple): Test data.
        """
        test_anchor_input_ids, test_anchor_attention_masks, test_positive_input_ids, test_positive_attention_masks, test_negative_input_ids, test_negative_attention_masks = test_data
        anchor_embeddings = self.forward(test_anchor_input_ids, test_anchor_attention_masks)
        positive_embeddings = self.forward(test_positive_input_ids, test_positive_attention_masks)
        negative_embeddings = self.forward(test_negative_input_ids, test_negative_attention_masks)
        similarities = []
        for i in range(len(anchor_embeddings)):
            similarity_positive = self.cosine_similarity(anchor_embeddings[i], positive_embeddings[i])
            similarity_negative = self.cosine_similarity(anchor_embeddings[i], negative_embeddings[i])
            similarities.append(similarity_positive > similarity_negative)
        accuracy = torch.mean(torch.tensor(similarities, dtype=torch.float))
        print('Test Accuracy:', accuracy)

    def prepare_test_data(self, test_triplets):
        """
        Prepares test data for evaluation.

        Args:
            test_triplets (list): List of test triplets.

        Returns:
            tuple: Test data.
        """
        test_anchor_input_ids = []
        test_anchor_attention_masks = []
        test_positive_input_ids = []
        test_positive_attention_masks = []
        test_negative_input_ids = []
        test_negative_attention_masks = []
        for triplet in test_triplets:
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
            test_anchor_input_ids.append(anchor['input_ids'])
            test_anchor_attention_masks.append(anchor['attention_mask'])
            test_positive_input_ids.append(positive['input_ids'])
            test_positive_attention_masks.append(positive['attention_mask'])
            test_negative_input_ids.append(negative['input_ids'])
            test_negative_attention_masks.append(negative['attention_mask'])
        test_anchor_input_ids = torch.cat(test_anchor_input_ids, dim=0)
        test_anchor_attention_masks = torch.cat(test_anchor_attention_masks, dim=0)
        test_positive_input_ids = torch.cat(test_positive_input_ids, dim=0)
        test_positive_attention_masks = torch.cat(test_positive_attention_masks, dim=0)
        test_negative_input_ids = torch.cat(test_negative_input_ids, dim=0)
        test_negative_attention_masks = torch.cat(test_negative_attention_masks, dim=0)
        return test_anchor_input_ids, test_anchor_attention_masks, test_positive_input_ids, test_positive_attention_masks, test_negative_input_ids, test_negative_attention_masks

    def pipeline(self, dataset_path, snippet_folder_path, num_negatives_per_positive=1, max_training_epochs=5, batch_size=32):
        """
        Pipeline for training and evaluating the model.

        Args:
            dataset_path (str): Path to the dataset.
            snippet_folder_path (str): Path to the folder of snippets.
            num_negatives_per_positive (int, optional): Number of negative snippets per positive snippet. Defaults to 1.
            max_training_epochs (int, optional): Maximum number of training epochs. Defaults to 5.
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        triplets = self.create_triplet_dataset(dataset_path, snippet_folder_path)
        train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
        train_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in train_triplets]
        test_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in test_triplets]
        test_data = self.prepare_test_data(test_triplets)
        train_dataset = TripletDataset(train_triplets, self.max_sequence_length, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = self
        trainer = Trainer(max_epochs=max_training_epochs, callbacks=[ModelCheckpoint(save_top_k=1, monitor='val_loss'), LearningRateMonitor()])
        trainer.fit(model, train_loader)
        self.evaluate_model(test_data)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    model = TripletModel()
    model.pipeline(dataset_path, snippet_folder_path)