import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List

class TripletModel:
    def __init__(self, embedding_size: int = 128, fully_connected_size: int = 64, dropout_rate: float = 0.2, max_sequence_length: int = 512, learning_rate_value: float = 1e-5):
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.learning_rate_value = learning_rate_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def load_json_data(self, file_path: str) -> List:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def load_dataset(self, file_path: str) -> np.ndarray:
        return np.load(file_path, allow_pickle=True)

    def load_snippets(self, folder_path: str) -> List:
        return [(os.path.join(folder_path, f), os.path.join(folder_path, f, 'snippet.json')) 
                for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def separate_code_snippets(self, snippets: List) -> Tuple[List, List]:
        return tuple(map(list, zip(*[
            ((snippet_data['snippet'], True) if snippet_data.get('is_bug', False) else (snippet_data['snippet'], False)) 
            for folder_path, snippet_file_path in snippets 
            for snippet_data in [self.load_json_data(snippet_file_path)]
            if snippet_data.get('snippet')
        ])))

    def create_triplets(self, problem_statement: str, positive_snippets: List[str], negative_snippets: List[str], num_negatives_per_positive: int) -> List:
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)} 
                for positive_doc in positive_snippets 
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

    def create_triplet_dataset(self, dataset_path: str, snippet_folder_path: str) -> List:
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

    def shuffle_samples(self, samples: List) -> List:
        random.shuffle(samples)
        return samples

    def encode_triplet(self, triplet: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = self.tokenizer.encode_plus(triplet['anchor'], 
                                             max_length=self.max_sequence_length, 
                                             padding='max_length', 
                                             truncation=True, 
                                             return_attention_mask=True, 
                                             return_tensors='pt')['input_ids'].to(self.device)
        positive = self.tokenizer.encode_plus(triplet['positive'], 
                                               max_length=self.max_sequence_length, 
                                               padding='max_length', 
                                               truncation=True, 
                                               return_attention_mask=True, 
                                               return_tensors='pt')['input_ids'].to(self.device)
        negative = self.tokenizer.encode_plus(triplet['negative'], 
                                               max_length=self.max_sequence_length, 
                                               padding='max_length', 
                                               truncation=True, 
                                               return_attention_mask=True, 
                                               return_tensors='pt')['input_ids'].to(self.device)
        return anchor, positive, negative

    def create_dataset(self, triplets: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_docs = []
        positive_docs = []
        negative_docs = []
        for triplet in triplets:
            anchor, positive, negative = self.encode_triplet(triplet)
            anchor_docs.append(anchor)
            positive_docs.append(positive)
            negative_docs.append(negative)
        return torch.stack(anchor_docs), torch.stack(positive_docs), torch.stack(negative_docs)

    def create_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
                self.dropout = nn.Dropout(self.dropout_rate)
                self.fc1 = nn.Linear(self.distilbert.config.hidden_size, self.fully_connected_size)
                self.fc2 = nn.Linear(self.fully_connected_size, self.embedding_size)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                outputs = self.distilbert(input_ids)
                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)
                pooled_output = F.relu(self.fc1(pooled_output))
                pooled_output = self.dropout(pooled_output)
                pooled_output = self.fc2(pooled_output)
                return pooled_output
        return Model()

    def train_model(self, model: nn.Module, anchor_docs: torch.Tensor, positive_docs: torch.Tensor, negative_docs: torch.Tensor, max_training_epochs: int) -> nn.Module:
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate_value)
        loss_fn = nn.TripletMarginLoss()
        for epoch in range(max_training_epochs):
            model.train()
            optimizer.zero_grad()
            anchor_embeddings = model(anchor_docs)
            positive_embeddings = model(positive_docs)
            negative_embeddings = model(negative_docs)
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        return model

    def plot_results(self, history: dict) -> None:
        plt.plot(history['loss'], label='Training Loss')
        plt.legend()
        plt.show()

    def pipeline(self, dataset_path: str, snippet_folder_path: str, num_negatives_per_positive: int = 1, max_training_epochs: int = 5) -> None:
        triplets = self.create_triplet_dataset(dataset_path, snippet_folder_path)
        train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
        train_triplets = self.shuffle_samples(train_triplets)
        test_triplets = self.shuffle_samples(test_triplets)
        train_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in train_triplets]
        test_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in test_triplets]
        anchor_docs, positive_docs, negative_docs = self.create_dataset(train_triplets)
        model = self.create_model()
        model = self.train_model(model, anchor_docs, positive_docs, negative_docs, max_training_epochs)
        
        # Test model
        test_anchor_docs, test_positive_docs, test_negative_docs = self.create_dataset(test_triplets)
        model.eval()
        with torch.no_grad():
            test_anchor_embeddings = model(test_anchor_docs)
            test_positive_embeddings = model(test_positive_docs)
            test_negative_embeddings = model(test_negative_docs)
            test_loss = nn.TripletMarginLoss()(test_anchor_embeddings, test_positive_embeddings, test_negative_embeddings)
            print(f'Test Loss: {test_loss.item()}')
            
            # Evaluate model
            correct = 0
            with torch.no_grad():
                for i in range(len(test_anchor_embeddings)):
                    anchor = test_anchor_embeddings[i]
                    positive = test_positive_embeddings[i]
                    negative = test_negative_embeddings[i]
                    pos_dist = torch.pairwise_distance(anchor, positive)
                    neg_dist = torch.pairwise_distance(anchor, negative)
                    if pos_dist < neg_dist:
                        correct += 1
            print(f'Test Accuracy: {correct / len(test_anchor_embeddings)}')


if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    TripletModel().pipeline(dataset_path, snippet_folder_path)