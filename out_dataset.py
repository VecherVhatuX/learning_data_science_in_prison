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

class TripletModel:
    def __init__(self, embedding_size=128, fully_connected_size=64, dropout_rate=0.2, max_sequence_length=512, learning_rate_value=1e-5):
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.learning_rate_value = learning_rate_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def load_json_data(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def load_dataset(self, file_path):
        return np.load(file_path, allow_pickle=True)

    def load_snippets(self, folder_path):
        return [(os.path.join(folder_path, f), os.path.join(folder_path, f, 'snippet.json')) 
                for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def separate_code_snippets(self, snippets):
        return tuple(map(list, zip(*[
            ((snippet_data['snippet'], True) if snippet_data.get('is_bug', False) else (snippet_data['snippet'], False)) 
            for folder_path, snippet_file_path in snippets 
            for snippet_data in [self.load_json_data(snippet_file_path)]
            if snippet_data.get('snippet')
        ])))

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)} 
                for positive_doc in positive_snippets 
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

    def create_triplet_dataset(self, dataset_path, snippet_folder_path):
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

    def shuffle_samples(self, samples):
        random.shuffle(samples)
        return samples

    def encode_triplet(self, triplet):
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

    def create_dataset(self, triplets):
        anchor_docs = []
        positive_docs = []
        negative_docs = []
        for triplet in triplets:
            anchor, positive, negative = self.encode_triplet(triplet)
            anchor_docs.append(anchor)
            positive_docs.append(positive)
            negative_docs.append(negative)
        return torch.stack(anchor_docs), torch.stack(positive_docs), torch.stack(negative_docs)

    def create_model(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
                self.dropout = nn.Dropout(self.dropout_rate)
                self.fc1 = nn.Linear(self.distilbert.config.hidden_size, self.fully_connected_size)
                self.fc2 = nn.Linear(self.fully_connected_size, self.embedding_size)

            def forward(self, input_ids):
                outputs = self.distilbert(input_ids)
                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)
                pooled_output = F.relu(self.fc1(pooled_output))
                pooled_output = self.dropout(pooled_output)
                pooled_output = self.fc2(pooled_output)
                return pooled_output
        return Model()

    def train_model(self, model, anchor_docs, positive_docs, negative_docs, max_training_epochs):
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate_value)
        loss_fn = nn.TripletLoss()
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

    def plot_results(self, history):
        plt.plot(history['loss'], label='Training Loss')
        plt.legend()
        plt.show()

    def pipeline(self, dataset_path, snippet_folder_path, num_negatives_per_positive=1, max_training_epochs=5):
        triplets = self.create_triplet_dataset(dataset_path, snippet_folder_path)
        train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
        anchor_docs, positive_docs, negative_docs = self.create_dataset(train_triplets)
        model = self.create_model()
        model = self.train_model(model, anchor_docs, positive_docs, negative_docs, max_training_epochs)


if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    TripletModel().pipeline(dataset_path, snippet_folder_path)