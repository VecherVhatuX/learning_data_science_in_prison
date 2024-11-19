import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing, optimizers, losses, metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import DistilBert
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TripletDataset:
    def __init__(self, triplets, max_sequence_length, tokenizer):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor = self.tokenizer.encode(triplet['anchor'], max_length=self.max_sequence_length, padding='max_length', truncation=True)
        positive = self.tokenizer.encode(triplet['positive'], max_length=self.max_sequence_length, padding='max_length', truncation=True)
        negative = self.tokenizer.encode(triplet['negative'], max_length=self.max_sequence_length, padding='max_length', truncation=True)
        return np.array(anchor), np.array(positive), np.array(negative)

    def shuffle_samples(self):
        random.shuffle(self.triplets)


class TripletModel:
    def __init__(self, embedding_size=128, fully_connected_size=64, dropout_rate=0.2, max_sequence_length=512, learning_rate_value=1e-5):
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.learning_rate_value = learning_rate_value
        self.tokenizer = preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(["placeholder"])

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

    def create_model(self):
        input_ids = layers.Input(shape=(self.max_sequence_length,), dtype='int32', name='input_ids')
        attention_masks = layers.Input(shape=(self.max_sequence_length,), dtype='int32', name='attention_masks')
        distilbert = DistilBert('distilbert-base-uncased')(input_ids, attention_masks=attention_masks)
        dropout = layers.Dropout(self.dropout_rate)(distilbert)
        fc1 = layers.Dense(self.fully_connected_size, activation='relu')(dropout)
        fc2 = layers.Dense(self.embedding_size)(fc1)
        model = Model(inputs=[input_ids, attention_masks], outputs=fc2)
        return model

    def train_model(self, model, train_data, test_data, max_training_epochs):
        anchor_input_ids, anchor_attention_masks, positive_input_ids, positive_attention_masks, negative_input_ids, negative_attention_masks = train_data
        test_anchor_input_ids, test_anchor_attention_masks, test_positive_input_ids, test_positive_attention_masks, test_negative_input_ids, test_negative_attention_masks = test_data
        model.compile(loss='mean_squared_error', optimizer=Adam(self.learning_rate_value))
        model.fit([anchor_input_ids, anchor_attention_masks, positive_input_ids, positive_attention_masks, negative_input_ids, negative_attention_masks],
                  np.random.rand(len(anchor_input_ids), self.embedding_size),
                  validation_data=([test_anchor_input_ids, test_anchor_attention_masks, test_positive_input_ids, test_positive_attention_masks, test_negative_input_ids, test_negative_attention_masks],
                                    np.random.rand(len(test_anchor_input_ids), self.embedding_size)),
                  epochs=max_training_epochs, batch_size=32)

    def evaluate_model(self, model, test_data):
        test_anchor_input_ids, test_anchor_attention_masks, test_positive_input_ids, test_positive_attention_masks, test_negative_input_ids, test_negative_attention_masks = test_data
        predictions = model.predict([test_anchor_input_ids, test_anchor_attention_masks, test_positive_input_ids, test_positive_attention_masks, test_negative_input_ids, test_negative_attention_masks])
        print('Test Loss:', np.mean((predictions - np.random.rand(len(test_anchor_input_ids), self.embedding_size))**2))

    def create_embeddings(self, model, data):
        input_ids, attention_masks = data
        embeddings = model.predict([input_ids, attention_masks])
        return embeddings

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def evaluate_triplet_model(self, model, test_data):
        test_anchor_input_ids, test_anchor_attention_masks, test_positive_input_ids, test_positive_attention_masks, test_negative_input_ids, test_negative_attention_masks = test_data
        anchor_embeddings = self.create_embeddings(model, (test_anchor_input_ids, test_anchor_attention_masks))
        positive_embeddings = self.create_embeddings(model, (test_positive_input_ids, test_positive_attention_masks))
        negative_embeddings = self.create_embeddings(model, (test_negative_input_ids, test_negative_attention_masks))
        similarities = []
        for i in range(len(anchor_embeddings)):
            similarity_positive = self.calculate_similarity(anchor_embeddings[i], positive_embeddings[i])
            similarity_negative = self.calculate_similarity(anchor_embeddings[i], negative_embeddings[i])
            similarities.append(similarity_positive > similarity_negative)
        accuracy = np.mean(similarities)
        print('Test Accuracy:', accuracy)

    def pipeline(self, dataset_path, snippet_folder_path, num_negatives_per_positive=1, max_training_epochs=5, batch_size=32):
        triplets = self.create_triplet_dataset(dataset_path, snippet_folder_path)
        train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
        train_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in train_triplets]
        test_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in test_triplets]
        train_dataset = TripletDataset(train_triplets, self.max_sequence_length, self.tokenizer)
        test_dataset = TripletDataset(test_triplets, self.max_sequence_length, self.tokenizer)
        train_anchor_input_ids, train_positive_input_ids, train_negative_input_ids = zip(*[(t[0], t[1], t[2]) for t in train_dataset])
        test_anchor_input_ids, test_positive_input_ids, test_negative_input_ids = zip(*[(t[0], t[1], t[2]) for t in test_dataset])
        train_anchor_attention_masks = np.ones((len(train_anchor_input_ids), self.max_sequence_length))
        train_positive_attention_masks = np.ones((len(train_positive_input_ids), self.max_sequence_length))
        train_negative_attention_masks = np.ones((len(train_negative_input_ids), self.max_sequence_length))
        test_anchor_attention_masks = np.ones((len(test_anchor_input_ids), self.max_sequence_length))
        test_positive_attention_masks = np.ones((len(test_positive_input_ids), self.max_sequence_length))
        test_negative_attention_masks = np.ones((len(test_negative_input_ids), self.max_sequence_length))
        train_data = (np.array(train_anchor_input_ids), train_anchor_attention_masks, np.array(train_positive_input_ids), train_positive_attention_masks, np.array(train_negative_input_ids), train_negative_attention_masks)
        test_data = (np.array(test_anchor_input_ids), test_anchor_attention_masks, np.array(test_positive_input_ids), test_positive_attention_masks, np.array(test_negative_input_ids), test_negative_attention_masks)
        model = self.create_model()
        self.train_model(model, train_data, test_data, max_training_epochs)
        self.evaluate_triplet_model(model, test_data)


if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    TripletModel().pipeline(dataset_path, snippet_folder_path)