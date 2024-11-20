import tensorflow as tf
import numpy as np
import random
import json
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CosineSimilarity
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
        anchor = self.tokenizer.encode_plus(
            triplet['anchor'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        positive = self.tokenizer.encode_plus(
            triplet['positive'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        negative = self.tokenizer.encode_plus(
            triplet['negative'],
            max_length=self.max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        return {
            'anchor': {'input_ids': anchor['input_ids'], 'attention_mask': anchor['attention_mask']},
            'positive': {'input_ids': positive['input_ids'], 'attention_mask': positive['attention_mask']},
            'negative': {'input_ids': negative['input_ids'], 'attention_mask': negative['attention_mask']}
        }

    def shuffle_triplets(self):
        random.shuffle(self.triplets)
        return self

class TripletModel:
    def __init__(self, embedding_size=128, fully_connected_size=64, dropout_rate=0.2, max_sequence_length=512, learning_rate_value=1e-5):
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.learning_rate_value = learning_rate_value
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.model = self.create_model()
        self.cosine_similarity = CosineSimilarity()

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
        input_ids = Input(shape=(self.max_sequence_length,), name='input_ids')
        attention_masks = Input(shape=(self.max_sequence_length,), name='attention_masks')
        embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=self.embedding_size, input_length=self.max_sequence_length)(input_ids)
        pooling = tf.keras.layers.GlobalAveragePooling1D()(embedding)
        dropout = Dropout(self.dropout_rate)(pooling)
        fc1 = Dense(self.fully_connected_size, activation='relu')(dropout)
        fc2 = Dense(self.embedding_size)(fc1)
        model = Model(inputs=[input_ids, attention_masks], outputs=fc2)
        return model

    def calculate_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = tf.reduce_mean(tf.square(anchor_embeddings - positive_embeddings))
        negative_distance = tf.reduce_mean(tf.square(anchor_embeddings - negative_embeddings))
        return positive_distance + tf.maximum(negative_distance - positive_distance, 0)

    def compile_model(self):
        self.model.compile(loss=lambda y_true, y_pred: 0, optimizer=Adam(self.learning_rate_value), metrics=['accuracy'])

    def train_model(self, dataset, batch_size=32, epochs=5):
        self.compile_model()
        for epoch in range(epochs):
            dataset.shuffle_triplets()
            train_triplets = dataset.triplets
            anchor_input_ids = np.array([item['anchor']['input_ids'] for item in train_triplets])
            anchor_attention_masks = np.array([item['anchor']['attention_mask'] for item in train_triplets])
            positive_input_ids = np.array([item['positive']['input_ids'] for item in train_triplets])
            positive_attention_masks = np.array([item['positive']['attention_mask'] for item in train_triplets])
            negative_input_ids = np.array([item['negative']['input_ids'] for item in train_triplets])
            negative_attention_masks = np.array([item['negative']['attention_mask'] for item in train_triplets])
            anchor_embeddings = self.model.predict([anchor_input_ids, anchor_attention_masks])
            positive_embeddings = self.model.predict([positive_input_ids, positive_attention_masks])
            negative_embeddings = self.model.predict([negative_input_ids, negative_attention_masks])
            loss = self.calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            print(f'Epoch {epoch+1}, Loss: {loss}')
            self.model.fit([anchor_input_ids, anchor_attention_masks], positive_embeddings, epochs=1, batch_size=batch_size, verbose=0)
            self.model.fit([positive_input_ids, positive_attention_masks], anchor_embeddings, epochs=1, batch_size=batch_size, verbose=0)
            self.model.fit([negative_input_ids, negative_attention_masks], anchor_embeddings, epochs=1, batch_size=batch_size, verbose=0)

    def evaluate_model(self, dataset):
        train_triplets = dataset.triplets
        anchor_input_ids = np.array([item['anchor']['input_ids'] for item in train_triplets])
        anchor_attention_masks = np.array([item['anchor']['attention_mask'] for item in train_triplets])
        positive_input_ids = np.array([item['positive']['input_ids'] for item in train_triplets])
        positive_attention_masks = np.array([item['positive']['attention_mask'] for item in train_triplets])
        negative_input_ids = np.array([item['negative']['input_ids'] for item in train_triplets])
        negative_attention_masks = np.array([item['negative']['attention_mask'] for item in train_triplets])
        anchor_embeddings = self.model.predict([anchor_input_ids, anchor_attention_masks])
        positive_embeddings = self.model.predict([positive_input_ids, positive_attention_masks])
        negative_embeddings = self.model.predict([negative_input_ids, negative_attention_masks])
        similarities = []
        for i in range(len(anchor_embeddings)):
            similarity_positive = self.cosine_similarity(anchor_embeddings[i], positive_embeddings[i])
            similarity_negative = self.cosine_similarity(anchor_embeddings[i], negative_embeddings[i])
            similarities.append(similarity_positive > similarity_negative)
        accuracy = np.mean(similarities)
        print('Test Accuracy:', accuracy)

    def pipeline(self, dataset_path, snippet_folder_path, num_negatives_per_positive=1, batch_size=32, epochs=5):
        triplets = self.create_triplet_dataset(dataset_path, snippet_folder_path)
        train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
        train_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in train_triplets]
        test_triplets = [{'anchor': t[0], 'positive': t[1], 'negative': t[2]} for t in test_triplets]
        train_data = TripletDataset(train_triplets, self.max_sequence_length, self.tokenizer)
        test_data = TripletDataset(test_triplets, self.max_sequence_length, self.tokenizer)
        train_triplets = [train_data.__getitem__(i) for i in range(len(train_triplets))]
        test_triplets = [test_data.__getitem__(i) for i in range(len(test_triplets))]
        self.train_model(train_data, batch_size=batch_size, epochs=epochs)
        self.evaluate_model(test_data)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    model = TripletModel()
    model.pipeline(dataset_path, snippet_folder_path)