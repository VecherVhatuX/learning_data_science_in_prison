import tensorflow as tf
import numpy as np
import random
import json
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, GlobalAveragePooling1D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CosineSimilarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer

class CodeSimilarityModel:
    def __init__(self):
        self.embedding_size = 128
        self.fully_connected_size = 64
        self.dropout_rate = 0.2
        self.max_sequence_length = 512
        self.learning_rate_value = 1e-5
        self.batch_size = 32
        self.epochs = 5

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

    class Dataset(tf.keras.utils.Sequence):
        def __init__(self, triplets, max_sequence_length, tokenizer, batch_size=32):
            self.triplets = triplets
            self.max_sequence_length = max_sequence_length
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.on_epoch_end()

        def __len__(self):
            return len(self.triplets) // self.batch_size + 1

        def on_epoch_end(self):
            random.shuffle(self.triplets)

        def __getitem__(self, idx):
            batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_anchor_input_ids = []
            batch_anchor_attention_masks = []
            batch_positive_input_ids = []
            batch_positive_attention_masks = []
            batch_negative_input_ids = []
            batch_negative_attention_masks = []
            for triplet in batch_triplets:
                anchor = tf.keras.preprocessing.sequence.pad_sequences(
                    self.tokenizer.texts_to_sequences([triplet['anchor']]),
                    maxlen=self.max_sequence_length,
                    padding='post',
                    truncating='post'
                )
                positive = tf.keras.preprocessing.sequence.pad_sequences(
                    self.tokenizer.texts_to_sequences([triplet['positive']]),
                    maxlen=self.max_sequence_length,
                    padding='post',
                    truncating='post'
                )
                negative = tf.keras.preprocessing.sequence.pad_sequences(
                    self.tokenizer.texts_to_sequences([triplet['negative']]),
                    maxlen=self.max_sequence_length,
                    padding='post',
                    truncating='post'
                )
                batch_anchor_input_ids.append(anchor[0])
                batch_anchor_attention_masks.append([1] * len(anchor[0]))
                batch_positive_input_ids.append(positive[0])
                batch_positive_attention_masks.append([1] * len(positive[0]))
                batch_negative_input_ids.append(negative[0])
                batch_negative_attention_masks.append([1] * len(negative[0]))
            return {
                'anchor': {'input_ids': np.array(batch_anchor_input_ids), 'attention_mask': np.array(batch_anchor_attention_masks)},
                'positive': {'input_ids': np.array(batch_positive_input_ids), 'attention_mask': np.array(batch_positive_attention_masks)},
                'negative': {'input_ids': np.array(batch_negative_input_ids), 'attention_mask': np.array(batch_negative_attention_masks)}
            }

    def create_model(self):
        input_ids = Input(shape=(self.max_sequence_length,), name='input_ids')
        attention_masks = Input(shape=(self.max_sequence_length,), name='attention_masks')
        embedding = Embedding(input_dim=10000, output_dim=self.embedding_size, input_length=self.max_sequence_length)(input_ids)
        pooling = GlobalAveragePooling1D()(embedding)
        dropout = Dropout(self.dropout_rate)(pooling)
        fc1 = Dense(self.fully_connected_size, activation='relu')(dropout)
        fc2 = Dense(self.embedding_size)(fc1)
        model = Model(inputs=[input_ids, attention_masks], outputs=fc2)
        return model

    def train_model(self, model, train_dataset, test_dataset):
        model.compile(loss=lambda y_true, y_pred: 0, optimizer=Adam(self.learning_rate_value), metrics=['accuracy'])
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_dataset:
                anchor_input_ids = batch['anchor']['input_ids']
                anchor_attention_masks = batch['anchor']['attention_mask']
                positive_input_ids = batch['positive']['input_ids']
                positive_attention_masks = batch['positive']['attention_mask']
                negative_input_ids = batch['negative']['input_ids']
                negative_attention_masks = batch['negative']['attention_mask']
                anchor_embeddings = model.predict([anchor_input_ids, anchor_attention_masks])
                positive_embeddings = model.predict([positive_input_ids, positive_attention_masks])
                negative_embeddings = model.predict([negative_input_ids, negative_attention_masks])
                loss = self.calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                total_loss += loss
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataset)}')
        self.evaluate_model(model, test_dataset)

    def evaluate_model(self, model, test_dataset):
        total_correct = 0
        for batch in test_dataset:
            anchor_input_ids = batch['anchor']['input_ids']
            anchor_attention_masks = batch['anchor']['attention_mask']
            positive_input_ids = batch['positive']['input_ids']
            positive_attention_masks = batch['positive']['attention_mask']
            negative_input_ids = batch['negative']['input_ids']
            negative_attention_masks = batch['negative']['attention_mask']
            anchor_embeddings = model.predict([anchor_input_ids, anchor_attention_masks])
            positive_embeddings = model.predict([positive_input_ids, positive_attention_masks])
            negative_embeddings = model.predict([negative_input_ids, negative_attention_masks])
            for i in range(len(anchor_embeddings)):
                similarity_positive = np.dot(anchor_embeddings[i], positive_embeddings[i]) / (np.linalg.norm(anchor_embeddings[i]) * np.linalg.norm(positive_embeddings[i]))
                similarity_negative = np.dot(anchor_embeddings[i], negative_embeddings[i]) / (np.linalg.norm(anchor_embeddings[i]) * np.linalg.norm(negative_embeddings[i]))
                total_correct += similarity_positive > similarity_negative
        accuracy = total_correct / (len(test_dataset) * self.batch_size)
        print(f'Test Accuracy: {accuracy}')

    def calculate_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = tf.reduce_mean(tf.square(anchor_embeddings - positive_embeddings))
        negative_distance = tf.reduce_mean(tf.square(anchor_embeddings - negative_embeddings))
        return positive_distance + tf.maximum(negative_distance - positive_distance, 0)

    def run_pipeline(self, dataset_path, snippet_folder_path, num_negatives_per_positive=1):
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in self.load_dataset(dataset_path)}
        snippets = self.load_snippets(snippet_folder_path)
        triplets = []
        for folder_path, _ in snippets:
            bug_snippets, non_bug_snippets = self.separate_code_snippets([(folder_path, os.path.join(folder_path, 'snippet.json'))])
            problem_statement = instance_id_map.get(os.path.basename(folder_path))
            for bug_snippet, non_bug_snippet in [(bug, non_bug) for bug in bug_snippets for non_bug in non_bug_snippets]:
                triplets.extend(self.create_triplets(problem_statement, [bug_snippet], non_bug_snippets, num_negatives_per_positive))

        train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([triplet['anchor'] for triplet in triplets] + [triplet['positive'] for triplet in triplets] + [triplet['negative'] for triplet in triplets])
        train_data = self.Dataset(train_triplets, 512, tokenizer, batch_size=self.batch_size)
        test_data = self.Dataset(test_triplets, 512, tokenizer, batch_size=self.batch_size)
        model = self.create_model()
        self.train_model(model, train_data, test_data)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    model = CodeSimilarityModel()
    model.run_pipeline(dataset_path, snippet_folder_path)