import json
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

nltk.download('punkt')

class TripletDataset:
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.instance_id_field = 'instance_id'
        self.num_negatives_per_positive = 3
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.batch_size = 16

    def create_instance_id_map(self, dataset):
        return {item['instance_id']: item['problem_statement'] for item in dataset}

    def load_dataset_file(self):
        return np.load(self.dataset_path, allow_pickle=True)

    def read_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def get_subfolder_paths(self, folder_path):
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def separate_snippets(self, snippets):
        return [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')], \
               [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)} 
                for positive_doc in positive_snippets for _ in range(min(self.num_negatives_per_positive, len(negative_snippets)))]

    def encode_triplet(self, triplet):
        encoded_triplet = {}
        encoded_triplet['anchor_input_ids'] = self.tokenizer.encode_plus(
            text=triplet['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten()
        encoded_triplet['anchor_attention_mask'] = self.tokenizer.encode_plus(
            text=triplet['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten()
        encoded_triplet['positive_input_ids'] = self.tokenizer.encode_plus(
            text=triplet['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten()
        encoded_triplet['positive_attention_mask'] = self.tokenizer.encode_plus(
            text=triplet['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten()
        encoded_triplet['negative_input_ids'] = self.tokenizer.encode_plus(
            text=triplet['negative'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten()
        encoded_triplet['negative_attention_mask'] = self.tokenizer.encode_plus(
            text=triplet['negative'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten()
        return encoded_triplet

    def create_triplet_dataset_generator(self):
        dataset = self.load_dataset_file()
        instance_id_map = self.create_instance_id_map(dataset)
        for folder in os.listdir(self.snippet_folder_path):
            folder_path = os.path.join(self.snippet_folder_path, folder)
            if os.path.isdir(folder_path):
                snippet_file = os.path.join(folder_path, 'snippet.json')
                snippets = self.read_json_file(snippet_file)
                if snippets:
                    bug_snippets, non_bug_snippets = self.separate_snippets(snippets)
                    problem_statement = instance_id_map.get(folder)
                    if problem_statement:
                        triplets = self.create_triplets(problem_statement, bug_snippets, non_bug_snippets)
                        for triplet in triplets:
                            yield self.encode_triplet(triplet)

    def get_dataset(self):
        dataset_generator = self.create_triplet_dataset_generator()
        dataset = tf.data.Dataset.from_generator(lambda: dataset_generator, 
                                                 output_types={'anchor_input_ids': tf.int32, 
                                                               'anchor_attention_mask': tf.int32, 
                                                               'positive_input_ids': tf.int32, 
                                                               'positive_attention_mask': tf.int32, 
                                                               'negative_input_ids': tf.int32, 
                                                               'negative_attention_mask': tf.int32}).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

class TripletModel(tf.keras.Model):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=30522, output_dim=128, input_length=512)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.fc = tf.keras.layers.Dense(64, activation='relu')

    def call(self, inputs):
        anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = inputs
        anchor_outputs = self.embedding(anchor_input_ids)
        anchor_outputs = self.dropout(anchor_outputs)
        anchor_outputs = tf.reduce_mean(anchor_outputs, axis=1)
        anchor_outputs = self.fc(anchor_outputs)

        positive_outputs = self.embedding(positive_input_ids)
        positive_outputs = self.dropout(positive_outputs)
        positive_outputs = tf.reduce_mean(positive_outputs, axis=1)
        positive_outputs = self.fc(positive_outputs)

        negative_outputs = self.embedding(negative_input_ids)
        negative_outputs = self.dropout(negative_outputs)
        negative_outputs = tf.reduce_mean(negative_outputs, axis=1)
        negative_outputs = self.fc(negative_outputs)

        return tf.concat([anchor_outputs, positive_outputs, negative_outputs], axis=0)

def train_model(model, dataset, epochs):
    loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.norm(y_pred[:64] - y_pred[64:128], axis=1) - tf.norm(y_pred[:64] - y_pred[128:], axis=1) + 1)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                outputs = model([batch['anchor_input_ids'], batch['anchor_attention_mask'], 
                                 batch['positive_input_ids'], batch['positive_attention_mask'], 
                                 batch['negative_input_ids'], batch['negative_attention_mask']])
                loss = loss_fn(tf.zeros((outputs.shape[0],)), outputs)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

def main(dataset_path, snippet_folder_path):
    dataset = TripletDataset(dataset_path, snippet_folder_path)
    data = dataset.get_dataset()
    model = TripletModel()
    train_model(model, data, epochs=5)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    main(dataset_path, snippet_folder_path)