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

    def create_instance_id_map(self, dataset):
        instance_id_map = {}
        for item in dataset:
            instance_id_map[item['instance_id']] = item['problem_statement']
        return instance_id_map

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
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        return [os.path.join(folder_path, f) for f in subfolders]

    def separate_snippets(self, snippets):
        bug_snippets = [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')]
        non_bug_snippets = [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
        return bug_snippets, non_bug_snippets

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets):
        triplets = []
        for positive_doc in positive_snippets:
            negative_docs = random.sample(negative_snippets, min(self.num_negatives_per_positive, len(negative_snippets)))
            for negative_doc in negative_docs:
                triplets.append({'anchor': problem_statement, 'positive': positive_doc, 'negative': negative_doc})
        return triplets

    def encode_triplet(self, tokenizer, triplet):
        encoded_triplet = {}
        encoded_triplet['anchor_input_ids'] = tokenizer.encode_plus(
            text=triplet['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten()
        encoded_triplet['anchor_attention_mask'] = tokenizer.encode_plus(
            text=triplet['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten()
        encoded_triplet['positive_input_ids'] = tokenizer.encode_plus(
            text=triplet['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten()
        encoded_triplet['positive_attention_mask'] = tokenizer.encode_plus(
            text=triplet['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten()
        encoded_triplet['negative_input_ids'] = tokenizer.encode_plus(
            text=triplet['negative'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten()
        encoded_triplet['negative_attention_mask'] = tokenizer.encode_plus(
            text=triplet['negative'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten()
        return encoded_triplet

    def create_triplet_dataset(self):
        dataset = self.load_dataset_file()
        instance_id_map = self.create_instance_id_map(dataset)
        triplet_data = []
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
                        triplet_data.extend(triplets)
        print(f"Number of triplets: {len(triplet_data)}")
        return triplet_data

    def create_triplet_dataset_generator(self, tokenizer):
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
                            yield self.encode_triplet(tokenizer, triplet)

class TripletModel:
    def __init__(self):
        self.embedding = tf.keras.layers.Embedding(input_dim=30522, output_dim=128, input_length=512)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.fc = tf.keras.layers.Dense(64, activation='relu')
        self.inputs = tf.keras.Input(shape=(512,))
        self.attention_mask = tf.keras.Input(shape=(512,))

    def build_model(self):
        outputs = self.embedding(self.inputs)
        outputs = self.dropout(outputs)
        outputs = tf.reduce_mean(outputs, axis=1)
        outputs = self.fc(outputs)
        return tf.keras.Model(inputs=[self.inputs, self.attention_mask], outputs=outputs)

    def train_model(self, dataset, epochs):
        loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.norm(y_pred[:64] - y_pred[64:128], axis=1) - tf.norm(y_pred[:64] - y_pred[128:], axis=1) + 1)
        optimizer = tf.keras.optimizers.Adam(1e-5)
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataset:
                anchor_input_ids = batch['anchor_input_ids'].numpy()
                anchor_attention_mask = batch['anchor_attention_mask'].numpy()
                positive_input_ids = batch['positive_input_ids'].numpy()
                positive_attention_mask = batch['positive_attention_mask'].numpy()
                negative_input_ids = batch['negative_input_ids'].numpy()
                negative_attention_mask = batch['negative_attention_mask'].numpy()

                anchor_inputs = (anchor_input_ids, anchor_attention_mask)
                positive_inputs = (positive_input_ids, positive_attention_mask)
                negative_inputs = (negative_input_ids, negative_attention_mask)

                anchor_output = self.build_model().predict(anchor_inputs)
                positive_output = self.build_model().predict(positive_inputs)
                negative_output = self.build_model().predict(negative_inputs)

                with tf.GradientTape() as tape:
                    outputs = tf.concat([anchor_output, positive_output, negative_output], axis=0)
                    loss = loss_fn(tf.zeros((anchor_output.shape[0]*3,)), outputs)

                gradients = tape.gradient(loss, self.build_model().trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.build_model().trainable_variables))
                total_loss += loss
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

def main(dataset_path, snippet_folder_path):
    dataset = TripletDataset(dataset_path, snippet_folder_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset_generator = dataset.create_triplet_dataset_generator(tokenizer)
    dataset = tf.data.Dataset.from_generator(lambda: dataset_generator, 
                                             output_types={'anchor_input_ids': tf.int32, 
                                                           'anchor_attention_mask': tf.int32, 
                                                           'positive_input_ids': tf.int32, 
                                                           'positive_attention_mask': tf.int32, 
                                                           'negative_input_ids': tf.int32, 
                                                           'negative_attention_mask': tf.int32}).batch(16).prefetch(tf.data.AUTOTUNE)

    model = TripletModel()
    model.train_model(dataset, epochs=5)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    main(dataset_path, snippet_folder_path)