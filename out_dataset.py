import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import random
import json
import os

class DataHandler:
    @staticmethod
    def load_data(file_path):
        return np.load(file_path, allow_pickle=True) if file_path.endswith('.npy') else json.load(open(file_path, 'r', encoding='utf-8'))

    @staticmethod
    def load_snippets(folder_path):
        return [(os.path.join(folder_path, f), os.path.join(folder_path, f, 'snippet.json')) 
                for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

class DataPreprocessor:
    @staticmethod
    def separate_snippets(snippets):
        return tuple(map(list, zip(*[
            ((snippet_data['snippet'], True) if snippet_data.get('is_bug', False) else (snippet_data['snippet'], False)) 
            for folder_path, snippet_file_path in snippets 
            for snippet_data in [DataHandler.load_data(snippet_file_path)]
            if snippet_data.get('snippet')
        ])))

    @staticmethod
    def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)} 
                for positive_doc in positive_snippets 
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

class Dataset:
    def __init__(self, triplets, max_sequence_length, tokenizer, batch_size=32):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def on_epoch_end(self):
        random.shuffle(self.triplets)

    def __getitem__(self, idx):
        batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        inputs = []
        attention_masks = []
        for triplet in batch_triplets:
            anchor = pad_sequences(
                self.tokenizer.texts_to_sequences([triplet['anchor']]),
                maxlen=self.max_sequence_length,
                padding='post',
                truncating='post'
            )
            positive = pad_sequences(
                self.tokenizer.texts_to_sequences([triplet['positive']]),
                maxlen=self.max_sequence_length,
                padding='post',
                truncating='post'
            )
            negative = pad_sequences(
                self.tokenizer.texts_to_sequences([triplet['negative']]),
                maxlen=self.max_sequence_length,
                padding='post',
                truncating='post'
            )
            inputs.extend([anchor[0], positive[0], negative[0]])
            attention_masks.extend([[1] * len(anchor[0]), [1] * len(positive[0]), [1] * len(negative[0])])
        return {'input_ids': np.array(inputs), 'attention_mask': np.array(attention_masks)}

    def __len__(self):
        return len(self.triplets) // self.batch_size + 1

class ModelBuilder:
    @staticmethod
    def build_model(embedding_size, fully_connected_size, dropout_rate, max_sequence_length):
        inputs = Input(shape=(max_sequence_length,), name='input_ids')
        attention_masks = Input(shape=(max_sequence_length,), name='attention_masks')
        embedding = Embedding(input_dim=10000, output_dim=embedding_size, input_length=max_sequence_length)(inputs)
        pooling = GlobalAveragePooling1D()(embedding)
        dropout = Dropout(dropout_rate)(pooling)
        fc1 = Dense(fully_connected_size, activation='relu')(dropout)
        fc2 = Dense(embedding_size)(fc1)
        model = Model(inputs=[inputs, attention_masks], outputs=fc2)
        return model

class ModelTrainer:
    @staticmethod
    def train(model, dataset, epochs, learning_rate_value):
        model.compile(loss=lambda y_true, y_pred: 0, optimizer=Adam(learning_rate_value), metrics=['accuracy'])
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataset:
                inputs = batch['input_ids']
                attention_masks = batch['attention_mask']
                inputs = np.split(inputs, 3, axis=0)
                attention_masks = np.split(attention_masks, 3, axis=0)
                with tf.GradientTape() as tape:
                    anchor_embeddings = model([inputs[0], attention_masks[0]], training=True)
                    positive_embeddings = model([inputs[1], attention_masks[1]], training=True)
                    negative_embeddings = model([inputs[2], attention_masks[2]], training=True)
                    loss = ModelTrainer.calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                total_loss += loss
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataset)}')

    @staticmethod
    def evaluate(model, dataset):
        total_correct = 0
        for batch in dataset:
            inputs = batch['input_ids']
            attention_masks = batch['attention_mask']
            inputs = np.split(inputs, 3, axis=0)
            attention_masks = np.split(attention_masks, 3, axis=0)
            anchor_embeddings = model([inputs[0], attention_masks[0]])
            positive_embeddings = model([inputs[1], attention_masks[1]])
            negative_embeddings = model([inputs[2], attention_masks[2]])
            for i in range(len(anchor_embeddings)):
                similarity_positive = np.dot(anchor_embeddings[i], positive_embeddings[i]) / (np.linalg.norm(anchor_embeddings[i]) * np.linalg.norm(positive_embeddings[i]))
                similarity_negative = np.dot(anchor_embeddings[i], negative_embeddings[i]) / (np.linalg.norm(anchor_embeddings[i]) * np.linalg.norm(negative_embeddings[i]))
                total_correct += similarity_positive > similarity_negative
        accuracy = total_correct / (len(dataset) * 32)
        print(f'Test Accuracy: {accuracy}')

    @staticmethod
    def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = tf.reduce_mean(tf.square(anchor_embeddings - positive_embeddings))
        negative_distance = tf.reduce_mean(tf.square(anchor_embeddings - negative_embeddings))
        return positive_distance + tf.maximum(negative_distance - positive_distance, 0)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    num_negatives_per_positive = 1
    embedding_size = 128
    fully_connected_size = 64
    dropout_rate = 0.2
    max_sequence_length = 512
    learning_rate_value = 1e-5
    epochs = 5
    batch_size = 32

    instance_id_map = {item['instance_id']: item['problem_statement'] for item in DataHandler.load_data(dataset_path)}
    snippets = DataHandler.load_snippets(snippet_folder_path)
    triplets = []
    for folder_path, _ in snippets:
        bug_snippets, non_bug_snippets = DataPreprocessor.separate_snippets([(folder_path, os.path.join(folder_path, 'snippet.json'))])
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        for bug_snippet, non_bug_snippet in [(bug, non_bug) for bug in bug_snippets for non_bug in non_bug_snippets]:
            triplets.extend(DataPreprocessor.create_triplets(problem_statement, [bug_snippet], non_bug_snippets, num_negatives_per_positive))

    train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([triplet['anchor'] for triplet in triplets] + [triplet['positive'] for triplet in triplets] + [triplet['negative'] for triplet in triplets])
    train_data = Dataset(train_triplets, max_sequence_length, tokenizer, batch_size=batch_size)
    test_data = Dataset(test_triplets, max_sequence_length, tokenizer, batch_size=batch_size)
    model = ModelBuilder.build_model(embedding_size, fully_connected_size, dropout_rate, max_sequence_length)
    ModelTrainer.train(model, train_data, epochs, learning_rate_value)
    ModelTrainer.evaluate(model, test_data)

if __name__ == "__main__":
    main()