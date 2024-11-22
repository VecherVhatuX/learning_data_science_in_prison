import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import json
import os
from sklearn.model_selection import train_test_split

# Constants
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_SIZE = 128
FULLY_CONNECTED_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE_VALUE = 1e-5
EPOCHS = 5
BATCH_SIZE = 32
NUM_NEGATIVES_PER_POSITIVE = 1

def load_data(file_path):
    return np.load(file_path, allow_pickle=True) if file_path.endswith('.npy') else json.load(open(file_path, 'r', encoding='utf-8'))

def load_snippets(folder_path):
    return [(os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

def separate_snippets(snippets):
    bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets for snippet_data in [load_data(snippet_file_path)] if snippet_data.get('is_bug', False)]
    non_bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets for snippet_data in [load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)]
    return bug_snippets, non_bug_snippets

def create_triplets(instance_id_map, snippets, num_negatives_per_positive):
    bug_snippets, non_bug_snippets = separate_snippets(snippets)
    return [{'anchor': instance_id_map[os.path.basename(folder_path)], 'positive': positive_doc, 'negative': random.choice(non_bug_snippets)} 
            for folder_path, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(num_negatives_per_positive, len(non_bug_snippets)))]

def prepare_data(dataset_path, snippet_folder_path, num_negatives_per_positive):
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    snippets = load_snippets(snippet_folder_path)
    triplets = create_triplets(instance_id_map, snippets, num_negatives_per_positive)
    return np.array_split(np.array(triplets), 2)

def tokenize_triplets(triplets):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([triplet['anchor'] for triplet in triplets] + [triplet['positive'] for triplet in triplets] + [triplet['negative'] for triplet in triplets])
    anchor_sequences = tokenizer.texts_to_sequences([triplet['anchor'] for triplet in triplets])
    positive_sequences = tokenizer.texts_to_sequences([triplet['positive'] for triplet in triplets])
    negative_sequences = tokenizer.texts_to_sequences([triplet['negative'] for triplet in triplets])
    anchor_padded = pad_sequences(anchor_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    positive_padded = pad_sequences(positive_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    negative_padded = pad_sequences(negative_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return np.stack((anchor_padded, positive_padded, negative_padded), axis=1)

def create_dataset(triplets):
    tokenized_triplets = tokenize_triplets(triplets)
    return tf.data.Dataset.from_tensor_slices(tokenized_triplets).batch(BATCH_SIZE)

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = layers.Embedding(input_dim=10000, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQUENCE_LENGTH)
        self.bi_lstm = layers.Bidirectional(layers.LSTM(FULLY_CONNECTED_SIZE, dropout=DROPOUT_RATE))
        self.fully_connected = layers.Dense(FULLY_CONNECTED_SIZE, activation='relu')
        self.embedding_size = layers.Dense(EMBEDDING_SIZE)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.bi_lstm(x)
        x = self.fully_connected(x)
        x = self.embedding_size(x)
        return x

def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    return tf.reduce_mean((anchor_embeddings - positive_embeddings) ** 2) + tf.keras.backend.maximum(tf.reduce_mean((anchor_embeddings - negative_embeddings) ** 2) - tf.reduce_mean((anchor_embeddings - positive_embeddings) ** 2), 0)

def train(model, dataset):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_VALUE)
    total_loss = 0
    for batch in dataset:
        with tf.GradientTape() as tape:
            anchor_embeddings = model(batch[:, 0])
            positive_embeddings = model(batch[:, 1])
            negative_embeddings = model(batch[:, 2])
            loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
    return total_loss / len(dataset)

def evaluate(model, dataset):
    total_correct = 0
    for batch in dataset:
        anchor_embeddings = model(batch[:, 0])
        positive_embeddings = model(batch[:, 1])
        negative_embeddings = model(batch[:, 2])
        for i in range(len(anchor_embeddings)):
            similarity_positive = tf.reduce_sum(anchor_embeddings[i] * positive_embeddings[i]) / (tf.norm(anchor_embeddings[i]) * tf.norm(positive_embeddings[i]))
            similarity_negative = tf.reduce_sum(anchor_embeddings[i] * negative_embeddings[i]) / (tf.norm(anchor_embeddings[i]) * tf.norm(negative_embeddings[i]))
            total_correct += int(similarity_positive > similarity_negative)
    return total_correct / len(dataset)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    train_triplets, test_triplets = prepare_data(dataset_path, snippet_folder_path, NUM_NEGATIVES_PER_POSITIVE)
    train_dataset = create_dataset(train_triplets)
    test_dataset = create_dataset(test_triplets)
    model = Model()
    for epoch in range(EPOCHS):
        loss = train(model, train_dataset)
        print(f'Epoch {epoch+1}, Loss: {loss}')
    print(f'Test Accuracy: {evaluate(model, test_dataset)}')

if __name__ == "__main__":
    main()