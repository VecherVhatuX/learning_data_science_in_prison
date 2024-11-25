import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import json
import os
from sklearn.model_selection import train_test_split

# Hyperparameters
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_SIZE = 128
FULLY_CONNECTED_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE_VALUE = 1e-5
EPOCHS = 5
BATCH_SIZE = 32
NUM_NEGATIVES_PER_POSITIVE = 1

def load_data(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    else:
        return json.load(open(file_path, 'r', encoding='utf-8'))

def load_snippets(snippet_folder_path):
    return [(os.path.join(snippet_folder_path, folder), os.path.join(snippet_folder_path, folder, 'snippet.json')) 
            for folder in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, folder))]

def separate_snippets(snippets):
    bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                    for snippet_data in [load_data(snippet_file_path)] if snippet_data.get('is_bug', False)]
    non_bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                        for snippet_data in [load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)]
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

def tokenize_triplets(triplets, max_sequence_length):
    tokenizer = Tokenizer()
    all_texts = [triplet['anchor'] for triplet in triplets] + [triplet['positive'] for triplet in triplets] + [triplet['negative'] for triplet in triplets]
    tokenizer.fit_on_texts(all_texts)
    anchor_sequences = tokenizer.texts_to_sequences([triplet['anchor'] for triplet in triplets])
    positive_sequences = tokenizer.texts_to_sequences([triplet['positive'] for triplet in triplets])
    negative_sequences = tokenizer.texts_to_sequences([triplet['negative'] for triplet in triplets])
    anchor_padded = pad_sequences(anchor_sequences, maxlen=max_sequence_length, padding='post')
    positive_padded = pad_sequences(positive_sequences, maxlen=max_sequence_length, padding='post')
    negative_padded = pad_sequences(negative_sequences, maxlen=max_sequence_length, padding='post')
    return np.stack((anchor_padded, positive_padded, negative_padded), axis=1)

def create_model(embedding_size, max_sequence_length, fully_connected_size, dropout_rate):
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=10000, output_dim=embedding_size, input_length=max_sequence_length),
        layers.Bidirectional(layers.LSTM(fully_connected_size, dropout=dropout_rate)),
        layers.Dense(fully_connected_size, activation='relu'),
        layers.Dense(embedding_size)
    ])
    return model

def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    return tf.reduce_mean((anchor_embeddings - positive_embeddings) ** 2) + tf.keras.backend.maximum(tf.reduce_mean((anchor_embeddings - negative_embeddings) ** 2) - tf.reduce_mean((anchor_embeddings - positive_embeddings) ** 2), 0)

def train(model, dataset, learning_rate_value):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)
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

def train_model(model, train_dataset, test_dataset, epochs, learning_rate_value):
    for epoch in range(epochs):
        loss = train(model, train_dataset, learning_rate_value)
        print(f'Epoch {epoch+1}, Loss: {loss}')
    print(f'Test Accuracy: {evaluate(model, test_dataset)}')

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    train_triplets, test_triplets = prepare_data(dataset_path, snippet_folder_path, NUM_NEGATIVES_PER_POSITIVE)
    train_tokenized_triplets = tokenize_triplets(train_triplets, MAX_SEQUENCE_LENGTH)
    test_tokenized_triplets = tokenize_triplets(test_triplets, MAX_SEQUENCE_LENGTH)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_tokenized_triplets).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_tokenized_triplets).batch(BATCH_SIZE)
    model = create_model(EMBEDDING_SIZE, MAX_SEQUENCE_LENGTH, FULLY_CONNECTED_SIZE, DROPOUT_RATE)
    train_model(model, train_dataset, test_dataset, EPOCHS, LEARNING_RATE_VALUE)

if __name__ == "__main__":
    main()