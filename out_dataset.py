import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import json
import os

# Constants
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_SIZE = 128
FULLY_CONNECTED_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE_VALUE = 1e-5
EPOCHS = 5
BATCH_SIZE = 32
NUM_NEGATIVES_PER_POSITIVE = 1

def load_data(file_path: str) -> np.ndarray or dict:
    if file_path.endswith('.npy'):
        return np.load(file_path, allow_pickle=True)
    else:
        return json.load(open(file_path, 'r', encoding='utf-8'))

def load_snippets(folder_path: str) -> list:
    return [(os.path.join(folder_path, folder), os.path.join(folder_path, folder, 'snippet.json')) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

def separate_snippets(snippets: list) -> tuple:
    bug_snippets = []
    non_bug_snippets = []
    for folder_path, snippet_file_path in snippets:
        snippet_data = load_data(snippet_file_path)
        if snippet_data.get('is_bug', False):
            bug_snippets.append((snippet_data['snippet'], True))
        else:
            non_bug_snippets.append((snippet_data['snippet'], False))
    return bug_snippets, non_bug_snippets

def create_triplets(problem_statement: str, positive_snippets: list, negative_snippets: list, num_negatives_per_positive: int) -> list:
    return [{'anchor': problem_statement, 'positive': positive_doc[0], 'negative': random.choice(negative_snippets)[0]} for positive_doc in positive_snippets for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def prepare_data(dataset_path: str, snippet_folder_path: str, num_negatives_per_positive: int) -> tuple:
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in load_data(dataset_path)}
    snippets = load_snippets(snippet_folder_path)
    bug_snippets, non_bug_snippets = separate_snippets(snippets)
    triplets = []
    for folder_path, _ in snippets:
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        triplets.extend(create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive))
    train_size = int(len(triplets)*0.8)
    return triplets[:train_size], triplets[train_size:]

def create_dataset(triplets: list, max_sequence_length: int, tokenizer: Tokenizer):
    anchor_texts = [triplet['anchor'] for triplet in triplets]
    positive_texts = [triplet['positive'] for triplet in triplets]
    negative_texts = [triplet['negative'] for triplet in triplets]
    
    anchor_sequences = tokenizer.texts_to_sequences(anchor_texts)
    positive_sequences = tokenizer.texts_to_sequences(positive_texts)
    negative_sequences = tokenizer.texts_to_sequences(negative_texts)
    
    anchor_padded = pad_sequences(anchor_sequences, maxlen=max_sequence_length, padding='post')
    positive_padded = pad_sequences(positive_sequences, maxlen=max_sequence_length, padding='post')
    negative_padded = pad_sequences(negative_sequences, maxlen=max_sequence_length, padding='post')
    
    return anchor_padded, positive_padded, negative_padded

def create_model(max_sequence_length: int, embedding_size: int, fully_connected_size: int, dropout_rate: float):
    inputs = tf.keras.Input(shape=(max_sequence_length,), dtype='float32')
    x = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(fully_connected_size, activation='relu')(x)
    outputs = tf.keras.layers.Dense(embedding_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    positive_distance = tf.reduce_mean(tf.square(anchor_embeddings - positive_embeddings))
    negative_distance = tf.reduce_mean(tf.square(anchor_embeddings - negative_embeddings))
    return positive_distance + tf.maximum(negative_distance - positive_distance, 0)

def train(model, anchor_padded, positive_padded, negative_padded, epochs, learning_rate_value, batch_size):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(anchor_padded), batch_size):
            anchor_batch = anchor_padded[i:i+batch_size]
            positive_batch = positive_padded[i:i+batch_size]
            negative_batch = negative_padded[i:i+batch_size]
            with tf.GradientTape() as tape:
                anchor_embeddings = model(anchor_batch, training=True)
                positive_embeddings = model(positive_batch, training=True)
                negative_embeddings = model(negative_batch, training=True)
                loss = calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
        print(f'Epoch {epoch+1}, Loss: {total_loss / (len(anchor_padded) // batch_size)}')

def evaluate(model, anchor_padded, positive_padded, negative_padded, batch_size):
    total_correct = 0
    for i in range(0, len(anchor_padded), batch_size):
        anchor_batch = anchor_padded[i:i+batch_size]
        positive_batch = positive_padded[i:i+batch_size]
        negative_batch = negative_padded[i:i+batch_size]
        anchor_embeddings = model(anchor_batch)
        positive_embeddings = model(positive_batch)
        negative_embeddings = model(negative_batch)
        for j in range(len(anchor_embeddings)):
            similarity_positive = tf.reduce_sum(tf.multiply(anchor_embeddings[j], positive_embeddings[j])) / (tf.norm(anchor_embeddings[j]) * tf.norm(positive_embeddings[j]))
            similarity_negative = tf.reduce_sum(tf.multiply(anchor_embeddings[j], negative_embeddings[j])) / (tf.norm(anchor_embeddings[j]) * tf.norm(negative_embeddings[j]))
            total_correct += int(similarity_positive > similarity_negative)
    accuracy = total_correct / len(anchor_padded)
    print(f'Test Accuracy: {accuracy}')

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    train_triplets, test_triplets = prepare_data(dataset_path, snippet_folder_path, NUM_NEGATIVES_PER_POSITIVE)
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([triplet['anchor'] for triplet in train_triplets] + [triplet['positive'] for triplet in train_triplets] + [triplet['negative'] for triplet in train_triplets])
    train_anchor, train_positive, train_negative = create_dataset(train_triplets, MAX_SEQUENCE_LENGTH, tokenizer)
    test_anchor, test_positive, test_negative = create_dataset(test_triplets, MAX_SEQUENCE_LENGTH, tokenizer)
    model = create_model(MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE, FULLY_CONNECTED_SIZE, DROPOUT_RATE)
    train(model, train_anchor, train_positive, train_negative, EPOCHS, LEARNING_RATE_VALUE, BATCH_SIZE)
    evaluate(model, test_anchor, test_positive, test_negative, BATCH_SIZE)

if __name__ == "__main__":
    main()