import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# A class for creating and manipulating triplet datasets
class TripletDataset:
    # Initializes the dataset with triplets, batch size, and max sequence length
    def __init__(self, triplets, batch_size, max_sequence_length):
        self.triplets = triplets
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([item['anchor'] for item in triplets] + [item['positive'] for item in triplets] + [item['negative'] for item in triplets])
        self.vocab_size = len(self.tokenizer.word_index) + 1

    # Maps a function to each item in the dataset
    def map_func(self, item):
        # Convert text to sequences for anchor, positive, and negative items
        anchor_sequence = self.tokenizer.texts_to_sequences([item['anchor']])[0]
        positive_sequence = self.tokenizer.texts_to_sequences([item['positive']])[0]
        negative_sequence = self.tokenizer.texts_to_sequences([item['negative']])[0]
        
        # Pad sequences to max sequence length
        return {'anchor_sequence': pad_sequences([anchor_sequence], maxlen=self.max_sequence_length)[0], 
                'positive_sequence': pad_sequences([positive_sequence], maxlen=self.max_sequence_length)[0], 
                'negative_sequence': pad_sequences([negative_sequence], maxlen=self.max_sequence_length)[0]}

    # Creates the dataset
    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.triplets)
        dataset = dataset.map(self.map_func)
        dataset = dataset.batch(self.batch_size)
        return dataset

# Loads data from a dataset path and snippet folder path
def load_data(dataset_path, snippet_folder_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return ({item['instance_id']: item['problem_statement'] for item in dataset},
            [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
             for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))])

# Creates triplets from instance id map and snippets
def create_triplets(instance_id_map, snippets):
    bug_snippets = []
    non_bug_snippets = []
    for _, snippet_file in snippets:
        snippet_data = json.load(open(snippet_file))
        if snippet_data.get('is_bug', False):
            bug_snippets.append(snippet_data['snippet'])
        else:
            non_bug_snippets.append(snippet_data['snippet'])
    bug_snippets = [snippet for snippet in bug_snippets if snippet]
    non_bug_snippets = [snippet for snippet in non_bug_snippets if snippet]
    return [{'anchor': instance_id_map[os.path.basename(folder)], 
             'positive': positive_doc, 
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(1, len(non_bug_snippets)))]

# Creates a model with the given parameters
def create_model(vocab_size, embedding_dim, max_sequence_length):
    # Anchor input
    anchor_input = Input(shape=(max_sequence_length,), name='anchor_input')
    # Positive input
    positive_input = Input(shape=(max_sequence_length,), name='positive_input')
    # Negative input
    negative_input = Input(shape=(max_sequence_length,), name='negative_input')
    
    # Embedding layer
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    # Anchor embedding
    anchor_embedding = embedding(anchor_input)
    # Positive embedding
    positive_embedding = embedding(positive_input)
    # Negative embedding
    negative_embedding = embedding(negative_input)
    
    # Global max pooling
    pooling = GlobalMaxPooling1D()
    anchor_pooling = pooling(anchor_embedding)
    positive_pooling = pooling(positive_embedding)
    negative_pooling = pooling(negative_embedding)
    
    # Dense layer
    dense = Dense(128, activation='relu')
    anchor_dense = dense(anchor_pooling)
    positive_dense = dense(positive_pooling)
    negative_dense = dense(negative_pooling)
    
    # Model
    model = Model(inputs=[anchor_input, positive_input, negative_input], 
                  outputs=[anchor_dense, positive_dense, negative_dense])
    
    # Compile model
    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
    return model

# Calculates the triplet loss
def calculate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeddings - positive_embeddings, axis=1) - tf.norm(anchor_embeddings - negative_embeddings, axis=1), 0))

# Trains the model
def train(model, train_dataset, test_dataset, epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(1, epochs+1):
        train_dataset = train_dataset.shuffle(1000).batch(32)
        total_loss = 0
        for batch in train_dataset:
            anchor_sequences = batch['anchor_sequence']
            positive_sequences = batch['positive_sequence']
            negative_sequences = batch['negative_sequence']
            
            # Gradient tape
            with tf.GradientTape() as tape:
                anchor_output, positive_output, negative_output = model([anchor_sequences, positive_sequences, negative_sequences], training=True)
                batch_loss = calculate_triplet_loss(anchor_output, positive_output, negative_output)
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += batch_loss
        print(f'Epoch {epoch}, Train Loss: {total_loss/len(train_dataset)}')
        train_losses.append(total_loss/len(train_dataset))
        
        total_loss = 0
        total_correct = 0
        for batch in test_dataset:
            anchor_sequences = batch['anchor_sequence']
            positive_sequences = batch['positive_sequence']
            negative_sequences = batch['negative_sequence']
            
            anchor_output, positive_output, negative_output = model([anchor_sequences, positive_sequences, negative_sequences])
            
            batch_loss = calculate_triplet_loss(anchor_output, positive_output, negative_output)
            total_loss += batch_loss
            positive_similarity = tf.reduce_sum(tf.multiply(anchor_output, positive_output), axis=1)
            negative_similarity = tf.reduce_sum(tf.multiply(anchor_output, negative_output), axis=1)
            total_correct += tf.reduce_sum((positive_similarity > negative_similarity))
        accuracy = total_correct / len(test_dataset)
        print(f'Test Loss: {total_loss/len(test_dataset)}')
        print(f'Test Accuracy: {accuracy}')
        test_losses.append(total_loss/len(test_dataset))
        test_accuracies.append(accuracy)
        train_accuracies.append(total_correct / len(train_dataset))
    return train_losses, test_losses, train_accuracies, test_accuracies

# Plots the losses and accuracies
def plot(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Main function
def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    
    instance_id_map, snippets = load_data(dataset_path, snippet_folder_path)
    triplets = create_triplets(instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    
    train_dataset = TripletDataset(train_triplets, 32, 512)
    train_data = train_dataset.create_dataset()
    
    test_dataset = TripletDataset(test_triplets, 32, 512)
    test_data = test_dataset.create_dataset()
    
    model = create_model(train_dataset.vocab_size, 128, 512)
    train_losses, test_losses, train_accuracies, test_accuracies = train(model, train_data, test_data, 5)
    plot(train_losses, test_losses, train_accuracies, test_accuracies)

if __name__ == "__main__":
    main()