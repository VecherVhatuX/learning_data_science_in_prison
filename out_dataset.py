import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, Sigmoid
import json
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
import numpy as np

# Function to load JSON data from a specified file path.
def fetch_json_data(path):
    return json.load(open(path))

# Function to gather directories containing snippets in a given folder.
def gather_snippet_directories(folder):
    # Iterate through each item in the folder, checking if it's a directory and returning its path along with the snippet JSON path.
    return [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
            for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

# Function to separate code snippets into bug and non-bug categories.
def separate_snippet_types(snippets):
    # Extract bug snippets from the provided list based on the 'is_bug' flag in the JSON data.
    bug_snippets = [fetch_json_data(path)['snippet'] for _, path in snippets 
                    if fetch_json_data(path).get('is_bug', False)]
    # Extract non-bug snippets from the provided list based on the absence of the 'is_bug' flag in the JSON data.
    non_bug_snippets = [fetch_json_data(path)['snippet'] for _, path in snippets 
                        if not fetch_json_data(path).get('is_bug', False)]
    return bug_snippets, non_bug_snippets

# Function to construct triplets for training the model.
def construct_triplets(num_negatives, instance_id_map, snippets):
    # Separate bug and non-bug snippets using the separate_snippet_types function.
    bug_snippets, non_bug_snippets = separate_snippet_types(snippets)
    # Create a list of triplets with anchor, positive, and negative samples.
    return [{'anchor': instance_id_map[os.path.basename(folder)], 
             'positive': positive_doc, 
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(num_negatives, len(non_bug_snippets)))]

# Function to load dataset from a specified path and snippet folder.
def load_dataset(dataset_path, snippet_folder_path):
    # Load instance ID map from the specified dataset path.
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in fetch_json_data(dataset_path)}
    # Gather snippet directories using the gather_snippet_directories function.
    snippets = gather_snippet_directories(snippet_folder_path)
    return instance_id_map, snippets

# Custom dataset class for code snippets.
class CodeSnippetDataset:
    # Initialize the dataset with a list of triplets and maximum sequence length.
    def __init__(self, triplets, max_sequence_length):
        self.triplets = triplets
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()

    # Return the length of the dataset.
    def __len__(self):
        return len(self.triplets)

    # Return a dictionary containing anchor, positive, and negative samples for a given index.
    def __getitem__(self, idx):
        # Extract anchor, positive, and negative samples from the triplet at the given index.
        anchor = self.triplets[idx]['anchor']
        positive = self.triplets[idx]['positive']
        negative = self.triplets[idx]['negative']

        # Tokenize anchor, positive, and negative samples using the tokenizer.
        self.tokenizer.fit_on_texts([anchor, positive, negative])
        anchor_sequence = self.tokenizer.texts_to_sequences([anchor])[0]
        positive_sequence = self.tokenizer.texts_to_sequences([positive])[0]
        negative_sequence = self.tokenizer.texts_to_sequences([negative])[0]

        # Pad sequences to maximum sequence length.
        anchor_sequence = pad_sequences([anchor_sequence], maxlen=self.max_sequence_length)[0]
        positive_sequence = pad_sequences([positive_sequence], maxlen=self.max_sequence_length)[0]
        negative_sequence = pad_sequences([negative_sequence], maxlen=self.max_sequence_length)[0]

        # Return a dictionary containing input IDs and attention masks for anchor, positive, and negative samples.
        return {'anchor': anchor_sequence, 
                'positive': positive_sequence, 
                'negative': negative_sequence}

# Custom neural network model for triplet learning.
class TripletNetwork(Model):
    # Initialize the model with embedding size, fully connected size, and dropout rate.
    def __init__(self, embedding_size, fully_connected_size, dropout_rate, vocab_size, max_sequence_length):
        super(TripletNetwork, self).__init__()
        # Initialize embedding layer.
        self.embedding = Embedding(vocab_size, embedding_size, input_length=max_sequence_length)
        # Initialize LSTM layer.
        self.lstm = LSTM(fully_connected_size)
        # Initialize dropout layer.
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # Initialize fully connected layers.
        self.fc1 = Dense(fully_connected_size, activation='relu')
        self.fc2 = Dense(embedding_size)

    # Forward pass through the model.
    def call(self, x):
        # Extract anchor, positive, and negative input IDs.
        anchor_sequence = x['anchor']
        positive_sequence = x['positive']
        negative_sequence = x['negative']

        # Pass anchor, positive, and negative samples through the embedding layer.
        anchor_embedding = self.embedding(anchor_sequence)
        positive_embedding = self.embedding(positive_sequence)
        negative_embedding = self.embedding(negative_sequence)

        # Pass anchor, positive, and negative samples through the LSTM layer.
        anchor_output = self.lstm(anchor_embedding)
        positive_output = self.lstm(positive_embedding)
        negative_output = self.lstm(negative_embedding)

        # Pass output through fully connected layers to obtain embeddings.
        anchor_embedding = self.fc2(self.fc1(self.dropout(anchor_output)))
        positive_embedding = self.fc2(self.fc1(self.dropout(positive_output)))
        negative_embedding = self.fc2(self.fc1(self.dropout(negative_output)))

        # Return anchor, positive, and negative embeddings.
        return anchor_embedding, positive_embedding, negative_embedding

# Custom trainer class for the triplet network.
class TripletTrainer:
    # Initialize the trainer with the model and device.
    def __init__(self, model):
        self.model = model

    # Function to calculate triplet loss.
    def calculate_triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        # Calculate distance between anchor and positive embeddings.
        positive_distance = tf.reduce_mean((anchor_embeddings - positive_embeddings) ** 2)
        # Calculate distance between anchor and negative embeddings.
        negative_distance = tf.reduce_mean((anchor_embeddings - negative_embeddings) ** 2)
        # Return triplet loss.
        return positive_distance + tf.maximum(negative_distance - positive_distance, 0.0)

    # Function to train the triplet network.
    def train_triplet_network(self, train_dataset, optimizer, epochs):
        # Iterate through each epoch.
        for epoch in range(epochs):
            # Iterate through each batch in the train dataset.
            for batch in train_dataset:
                # Zero out gradients.
                with tf.GradientTape() as tape:
                    # Calculate batch loss.
                    batch_loss = self.calculate_triplet_loss(*self.model(batch))
                # Backpropagate gradients.
                gradients = tape.gradient(batch_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                # Print batch loss.
                print(f'Epoch {epoch+1}, Batch Loss: {batch_loss.numpy()}')

    # Function to evaluate the triplet network.
    def evaluate_triplet_network(self, test_dataset):
        # Initialize correct count.
        total_correct = 0
        # Iterate through each batch in the test dataset.
        for batch in test_dataset:
            # Pass batch through the model to obtain embeddings.
            anchor_embeddings, positive_embeddings, negative_embeddings = self.model(batch)
            # Iterate through each anchor embedding.
            for i in range(len(anchor_embeddings)):
                # Calculate similarity between anchor and positive embeddings.
                positive_similarity = tf.reduce_sum(anchor_embeddings[i] * positive_embeddings[i]) / (tf.norm(anchor_embeddings[i]) * tf.norm(positive_embeddings[i]))
                # Calculate similarity between anchor and negative embeddings.
                negative_similarity = tf.reduce_sum(anchor_embeddings[i] * negative_embeddings[i]) / (tf.norm(anchor_embeddings[i]) * tf.norm(negative_embeddings[i]))
                # Check if positive similarity is greater than negative similarity.
                if positive_similarity > negative_similarity:
                    # Increment correct count.
                    total_correct += 1
        # Return accuracy.
        return total_correct / len(test_dataset)

# Main function.
def main():
    # Specify dataset path and snippet folder path.
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    # Load dataset and snippet directories.
    instance_id_map, snippets = load_dataset(dataset_path, snippet_folder_path)
    # Construct triplets.
    triplets = construct_triplets(1, instance_id_map, snippets)
    # Split triplets into train and test sets.
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    # Create train and test datasets.
    train_dataset = CodeSnippetDataset(train_triplets, 512)
    test_dataset = CodeSnippetDataset(test_triplets, 512)
    # Create train and test data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices(train_dataset.triplets).batch(32)
    test_loader = tf.data.Dataset.from_tensor_slices(test_dataset.triplets).batch(32)

    # Initialize model.
    model = TripletNetwork(128, 64, 0.2, 10000, 512)
    # Initialize optimizer.
    optimizer = tf.keras.optimizers.Adam(1e-5)

    # Initialize trainer.
    trainer = TripletTrainer(model)
    # Train model.
    trainer.train_triplet_network(train_loader, optimizer, 5)
    # Evaluate model.
    print(f'Test Accuracy: {trainer.evaluate_triplet_network(test_loader)}')

if __name__ == "__main__":
    main()