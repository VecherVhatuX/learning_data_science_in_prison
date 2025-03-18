import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DataProcessor:
    """Processes and encodes text data for triplet loss training."""
    def __init__(self, data):
        """Initializes the DataProcessor with the given data and fits a LabelEncoder."""
        self.data = data
        self.encoder = LabelEncoder().fit(self._gather_texts(data))
    
    def _gather_texts(self, data):
        """Collects all anchor, positive, and negative texts from the data."""
        return [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]
    
    def retrieve_data(self):
        """Returns the processed data."""
        return self.data

class TripletDataset(tf.data.Dataset):
    """Custom TensorFlow Dataset for handling triplet data."""
    def __init__(self, data):
        """Initializes the dataset with processed data."""
        self.data = DataProcessor(data)
        self.samples = self.data.retrieve_data()
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Retrieves a single triplet sample by index."""
        item = self.samples[idx]
        return {
            'anchor_seq': tf.convert_to_tensor(self.data.encoder.transform([item['anchor']])[0]),
            'positive_seq': tf.convert_to_tensor(self.data.encoder.transform([item['positive']])[0]),
            'negative_seq': tf.convert_to_tensor(self.data.encoder.transform([item['negative']])[0])
        }

class EmbeddingModel(models.Model):
    """Neural network model for generating embeddings."""
    def __init__(self, vocab_size, embed_dim):
        """Initializes the model with embedding and dense layers."""
        super(EmbeddingModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.network = models.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])
    
    def call(self, anchor, positive, negative):
        """Generates embeddings for anchor, positive, and negative samples."""
        return (
            self.network(self.embedding(anchor)),
            self.network(self.embedding(positive)),
            self.network(self.embedding(negative))
        )

def calculate_loss(anchor, positive, negative):
    """Calculates the triplet loss for the given embeddings."""
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1), 0))

def train_model(model, train_data, valid_data, epochs):
    """Trains the model using the provided training and validation data."""
    optimizer = optimizers.Adam(learning_rate=0.001)
    scheduler = optimizers.schedules.ExponentialDecay(0.001, decay_steps=1000, decay_rate=0.1)
    history = []
    
    for _ in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_data:
            with tf.GradientTape() as tape:
                anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
                loss = calculate_loss(anchor, positive, negative)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss.numpy()
        train_loss /= len(train_data)
        eval_loss, accuracy = evaluate_model(model, valid_data)
        history.append((train_loss, eval_loss, accuracy))
    return history

def evaluate_model(model, data):
    """Evaluates the model on the provided data."""
    model.eval()
    total_loss = 0
    correct = 0
    for batch in data:
        anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        total_loss += calculate_loss(anchor, positive, negative).numpy()
        correct += count_accurate(anchor, positive, negative)
    return total_loss / len(data), correct / len(data)

def count_accurate(anchor, positive, negative):
    """Counts the number of accurate predictions based on cosine similarity."""
    return tf.reduce_sum(tf.cast(tf.reduce_sum(anchor * positive, axis=1) > tf.reduce_sum(anchor * negative, axis=1), tf.float32))

def display_results(history):
    """Displays training and validation loss and accuracy over epochs."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in history], label='Training Loss')
    plt.plot([x[1] for x in history], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([x[2] for x in history], label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def store_model(model, path):
    """Saves the model weights to the specified path."""
    model.save_weights(path)
    print(f'Model saved at {path}')

def load_model(model, path):
    """Loads the model weights from the specified path."""
    model.load_weights(path)
    print(f'Model loaded from {path}')
    return model

def visualize_embeddings(model, data):
    """Visualizes the embeddings in a 3D scatter plot."""
    model.eval()
    embeddings = []
    for batch in data:
        anchor, _, _ = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        embeddings.append(anchor.numpy())
    embeddings = np.concatenate(embeddings)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c='Spectral')
    plt.title('3D Embedding Visualization')
    plt.show()

def fetch_data(file_path, root_dir):
    """Fetches and maps data from the specified file and directory."""
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def process_data(mapping, snippet_files):
    """Processes the data into a format suitable for triplet loss training."""
    return [{'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
            for dir, _ in snippet_files for bug_sample, non_bug_samples in [json.load(open(path)) for path in snippet_files]]

def run_pipeline():
    """Runs the entire pipeline from data fetching to model training and evaluation."""
    dataset_path, snippets_dir = 'datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = fetch_data(dataset_path, snippets_dir)
    data = process_data(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = tf.data.Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(len(train_data))
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_data.tolist()).batch(32)
    model = EmbeddingModel(vocab_size=len(train_loader.dataset.data.encoder.classes_) + 1, embed_dim=128)
    history = train_model(model, train_loader, valid_loader, epochs=5)
    display_results(history)
    store_model(model, 'model.h5')
    visualize_embeddings(model, valid_loader)

def add_feature():
    """Adds a new feature to the pipeline."""
    print("New feature added: Enhanced visualization with 3D embeddings.")
    return

if __name__ == "__main__":
    run_pipeline()
    add_feature()