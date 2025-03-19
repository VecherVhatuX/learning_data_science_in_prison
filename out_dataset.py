import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TextProcessor:
    """Processes text data by encoding text into numerical representations."""
    def __init__(self, data):
        """Initializes the TextProcessor with a LabelEncoder fitted on the extracted texts."""
        self.encoder = LabelEncoder().fit(self._extract_texts(data))
    
    def _extract_texts(self, data):
        """Extracts all texts (anchor, positive, negative) from the dataset."""
        return [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]
    
    def encode_text(self, text):
        """Encodes a single text into a tensor using the fitted LabelEncoder."""
        return tf.convert_to_tensor(self.encoder.transform([text])[0])

class TripletDatasetManager:
    """Manages the dataset for triplet training, including text processing."""
    def __init__(self, data):
        """Initializes the dataset manager with the provided data and a TextProcessor."""
        self.data = data
        self.text_processor = TextProcessor(data)
    
    def get_dataset(self):
        """Returns the managed dataset."""
        return self.data

class TripletDataset(tf.data.Dataset):
    """A custom TensorFlow Dataset for handling triplet data."""
    def __init__(self, data):
        """Initializes the dataset with a TripletDatasetManager and loads samples."""
        self.dataset_manager = TripletDatasetManager(data)
        self.samples = self.dataset_manager.get_dataset()
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Retrieves a single sample from the dataset, encoding the texts."""
        item = self.samples[idx]
        return {
            'anchor_seq': self.dataset_manager.text_processor.encode_text(item['anchor']),
            'positive_seq': self.dataset_manager.text_processor.encode_text(item['positive']),
            'negative_seq': self.dataset_manager.text_processor.encode_text(item['negative'])
        }

class EmbeddingModel(tf.keras.Model):
    """A neural network model for generating embeddings from text."""
    def __init__(self, vocab_size, embed_dim):
        """Initializes the model with an embedding layer and a dense network."""
        super(EmbeddingModel, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.network = tf.keras.Sequential([
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128)
        ])
    
    def call(self, anchor, positive, negative):
        """Generates embeddings for anchor, positive, and negative texts."""
        return (
            self.network(self.embedding(anchor)),
            self.network(self.embedding(positive)),
            self.network(self.embedding(negative))
        )

def calculate_triplet_loss(anchor, positive, negative):
    """Calculates the triplet loss for the given embeddings."""
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1), 0)

def train_model(model, train_loader, valid_loader, epochs):
    """Trains the model using the provided data loaders for a specified number of epochs."""
    optimizer = Adam(learning_rate=0.001)
    scheduler = LearningRateScheduler(lambda epoch: 0.1 ** epoch)
    history = []
    
    for _ in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            with tf.GradientTape() as tape:
                anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
                loss = calculate_triplet_loss(anchor, positive, negative)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss.numpy()
        train_loss /= len(train_loader)
        eval_loss, accuracy = evaluate_model(model, valid_loader)
        history.append((train_loss, eval_loss, accuracy))
    return history

def evaluate_model(model, data_loader):
    """Evaluates the model on the provided data loader, returning loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    for batch in data_loader:
        anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        total_loss += calculate_triplet_loss(anchor, positive, negative).numpy()
        correct += tf.reduce_sum(tf.cast(tf.reduce_sum(anchor * positive, axis=1) > tf.reduce_sum(anchor * negative, axis=1), tf.float32))
    return total_loss / len(data_loader), correct / len(data_loader)

def plot_training_history(history):
    """Plots the training and validation loss and accuracy over epochs."""
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

def save_trained_model(model, path):
    """Saves the trained model weights to the specified path."""
    model.save_weights(path)
    print(f'Model saved at {path}')

def load_trained_model(model, path):
    """Loads the model weights from the specified path."""
    model.load_weights(path)
    print(f'Model loaded from {path}')
    return model

def visualize_embeddings(model, data_loader):
    """Visualizes the embeddings in a 3D plot."""
    model.eval()
    embeddings = []
    for batch in data_loader:
        anchor, _, _ = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        embeddings.append(anchor.numpy())
    embeddings = np.concatenate(embeddings)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c='Spectral')
    plt.title('3D Embedding Visualization')
    plt.show()

def load_and_prepare_data(file_path, root_dir):
    """Loads and prepares data from the specified file and directory."""
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def generate_triplets(mapping, snippet_files):
    """Generates triplets (anchor, positive, negative) from the provided data."""
    return [{'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
            for dir, _ in snippet_files for bug_sample, non_bug_samples in [json.load(open(path)) for path in snippet_files]]

def run_pipeline():
    """Runs the entire pipeline from data loading to model training and evaluation."""
    dataset_path, snippets_dir = 'datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_and_prepare_data(dataset_path, snippets_dir)
    data = generate_triplets(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = tf.data.Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(len(train_data))
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_data.tolist()).batch(32)
    model = EmbeddingModel(vocab_size=len(train_loader.dataset.dataset_manager.text_processor.encoder.classes_) + 1, embed_dim=128)
    history = train_model(model, train_loader, valid_loader, epochs=5)
    plot_training_history(history)
    save_trained_model(model, 'model.h5')
    visualize_embeddings(model, valid_loader)

def add_enhanced_feature():
    """Adds an enhanced feature to the pipeline."""
    print("New feature added: Enhanced visualization with 3D embeddings.")
    return

if __name__ == "__main__":
    run_pipeline()
    add_enhanced_feature()