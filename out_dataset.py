import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_text_processor(data):
    """
    Creates a text processor function that encodes text data into integer tensors.
    
    Args:
        data (list): List of dictionaries containing 'anchor', 'positive', and 'negative' text samples.
    
    Returns:
        function: A lambda function that takes a text string and returns its encoded tensor representation.
    """
    texts = [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]
    encoder = LabelEncoder().fit(texts)
    return lambda text: tf.convert_to_tensor(encoder.transform([text])[0], dtype=tf.int32)

def create_dataset_manager(data):
    """
    Creates a dataset manager that returns the dataset and a text processor.
    
    Args:
        data (list): List of dictionaries containing 'anchor', 'positive', and 'negative' text samples.
    
    Returns:
        function: A lambda function that returns the dataset and the text processor.
    """
    text_processor = create_text_processor(data)
    return lambda: data, text_processor

def create_triplet_dataset(data):
    """
    Creates a triplet dataset function that processes data into anchor, positive, and negative samples.
    
    Args:
        data (list): List of dictionaries containing 'anchor', 'positive', and 'negative' text samples.
    
    Returns:
        function: A lambda function that takes an index and returns a dictionary of processed anchor, positive, and negative samples.
    """
    dataset_manager = create_dataset_manager(data)
    return lambda idx: {
        'anchor': dataset_manager[1](data[idx]['anchor']),
        'positive': dataset_manager[1](data[idx]['positive']),
        'negative': dataset_manager[1](data[idx]['negative'])
    }

def create_embedding_model(vocab_size, embed_dim):
    """
    Creates an embedding model for text data.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embedding vectors.
    
    Returns:
        tf.keras.Model: A Keras model that takes integer-encoded text and outputs embeddings.
    """
    inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)
    embedding = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = layers.Dense(128, activation='relu')(embedding)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(128)(x)
    return tf.keras.Model(inputs, outputs)

def calculate_triplet_loss(anchor, positive, negative):
    """
    Calculates the triplet loss for a batch of anchor, positive, and negative samples.
    
    Args:
        anchor (tf.Tensor): Embeddings of anchor samples.
        positive (tf.Tensor): Embeddings of positive samples.
        negative (tf.Tensor): Embeddings of negative samples.
    
    Returns:
        tf.Tensor: The computed triplet loss.
    """
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1), 0))

def train_model(model, train_loader, valid_loader, epochs):
    """
    Trains the embedding model using triplet loss.
    
    Args:
        model (tf.keras.Model): The embedding model to train.
        train_loader (tf.data.Dataset): Training data loader.
        valid_loader (tf.data.Dataset): Validation data loader.
        epochs (int): Number of epochs to train the model.
    
    Returns:
        list: A list of tuples containing training and validation loss and accuracy for each epoch.
    """
    optimizer = optimizers.Adam(learning_rate=0.001)
    training_history = []
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in train_loader:
            with tf.GradientTape() as tape:
                anchor, positive, negative = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
                loss = calculate_triplet_loss(anchor, positive, negative)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        training_history.append((total_loss / len(train_loader), *evaluate_model(model, valid_loader)))
    return training_history

def evaluate_model(model, data_loader):
    """
    Evaluates the model on a given dataset.
    
    Args:
        model (tf.keras.Model): The embedding model to evaluate.
        data_loader (tf.data.Dataset): Data loader for evaluation.
    
    Returns:
        tuple: A tuple containing the average loss and accuracy over the dataset.
    """
    total_loss = 0
    correct = 0
    model.eval()
    for batch in data_loader:
        anchor, positive, negative = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
        total_loss += calculate_triplet_loss(anchor, positive, negative).numpy()
        correct += tf.reduce_sum(tf.cast(tf.reduce_sum(anchor * positive, axis=1) > tf.reduce_sum(anchor * negative, axis=1), tf.float32)).numpy()
    return total_loss / len(data_loader), correct / len(data_loader)

def plot_training_history(history):
    """
    Plots the training and validation loss and accuracy over epochs.
    
    Args:
        history (list): List of tuples containing training and validation loss and accuracy for each epoch.
    """
    fig = plt.figure(figsize=(10, 5))
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
    """
    Saves the trained model weights to a specified path.
    
    Args:
        model (tf.keras.Model): The trained model to save.
        path (str): Path to save the model weights.
    """
    model.save_weights(path)
    print(f'Model saved at {path}')

def load_saved_model(model, path):
    """
    Loads the model weights from a specified path.
    
    Args:
        model (tf.keras.Model): The model to load weights into.
        path (str): Path to load the model weights from.
    
    Returns:
        tf.keras.Model: The model with loaded weights.
    """
    model.load_weights(path)
    print(f'Model loaded from {path}')
    return model

def visualize_embedding_space(model, data_loader):
    """
    Visualizes the embedding space in 3D.
    
    Args:
        model (tf.keras.Model): The embedding model.
        data_loader (tf.data.Dataset): Data loader to generate embeddings for visualization.
    """
    embeddings = []
    model.eval()
    for batch in data_loader:
        anchor, _, _ = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
        embeddings.append(anchor.numpy())
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.concatenate(embeddings)[:, 0], np.concatenate(embeddings)[:, 1], np.concatenate(embeddings)[:, 2], c='Spectral')
    ax.set_title('3D Embedding Visualization')
    plt.show()

def load_dataset(file_path, root_dir):
    """
    Loads the dataset from a JSON file and maps instance IDs to problem statements.
    
    Args:
        file_path (str): Path to the JSON file containing the dataset.
        root_dir (str): Root directory containing snippet files.
    
    Returns:
        tuple: A tuple containing a mapping of instance IDs to problem statements and a list of snippet files.
    """
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def generate_triplets(mapping, snippet_files):
    """
    Generates triplets of anchor, positive, and negative samples from the dataset.
    
    Args:
        mapping (dict): Mapping of instance IDs to problem statements.
        snippet_files (list): List of snippet files.
    
    Returns:
        list: A list of dictionaries containing anchor, positive, and negative samples.
    """
    return [{'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)} for dir, path in snippet_files for bug_sample, non_bug_samples in [json.load(open(path))]]

def run_pipeline():
    """
    Runs the entire pipeline including dataset loading, model training, evaluation, and visualization.
    """
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_dir = 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_dataset(dataset_path, snippets_dir)
    data = generate_triplets(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = tf.data.Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(len(train_data))
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_data.tolist()).batch(32)
    model = create_embedding_model(vocab_size=len(train_loader.element_spec['anchor'].shape[0]) + 1, embed_dim=128)
    history = train_model(model, train_loader, valid_loader, epochs=5)
    plot_training_history(history)
    save_trained_model(model, 'model.h5')
    visualize_embedding_space(model, valid_loader)

def add_new_feature():
    """
    Adds a new feature to the pipeline, such as enhanced visualization.
    """
    print("New feature added: Enhanced visualization with 3D embeddings.")

if __name__ == "__main__":
    run_pipeline()
    add_new_feature()