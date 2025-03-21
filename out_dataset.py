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
    texts = [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]
    encoder = LabelEncoder().fit(texts)
    return lambda text: tf.convert_to_tensor(encoder.transform([text])[0], dtype=tf.int32)

def create_dataset_manager(data):
    text_processor = create_text_processor(data)
    return lambda: data, text_processor

def create_triplet_dataset(data):
    dataset_manager = create_dataset_manager(data)
    return lambda idx: {
        'anchor': dataset_manager()[1](data[idx]['anchor']),
        'positive': dataset_manager()[1](data[idx]['positive']),
        'negative': dataset_manager()[1](data[idx]['negative'])
    }

def create_embedding_model(vocab_size, embed_dim):
    inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)
    embedding = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = layers.Dense(128, activation='relu')(embedding)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(128)(x)
    return tf.keras.Model(inputs, outputs)

def calculate_triplet_loss(anchor, positive, negative):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1), 0))

def train_model(model, train_loader, valid_loader, epochs):
    optimizer = optimizers.Adam(learning_rate=0.001)
    training_history = []
    for epoch in range(epochs):
        total_loss = 0
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
    total_loss = 0
    correct = 0
    for batch in data_loader:
        anchor, positive, negative = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
        total_loss += calculate_triplet_loss(anchor, positive, negative).numpy()
        correct += tf.reduce_sum(tf.cast(tf.reduce_sum(anchor * positive, axis=1) > tf.reduce_sum(anchor * negative, axis=1), tf.float32)).numpy()
    return total_loss / len(data_loader), correct / len(data_loader)

def plot_training_history(history):
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
    model.save_weights(path)
    print(f'Model saved at {path}')

def load_saved_model(model, path):
    model.load_weights(path)
    print(f'Model loaded from {path}')
    return model

def visualize_embedding_space(model, data_loader):
    embeddings = []
    for batch in data_loader:
        anchor, _, _ = model(batch['anchor']), model(batch['positive']), model(batch['negative'])
        embeddings.append(anchor.numpy())
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.concatenate(embeddings)[:, 0], np.concatenate(embeddings)[:, 1], np.concatenate(embeddings)[:, 2], c='Spectral')
    ax.set_title('3D Embedding Visualization')
    plt.show()

def load_dataset(file_path, root_dir):
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def generate_triplets(mapping, snippet_files):
    return [{'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)} for dir, path in snippet_files for bug_sample, non_bug_samples in [json.load(open(path))]]

def run_pipeline():
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
    print("New feature added: Enhanced visualization with 3D embeddings.")

if __name__ == "__main__":
    run_pipeline()
    add_new_feature()