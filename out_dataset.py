import json
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, losses

def extract_texts(data):
    return [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]

def tokenize_data(tokenizer, item):
    return {
        'anchor_seq': tf.convert_to_tensor(tokenizer.transform([item['anchor']])[0]),
        'positive_seq': tf.convert_to_tensor(tokenizer.transform([item['positive']])[0]),
        'negative_seq': tf.convert_to_tensor(tokenizer.transform([item['negative']])[0])
    }

def randomize_samples(samples):
    random.shuffle(samples)
    return samples

def create_triplets(mapping, bug_samples, non_bug_samples):
    return [
        {
            'anchor': mapping[os.path.basename(folder)],
            'positive': bug_sample,
            'negative': random.choice(non_bug_samples)
        }
        for folder, _ in snippet_files
        for bug_sample in bug_samples
    ]

def load_data(file_path, root_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)
    mapping = {entry['instance_id']: entry['problem_statement'] for entry in data}
    snippet_files = [(folder, os.path.join(root_dir, 'snippet.json')) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    return mapping, snippet_files

def prepare_data(mapping, snippet_files):
    return create_triplets(mapping, *zip(*[json.load(open(path)) for path in snippet_files]))

class DataHandler:
    def __init__(self, data):
        self.data = data
        self.tokenizer = LabelEncoder().fit(extract_texts(data))

    def get_data(self):
        return self.data

class CustomDataset(tf.data.Dataset):
    def __init__(self, data):
        self.data = DataHandler(data)
        self.samples = self.data.get_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return tokenize_data(self.data.tokenizer, self.samples[index])

class NeuralModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(NeuralModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.network = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    def call(self, anchor, positive, negative):
        return (
            self.network(self.embedding(anchor)),
            self.network(self.embedding(positive)),
            self.network(self.embedding(negative))
        )

def compute_loss(anchor, positive, negative):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1), 0))

def train(model, train_data, valid_data, epochs):
    optimizer = optimizers.Adam(learning_rate=1e-5)
    history = []
    for _ in range(epochs):
        for batch in train_data:
            with tf.GradientTape() as tape:
                anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
                loss = compute_loss(anchor, positive, negative)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            history.append((loss.numpy(), *evaluate(model, valid_data)))
    return history

def evaluate(model, data):
    loss = sum(compute_loss(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])).numpy() for batch in data)
    correct = sum(count(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])) for batch in data)
    return loss / len(data), correct / len(data.dataset)

def count(anchor, positive, negative):
    return tf.reduce_sum(tf.cast(tf.reduce_sum(anchor * positive, axis=1) > tf.reduce_sum(anchor * negative, axis=1), tf.int32)).numpy()

def plot(history):
    train_loss, val_loss, train_acc = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def save(model, path):
    model.save_weights(path)
    print(f'Model saved at {path}')

def load(model, path):
    model.load_weights(path)
    print(f'Model loaded from {path}')
    return model

def visualize(model, data):
    embeddings = []
    labels = []
    for batch in data:
        anchor, _, _ = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        embeddings.extend(anchor.numpy())
        labels.extend(batch['anchor_seq'].numpy())
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Spectral')
    plt.colorbar()
    plt.title('2D Embedding Visualization')
    plt.show()

def execute():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_directory = 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_data(dataset_path, snippets_directory)
    data = prepare_data(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = tf.data.Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(len(train_data))
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_data.tolist()).batch(32)
    model = NeuralModel(vocab_size=len(train_loader.dataset.data.tokenizer.classes_) + 1, embedding_dim=128)
    history = train(model, train_loader, valid_loader, epochs=5)
    plot(history)
    save(model, 'model.h5')
    visualize(model, valid_loader)

if __name__ == "__main__":
    execute()