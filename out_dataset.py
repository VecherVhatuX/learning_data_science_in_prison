import json
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.data import Dataset

def collect_texts(data):
    return [text for item in data for text in (item['anchor'], item['positive'], item['negative'])]

def encode_data(encoder, item):
    return {
        'anchor_seq': tf.convert_to_tensor(encoder.transform([item['anchor']])[0]),
        'positive_seq': tf.convert_to_tensor(encoder.transform([item['positive']])[0]),
        'negative_seq': tf.convert_to_tensor(encoder.transform([item['negative']])[0])
    }

def shuffle_data(data):
    random.shuffle(data)
    return data

def create_triplets(mapping, bug_samples, non_bug_samples):
    return [
        {'anchor': mapping[os.path.basename(dir)], 'positive': bug_sample, 'negative': random.choice(non_bug_samples)}
        for dir, _ in snippet_files for bug_sample in bug_samples
    ]

def load_data(file_path, root_dir):
    data = json.load(open(file_path))
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [(dir, os.path.join(root_dir, 'snippet.json')) for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir))]
    return mapping, snippet_files

def prepare_data(mapping, snippet_files):
    return create_triplets(mapping, *zip(*[json.load(open(path)) for path in snippet_files]))

class DataHandler:
    def __init__(self, data):
        self.data = data
        self.encoder = LabelEncoder().fit(collect_texts(data))
    def get_data(self):
        return self.data

class TripletData(Dataset):
    def __init__(self, data):
        self.data = DataHandler(data)
        self.samples = self.data.get_data()
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return encode_data(self.data.encoder, self.samples[idx])

class EmbeddingNet(models.Model):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingNet, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.network = models.Sequential([
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

def train_net(model, train_data, valid_data, epochs):
    optimizer = optimizers.Adam()
    scheduler = optimizers.schedules.ExponentialDecay(0.001, decay_steps=10, decay_rate=0.1)
    history = []
    for _ in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_data:
            with tf.GradientTape() as tape:
                anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
                loss = compute_loss(anchor, positive, negative)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss.numpy()
        scheduler.step()
        train_loss /= len(train_data)
        eval_loss, accuracy = evaluate_net(model, valid_data)
        history.append((train_loss, eval_loss, accuracy))
    return history

def evaluate_net(model, data):
    model.eval()
    total_loss = 0
    correct = 0
    for batch in data:
        anchor, positive, negative = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        total_loss += compute_loss(anchor, positive, negative).numpy()
        correct += count_correct(anchor, positive, negative)
    return total_loss / len(data), correct / len(data.dataset)

def count_correct(anchor, positive, negative):
    return tf.reduce_sum(tf.cast(tf.reduce_sum(anchor * positive, axis=1) > tf.reduce_sum(anchor * negative, axis=1), tf.float32))

def plot_results(history):
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

def save_net(model, path):
    model.save_weights(path)
    print(f'Model saved at {path}')

def load_net(model, path):
    model.load_weights(path)
    print(f'Model loaded from {path}')
    return model

def visualize_embeddings_3d(model, data):
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

def execute_pipeline():
    dataset_path, snippets_dir = 'datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest'
    mapping, snippet_files = load_data(dataset_path, snippets_dir)
    data = prepare_data(mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(data), 2)
    train_loader = Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(1000)
    valid_loader = Dataset.from_tensor_slices(valid_data.tolist()).batch(32)
    model = EmbeddingNet(vocab_size=len(train_loader.dataset.data.encoder.classes_) + 1, embed_dim=128)
    history = train_net(model, train_loader, valid_loader, epochs=5)
    plot_results(history)
    save_net(model, 'model.h5')
    visualize_embeddings_3d(model, valid_loader)

def add_new_feature():
    print("New feature added: Enhanced visualization with 3D embeddings.")
    return

if __name__ == "__main__":
    execute_pipeline()
    add_new_feature()