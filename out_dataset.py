import json
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, losses

def gather_texts(triplet_data):
    return [text for item in triplet_data for text in (item['anchor'], item['positive'], item['negative'])]

def tokenize_sequences(tokenizer, data_item):
    return {
        'anchor_seq': tf.convert_to_tensor(tokenizer.transform([data_item['anchor']])[0]),
        'positive_seq': tf.convert_to_tensor(tokenizer.transform([data_item['positive']])[0]),
        'negative_seq': tf.convert_to_tensor(tokenizer.transform([data_item['negative']])[0])
    }

def shuffle_data(data_samples):
    random.shuffle(data_samples)
    return data_samples

def generate_triplets(instance_dict, bug_samples, non_bug_samples):
    return [
        {
            'anchor': instance_dict[os.path.basename(folder)],
            'positive': bug_sample,
            'negative': random.choice(non_bug_samples)
        }
        for folder, _ in snippet_files
        for bug_sample in bug_samples
    ]

def load_json_data(file_path, root_dir):
    with open(file_path, 'r') as f:
        json_content = json.load(f)
    instance_dict = {entry['instance_id']: entry['problem_statement'] for entry in json_content}
    snippet_files = [(folder, os.path.join(root_dir, 'snippet.json')) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    return instance_dict, snippet_files

def prepare_triplet_dataset(instance_dict, snippet_files):
    return generate_triplets(instance_dict, *zip(*[json.load(open(path)) for path in snippet_files]))

class TripletData:
    def __init__(self, triplet_data):
        self.triplet_data = triplet_data
        self.tokenizer = LabelEncoder().fit(gather_texts(triplet_data))

    def get_samples(self):
        return self.triplet_data

class TripletDataset(tf.data.Dataset):
    def __init__(self, triplet_data):
        self.data = TripletData(triplet_data)
        self.samples = self.data.get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return tokenize_sequences(self.data.tokenizer, self.samples[index])

class TripletModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim)
        self.dense_network = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    def call(self, anchor, positive, negative):
        return (
            self.dense_network(self.embedding_layer(anchor)),
            self.dense_network(self.embedding_layer(positive)),
            self.dense_network(self.embedding_layer(negative))
        )

def calculate_loss(anchor_embeds, positive_embeds, negative_embeds):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) - tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train_model(model, train_loader, valid_loader, epochs):
    optimizer = optimizers.Adam(learning_rate=1e-5)
    history = []
    for _ in range(epochs):
        for batch in train_loader:
            with tf.GradientTape() as tape:
                anchor_embeds, positive_embeds, negative_embeds = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
                loss = calculate_loss(anchor_embeds, positive_embeds, negative_embeds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            history.append((loss.numpy(), *evaluate_model(model, valid_loader)))
    return history

def evaluate_model(model, valid_loader):
    loss = sum(calculate_loss(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])).numpy() for batch in valid_loader)
    correct = sum(count_matches(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])) for batch in valid_loader)
    return loss / len(valid_loader), correct / len(valid_loader.dataset)

def count_matches(anchor_output, positive_output, negative_output):
    return tf.reduce_sum(tf.cast(tf.reduce_sum(anchor_output * positive_output, axis=1) > tf.reduce_sum(anchor_output * negative_output, axis=1), tf.int32)).numpy()

def plot_results(history):
    train_losses, val_losses, train_accuracies = zip(*history)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def save_model(model, filepath):
    model.save_weights(filepath)
    print(f'Model saved at {filepath}')

def load_model(model, filepath):
    model.load_weights(filepath)
    print(f'Model loaded from {filepath}')
    return model

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_directory = 'datasets/10_10_after_fix_pytest'
    instance_dict, snippet_paths = load_json_data(dataset_path, snippets_directory)
    triplet_data = prepare_triplet_dataset(instance_dict, snippet_paths)
    train_data, valid_data = np.array_split(np.array(triplet_data), 2)
    train_loader = tf.data.Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(len(train_data))
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_data.tolist()).batch(32)
    model = TripletModel(vocab_size=len(train_loader.dataset.data.tokenizer.classes_) + 1, embedding_dim=128)
    history = train_model(model, train_loader, valid_loader, epochs=5)
    plot_results(history)
    save_model(model, 'triplet_model.h5')

if __name__ == "__main__":
    main()