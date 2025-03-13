import json
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, losses

def collect_texts(triplet_info):
    return [text for entry in triplet_info for text in (entry['anchor'], entry['positive'], entry['negative'])]

def encode_sequences(encoder, data_entry):
    return {
        'anchor_seq': tf.convert_to_tensor(encoder.transform([data_entry['anchor']])[0]),
        'positive_seq': tf.convert_to_tensor(encoder.transform([data_entry['positive']])[0]),
        'negative_seq': tf.convert_to_tensor(encoder.transform([data_entry['negative']])[0])
    }

def randomize_data(data_entries):
    random.shuffle(data_entries)
    return data_entries

def create_triplets(instance_map, bug_entries, non_bug_entries):
    return [
        {
            'anchor': instance_map[os.path.basename(dir)],
            'positive': bug_entry,
            'negative': random.choice(non_bug_entries)
        }
        for dir, _ in snippet_paths
        for bug_entry in bug_entries
    ]

def load_json(file_path, base_dir):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    instance_map = {item['instance_id']: item['problem_statement'] for item in json_data}
    snippet_paths = [(dir, os.path.join(base_dir, 'snippet.json')) for dir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dir))]
    return instance_map, snippet_paths

def prepare_dataset(instance_map, snippet_paths):
    bug_entries, non_bug_entries = zip(*[json.load(open(path)) for path in snippet_paths])
    return create_triplets(instance_map, bug_entries, non_bug_entries)

class TripletData:
    def __init__(self, triplet_info):
        self.triplet_info = triplet_info
        self.encoder = LabelEncoder().fit(collect_texts(triplet_info))

    def get_entries(self):
        return self.triplet_info

class TripletDataset(tf.data.Dataset):
    def __init__(self, triplet_info):
        self.data = TripletData(triplet_info)
        self.entries = self.data.get_entries()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return encode_sequences(self.data.encoder, self.entries[idx])

class TripletModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.network = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    def call(self, anchor, positive, negative):
        anchor_embed = self.network(self.embedding(anchor))
        positive_embed = self.network(self.embedding(positive))
        negative_embed = self.network(self.embedding(negative))
        return anchor_embed, positive_embed, negative_embed

def compute_loss(anchor_embeds, positive_embeds, negative_embeds):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) - tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train(model, train_loader, valid_loader, epochs):
    optimizer = optimizers.Adam(learning_rate=1e-5)
    history = []
    for _ in range(epochs):
        for batch in train_loader:
            with tf.GradientTape() as tape:
                anchor_embeds, positive_embeds, negative_embeds = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
                loss = compute_loss(anchor_embeds, positive_embeds, negative_embeds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        history.append((loss.numpy(), *evaluate(model, valid_loader)))
    return history

def evaluate(model, valid_loader):
    loss, correct = 0, 0
    for batch in valid_loader:
        anchor_embeds, positive_embeds, negative_embeds = model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])
        loss += compute_loss(anchor_embeds, positive_embeds, negative_embeds).numpy()
        correct += count_correct(anchor_embeds, positive_embeds, negative_embeds)
    return loss / len(valid_loader), correct / len(valid_loader.dataset)

def count_correct(anchor_output, positive_output, negative_output):
    return tf.reduce_sum(tf.cast(tf.reduce_sum(anchor_output * positive_output, axis=1) > tf.reduce_sum(anchor_output * negative_output, axis=1), tf.int32)).numpy()

def plot_history(history):
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

def save_model_weights(model, path):
    model.save_weights(path)
    print(f'Model saved at {path}')

def load_model_weights(model, path):
    model.load_weights(path)
    print(f'Model loaded from {path}')
    return model

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_dir = 'datasets/10_10_after_fix_pytest'
    instance_map, snippet_paths = load_json(dataset_path, snippets_dir)
    triplet_info = prepare_dataset(instance_map, snippet_paths)
    train_data, valid_data = np.array_split(np.array(triplet_info), 2)
    train_loader = tf.data.Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(len(train_data))
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_data.tolist()).batch(32)
    model = TripletModel(vocab_size=len(train_loader.dataset.data.encoder.classes_) + 1, embed_dim=128)
    history = train(model, train_loader, valid_loader, epochs=5)
    plot_history(history)
    save_model_weights(model, 'triplet_model.h5')

if __name__ == "__main__":
    main()