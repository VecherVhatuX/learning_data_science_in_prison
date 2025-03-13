import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def get_texts(triplet_data):
    return [item for entry in triplet_data for item in (entry['anchor'], entry['positive'], entry['negative'])]

def convert_to_sequences(tokenizer, entry):
    return {
        'anchor_seq': tokenizer.transform([entry['anchor']])[0],
        'positive_seq': tokenizer.transform([entry['positive']])[0],
        'negative_seq': tokenizer.transform([entry['negative']])[0]
    }

def shuffle_samples(data):
    random.shuffle(data)
    return data

def create_triplet_structure(instance_mapping, bug_snips, non_bug_snips):
    return [
        {
            'anchor': instance_mapping[os.path.basename(folder)],
            'positive': bug_snip,
            'negative': random.choice(non_bug_snips)
        }
        for folder, _ in snippet_paths
        for bug_snip in bug_snips
    ]

def load_json(file_path, base_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)
    mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_paths = [
        (folder, os.path.join(base_dir, 'snippet.json'))
        for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))
    ]
    return mapping, snippet_paths

def generate_triplet_data(instance_mapping, snippet_paths):
    bug_snips, non_bug_snips = zip(*(map(lambda path: json.load(open(path)), snippet_paths)))
    bug_snips = [s['snippet'] for s in bug_snips if s.get('is_bug') and s['snippet']]
    non_bug_snips = [s['snippet'] for s in non_bug_snips if not s.get('is_bug') and s['snippet']]
    return create_triplet_structure(instance_mapping, bug_snips, non_bug_snips)

class TripletDataset:
    def __init__(self, triplet_data):
        self.triplet_data = triplet_data
        self.tokenizer = LabelEncoder()
        self.tokenizer.fit(get_texts(triplet_data))

    def get_samples(self):
        return self.triplet_data

class TripletDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, triplet_data, batch_size):
        self.dataset = TripletDataset(triplet_data)
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.dataset.get_samples()) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch = self.dataset.get_samples()[idx * self.batch_size : (idx + 1) * self.batch_size]
        return np.array([convert_to_sequences(self.dataset.tokenizer, entry) for entry in batch])

    def on_epoch_end(self):
        self.dataset.triplet_data = shuffle_samples(self.dataset.triplet_data)

class TripletNetwork(models.Model):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.dense_layers = models.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    def call(self, inputs):
        anchor, positive, negative = inputs
        return (self.dense_layers(self.embedding(anchor)),
                self.dense_layers(self.embedding(positive)),
                self.dense_layers(self.embedding(negative)))

def triplet_loss(y_true, y_pred):
    anchor_embeds, positive_embeds, negative_embeds = y_pred
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) -
                                      tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train_model(model, train_gen, valid_gen, num_epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=triplet_loss)

    history = []
    for epoch in range(num_epochs):
        model.fit(train_gen, epochs=1, verbose=1)
        train_loss = model.evaluate(train_gen, verbose=0)
        val_loss, accuracy = evaluate_model(model, valid_gen)
        history.append((train_loss, val_loss, accuracy))
    return history

def evaluate_model(model, valid_gen):
    loss = model.evaluate(valid_gen)
    correct_preds = sum(count_correct_predictions(model(batch['anchor_seq']),
                                                  model(batch['positive_seq']),
                                                  model(batch['negative_seq'])) 
                        for batch in valid_gen)

    accuracy = correct_preds / len(valid_gen.dataset.get_samples())
    return loss, accuracy

def count_correct_predictions(anchor_output, positive_output, negative_output):
    positive_similarity = tf.reduce_sum(anchor_output * positive_output, axis=1)
    negative_similarity = tf.reduce_sum(anchor_output * negative_output, axis=1)
    return tf.reduce_sum(positive_similarity > negative_similarity).numpy()

def plot_results(history):
    train_losses, val_losses, train_accuracies = zip(*history)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Throughout Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.title('Accuracy Throughout Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def save_model(model, path):
    model.save(path)
    print(f'Model saved at {path}')

def load_model(path):
    model = models.load_model(path)
    print(f'Model loaded from {path}')
    return model

def main():
    data_path = 'datasets/SWE-bench_oracle.npy'
    snippets_path = 'datasets/10_10_after_fix_pytest'
    
    instance_map, snippet_files = load_json(data_path, snippets_path)
    triplet_data = generate_triplet_data(instance_map, snippet_files)
    train_data, valid_data = np.array_split(np.array(triplet_data), 2)
    
    train_gen = TripletDataGenerator(train_data.tolist(), batch_size=32)
    valid_gen = TripletDataGenerator(valid_data.tolist(), batch_size=32)
    
    model = TripletNetwork(vocab_size=len(train_gen.dataset.tokenizer.classes_) + 1, embedding_dim=128)
    
    history = train_model(model, train_gen, valid_gen, num_epochs=5)
    plot_results(history)

    save_model(model, 'triplet_model.h5')

if __name__ == "__main__":
    main()