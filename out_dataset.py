import json
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers

def extract_texts(triplet_entries):
    return [text for entry in triplet_entries for text in (entry['anchor'], entry['positive'], entry['negative'])]

def sequence_conversion(tokenizer, entry):
    return {
        'anchor_seq': tf.convert_to_tensor(tokenizer.transform([entry['anchor']])[0]),
        'positive_seq': tf.convert_to_tensor(tokenizer.transform([entry['positive']])[0]),
        'negative_seq': tf.convert_to_tensor(tokenizer.transform([entry['negative']])[0])
    }

def randomize_samples(sample_data):
    random.shuffle(sample_data)
    return sample_data

def build_triplet_data(instance_map, bug_snippets, non_bug_snippets):
    return [
        {
            'anchor': instance_map[os.path.basename(directory)],
            'positive': bug_snippet,
            'negative': random.choice(non_bug_snippets)
        }
        for directory, _ in snippet_files
        for bug_snippet in bug_snippets
    ]

def read_json(file, base_directory):
    with open(file, 'r') as f:
        json_data = json.load(f)
    instance_mapping = {item['instance_id']: item['problem_statement'] for item in json_data}
    snippet_files = [
        (folder, os.path.join(base_directory, 'snippet.json'))
        for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))
    ]
    return instance_mapping, snippet_files

def create_triplet_dataset(instance_map, snippet_files):
    bug_snips, non_bug_snips = zip(*(map(lambda path: json.load(open(path)), snippet_files)))
    bug_snips = [s['snippet'] for s in bug_snips if s.get('is_bug') and s['snippet']]
    non_bug_snips = [s['snippet'] for s in non_bug_snips if not s.get('is_bug') and s['snippet']]
    return build_triplet_data(instance_map, bug_snips, non_bug_snips)

class TripletData:
    def __init__(self, triplet_entries):
        self.triplet_entries = triplet_entries
        self.tokenizer = LabelEncoder()
        self.tokenizer.fit(extract_texts(triplet_entries))

    def fetch_samples(self):
        return self.triplet_entries

class TripletDataset(tf.data.Dataset):
    def __init__(self, triplet_entries):
        self.data = TripletData(triplet_entries)

    def __len__(self):
        return len(self.data.fetch_samples())

    def __getitem__(self, index):
        entry = self.data.fetch_samples()[index]
        return sequence_conversion(self.data.tokenizer, entry)

class TripletModel(models.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim)
        self.dense_network = models.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    def call(self, anchor, positive, negative):
        anchor_embed = self.dense_network(self.embedding_layer(anchor))
        positive_embed = self.dense_network(self.embedding_layer(positive))
        negative_embed = self.dense_network(self.embedding_layer(negative))
        return anchor_embed, positive_embed, negative_embed

def compute_triplet_loss(anchor_embeds, positive_embeds, negative_embeds):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) -
                          tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train_triplet_model(model, train_loader, valid_loader, epochs):
    optimizer = optimizers.Adam(learning_rate=1e-5)
    training_history = []
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            with tf.GradientTape() as tape:
                anchor, positive, negative = batch['anchor_seq'], batch['positive_seq'], batch['negative_seq']
                anchor_embeds, positive_embeds, negative_embeds = model(anchor, positive, negative)
                loss = compute_triplet_loss(anchor_embeds, positive_embeds, negative_embeds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss = loss.numpy()
        validation_loss, accuracy = validate_model(model, valid_loader)
        training_history.append((train_loss, validation_loss, accuracy))
    
    return training_history

def validate_model(model, valid_loader):
    model.eval()
    loss = 0
    correct_predictions = 0
    for batch in valid_loader:
        anchor, positive, negative = batch['anchor_seq'], batch['positive_seq'], batch['negative_seq']
        anchor_embeds, positive_embeds, negative_embeds = model(anchor, positive, negative)
        loss += compute_triplet_loss(anchor_embeds, positive_embeds, negative_embeds).numpy()
        correct_predictions += count_matches(anchor_embeds, positive_embeds, negative_embeds)
    accuracy = correct_predictions / len(valid_loader.dataset)
    return loss / len(valid_loader), accuracy

def count_matches(anchor_output, positive_output, negative_output):
    positive_similarity = tf.reduce_sum(anchor_output * positive_output, axis=1)
    negative_similarity = tf.reduce_sum(anchor_output * negative_output, axis=1)
    return tf.reduce_sum(tf.cast(positive_similarity > negative_similarity, tf.int32)).numpy()

def display_results(history):
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

def save_triplet_model(model, filepath):
    model.save_weights(filepath)
    print(f'Model saved at {filepath}')

def load_triplet_model(model, filepath):
    model.load_weights(filepath)
    print(f'Model loaded from {filepath}')
    return model

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippets_directory = 'datasets/10_10_after_fix_pytest'
    
    instance_mapping, snippet_paths = read_json(dataset_path, snippets_directory)
    triplet_entries = create_triplet_dataset(instance_mapping, snippet_paths)
    train_entries, valid_entries = np.array_split(np.array(triplet_entries), 2)
    
    train_loader = tf.data.Dataset.from_tensor_slices(train_entries.tolist()).batch(32).shuffle(True)
    valid_loader = tf.data.Dataset.from_tensor_slices(valid_entries.tolist()).batch(32).shuffle(False)
    
    model = TripletModel(vocab_size=len(train_loader.dataset.data.tokenizer.classes_) + 1, embedding_dim=128)
    
    history = train_triplet_model(model, train_loader, valid_loader, epochs=5)
    display_results(history)

    save_triplet_model(model, 'triplet_model.h5')

if __name__ == "__main__":
    main()