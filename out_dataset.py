import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def gather_texts(triplet_data):
    return [item[key] for item in triplet_data for key in ['anchor', 'positive', 'negative']]

def convert_to_sequences(tokenizer, item):
    return {
        'anchor_seq': tokenizer.transform([item['anchor']])[0],
        'positive_seq': tokenizer.transform([item['positive']])[0],
        'negative_seq': tokenizer.transform([item['negative']])[0]
    }

class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, triplet_data, max_length, batch_size):
        self.triplet_data = triplet_data
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = LabelEncoder()
        self.tokenizer.fit(gather_texts(triplet_data))

    def __len__(self):
        return int(np.ceil(len(self.triplet_data) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.triplet_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([convert_to_sequences(self.tokenizer, item) for item in batch])

    def shuffle_data(self):
        random.shuffle(self.triplet_data)

    def next_epoch(self):
        self.shuffle_data()

def load_json_data(data_path, folder_path):
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    instance_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    snippets = [
        (folder, os.path.join(folder, 'snippet.json'))
        for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))
    ]
    return instance_map, snippets

def generate_triplets(instance_map, snippets):
    bug_snippets, non_bug_snippets = zip(*(map(lambda snippet_file: json.load(open(snippet_file)), snippets)))
    bug_snippets = [s['snippet'] for s in bug_snippets if s.get('is_bug') and s['snippet']]
    non_bug_snippets = [s['snippet'] for s in non_bug_snippets if not s.get('is_bug') and s['snippet']]
    return create_triplet_structure(instance_map, snippets, bug_snippets, non_bug_snippets)

def create_triplet_structure(instance_map, snippets, bug_snippets, non_bug_snippets):
    return [
        {
            'anchor': instance_map[os.path.basename(folder)],
            'positive': pos_doc,
            'negative': random.choice(non_bug_snippets)
        }
        for folder, _ in snippets
        for pos_doc in bug_snippets
    ]

class TripletModel(models.Model):
    def __init__(self, vocab_size, embed_dim):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.fc = models.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    def call(self, anchor, positive, negative):
        return (self.fc(self.embedding(anchor)), 
                self.fc(self.embedding(positive)), 
                self.fc(self.embedding(negative)))

def triplet_loss(anchor_embeds, positive_embeds, negative_embeds):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) - 
                                      tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train_model(model, train_loader, test_loader, num_epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=triplet_loss)
    train_losses, test_losses, train_accs = [], [], []

    for epoch in range(num_epochs):
        model.fit(train_loader, epochs=1, verbose=1)
        train_loss = model.evaluate(train_loader, verbose=0)
        train_losses.append(train_loss)

        test_loss, acc = evaluate_model(model, test_loader)
        test_losses.append(test_loss)
        train_accs.append(acc)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {acc}')

    return train_losses, test_losses, train_accs

def evaluate_model(model, test_loader):
    test_loss = model.evaluate(test_loader)
    correct_preds = sum(count_correct_predictions(model(batch['anchor_seq']), 
                                                  model(batch['positive_seq']), 
                                                  model(batch['negative_seq'])) 
                        for batch in test_loader)

    acc = correct_preds / len(test_loader.dataset)
    return test_loss, acc

def count_correct_predictions(anchor_out, positive_out, negative_out):
    pos_similarity = tf.reduce_sum(anchor_out * positive_out, axis=1)
    neg_similarity = tf.reduce_sum(anchor_out * negative_out, axis=1)
    return tf.reduce_sum(pos_similarity > neg_similarity).numpy()

def plot_results(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Rate')
    plt.legend()
    plt.show()

def save_model(model, path):
    model.save(path)
    print(f'Model saved to {path}')

def load_model(path):
    model = models.load_model(path)
    model.evaluate()
    print(f'Model loaded from {path}')
    return model

def main():
    data_path = 'datasets/SWE-bench_oracle.npy'
    snippet_path = 'datasets/10_10_after_fix_pytest'
    
    instance_map, snippets = load_json_data(data_path, snippet_path)
    triplets = generate_triplets(instance_map, snippets)
    train_triplets, valid_triplets = np.array_split(np.array(triplets), 2)
    
    train_loader = TripletDataset(train_triplets.tolist(), max_length=512, batch_size=32)
    test_loader = TripletDataset(valid_triplets.tolist(), max_length=512, batch_size=32)
    
    model = TripletModel(vocab_size=len(train_loader.tokenizer.classes_) + 1, embed_dim=128)
    
    train_losses, test_losses, train_accs = train_model(model, train_loader, test_loader, num_epochs=5)
    plot_results(train_losses, test_losses, train_accs, [])

    save_model(model, 'triplet_model.h5')

if __name__ == "__main__":
    main()