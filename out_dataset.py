import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def extract_texts(triplet_data):
    return [entry[key] for entry in triplet_data for key in ['anchor', 'positive', 'negative']]

def sequence_conversion(tokenizer, entry):
    return {
        'anchor_seq': tokenizer.transform([entry['anchor']])[0],
        'positive_seq': tokenizer.transform([entry['positive']])[0],
        'negative_seq': tokenizer.transform([entry['negative']])[0]
    }

class TripletDataLoader(tf.keras.utils.Sequence):
    def __init__(self, triplet_data, max_length, batch_size):
        self.triplet_data = triplet_data
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = LabelEncoder()
        self.tokenizer.fit(extract_texts(triplet_data))

    def __len__(self):
        return -(-len(self.triplet_data) // self.batch_size)

    def __getitem__(self, index):
        batch = self.triplet_data[index * self.batch_size:(index + 1) * self.batch_size]
        return np.array([sequence_conversion(self.tokenizer, entry) for entry in batch])

    def shuffle(self):
        random.shuffle(self.triplet_data)

    def prepare_next_epoch(self):
        self.shuffle()

def read_json_data(file_path, directory_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    instance_mapping = {item['instance_id']: item['problem_statement'] for item in data}
    snippet_files = [
        (folder, os.path.join(folder, 'snippet.json'))
        for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))
    ]
    return instance_mapping, snippet_files

def create_triplet_data(instance_mapping, snippet_files):
    bug_snippets, non_bug_snippets = zip(*(map(lambda file: json.load(open(file)), snippet_files)))
    bug_snippets = [s['snippet'] for s in bug_snippets if s.get('is_bug') and s['snippet']]
    non_bug_snippets = [s['snippet'] for s in non_bug_snippets if not s.get('is_bug') and s['snippet']]
    return build_triplet_structure(instance_mapping, snippet_files, bug_snippets, non_bug_snippets)

def build_triplet_structure(instance_mapping, snippet_files, bug_snippets, non_bug_snippets):
    return [
        {
            'anchor': instance_mapping[os.path.basename(folder)],
            'positive': pos_snippet,
            'negative': random.choice(non_bug_snippets)
        }
        for folder, _ in snippet_files
        for pos_snippet in bug_snippets
    ]

class TripletNN(models.Model):
    def __init__(self, vocab_size, embed_dim):
        super(TripletNN, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embed_dim)
        self.fc_layers = models.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    def call(self, inputs):
        anchor, positive, negative = inputs
        return (self.fc_layers(self.embedding_layer(anchor)), 
                self.fc_layers(self.embedding_layer(positive)), 
                self.fc_layers(self.embedding_layer(negative)))

def compute_triplet_loss(y_true, y_pred):
    anchor_embeds, positive_embeds, negative_embeds = y_pred
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) - 
                                      tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train_triplet_model(model, train_loader, test_loader, epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=compute_triplet_loss)
    training_losses, validation_losses, training_accuracies = [], [], []

    for epoch in range(epochs):
        model.fit(train_loader, epochs=1, verbose=1)
        train_loss = model.evaluate(train_loader, verbose=0)
        training_losses.append(train_loss)

        val_loss, accuracy = validate_model(model, test_loader)
        validation_losses.append(val_loss)
        training_accuracies.append(accuracy)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {accuracy}')

    return training_losses, validation_losses, training_accuracies

def validate_model(model, test_loader):
    test_loss = model.evaluate(test_loader)
    correct_predictions = sum(count_correct_preds(model(batch['anchor_seq']), 
                                                  model(batch['positive_seq']), 
                                                  model(batch['negative_seq'])) 
                              for batch in test_loader)

    accuracy = correct_predictions / len(test_loader.triplet_data)
    return test_loss, accuracy

def count_correct_preds(anchor_out, positive_out, negative_out):
    pos_similarity = tf.reduce_sum(anchor_out * positive_out, axis=1)
    neg_similarity = tf.reduce_sum(anchor_out * negative_out, axis=1)
    return tf.reduce_sum(pos_similarity > neg_similarity).numpy()

def visualize_results(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def save_trained_model(model, file_path):
    model.save(file_path)
    print(f'Model saved at {file_path}')

def load_trained_model(file_path):
    model = models.load_model(file_path)
    model.evaluate()
    print(f'Model loaded from {file_path}')
    return model

def run_main():
    data_file = 'datasets/SWE-bench_oracle.npy'
    snippets_directory = 'datasets/10_10_after_fix_pytest'
    
    instance_mapping, snippet_files = read_json_data(data_file, snippets_directory)
    triplet_data = create_triplet_data(instance_mapping, snippet_files)
    train_data, valid_data = np.array_split(np.array(triplet_data), 2)
    
    train_loader = TripletDataLoader(train_data.tolist(), max_length=512, batch_size=32)
    test_loader = TripletDataLoader(valid_data.tolist(), max_length=512, batch_size=32)
    
    model = TripletNN(vocab_size=len(train_loader.tokenizer.classes_) + 1, embed_dim=128)
    
    training_losses, validation_losses, training_accuracies = train_triplet_model(model, train_loader, test_loader, epochs=5)
    visualize_results(training_losses, validation_losses, training_accuracies, [])

    save_trained_model(model, 'triplet_model.h5')

if __name__ == "__main__":
    run_main()