import os
import random
import json
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

class Config:
    def __init__(self):
        self.instance_id_key = 'instance_id'
        self.max_sequence_length = 512
        self.minibatch_size = 16
        self.negative_samples_per_positive = 3
        self.embedding_size = 128
        self.fully_connected_size = 64
        self.dropout_rate = 0.2
        self.learning_rate_value = 1e-5
        self.max_training_epochs = 5

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

def separate_code_snippets(snippets):
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
            for positive_doc in positive_snippets
            for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def create_triplet_dataset(dataset_path, snippet_folder_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [load_json_data(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
    bug_snippets, non_bug_snippets = zip(*[separate_code_snippets(snippet) for snippet in snippets])
    problem_statements = [instance_id_map.get(os.path.basename(folder_path)) for folder_path in folder_paths]
    triplets = [create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], 3) 
                for i, problem_statement in enumerate(problem_statements)]
    return [item for sublist in triplets for item in sublist]

def encode_text(inputs, tokenizer, max_sequence_length):
    encoding = tokenizer.encode_plus(
        inputs, 
        max_length=max_sequence_length, 
        padding='max_length', 
        truncation=True, 
        return_attention_mask=True, 
        return_tensors='tf'
    )
    return encoding['input_ids'][:, 0, :]

def triplet_loss(anchor, positive, negative):
    return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1) + 1.0, 0.0))

def batch_data(data, minibatch_size):
    random.shuffle(data)
    for i in range(0, len(data), minibatch_size):
        batch = data[i:i + minibatch_size]
        anchors = np.stack([item['anchor'] for item in batch])
        positives = np.stack([item['positive'] for item in batch])
        negatives = np.stack([item['negative'] for item in batch])
        yield tf.data.Dataset.from_tensor_slices((anchors, positives, negatives)).batch(minibatch_size)

class TripletModel:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model = tf.keras.models.Sequential([
            layers.Dense(config.embedding_size, activation='relu', name='embedding_layer'),
            layers.Dropout(config.dropout_rate, name='dropout_layer_1'),
            layers.Dense(config.fully_connected_size, activation='relu', name='fully_connected_layer'),
            layers.Dropout(config.dropout_rate, name='dropout_layer_2')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate_value))

    def create_dataset(self, triplets):
        return list(map(lambda triplet: {
            'anchor': encode_text(triplet['anchor'], self.tokenizer, self.config.max_sequence_length),
            'positive': encode_text(triplet['positive'], self.tokenizer, self.config.max_sequence_length),
            'negative': encode_text(triplet['negative'], self.tokenizer, self.config.max_sequence_length)
        }, triplets))

    def train_step(self, data):
        anchor, positive, negative = data
        with tf.GradientTape() as tape:
            anchor_embedding = self.model(anchor, training=True)
            positive_embedding = self.model(positive, training=True)
            negative_embedding = self.model(negative, training=True)
            loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    def train(self, dataset):
        total_loss = 0
        for batch in batch_data(dataset, self.config.minibatch_size):
            loss = self.train_step(batch)[['loss']]
            total_loss += loss
        return total_loss / len(dataset)

    def evaluate(self, dataset):
        total_loss = 0
        for batch in batch_data(dataset, self.config.minibatch_size):
            anchor, positive, negative = batch
            anchor_embedding = self.model(anchor, training=False)
            positive_embedding = self.model(positive, training=False)
            negative_embedding = self.model(negative, training=False)
            loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            total_loss += loss
        return total_loss / len(dataset)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

class TripletTrainer:
    def __init__(self, model, dataset_path, snippet_folder_path, tokenizer):
        self.model = model
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.tokenizer = tokenizer

    def train(self):
        triplets = create_triplet_dataset(self.dataset_path, self.snippet_folder_path)
        random.shuffle(triplets)
        train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
        train_dataset = self.model.create_dataset(train_triplets)
        test_dataset = self.model.create_dataset(test_triplets)
        history = {'loss': [], 'val_loss': []}
        for epoch in range(self.model.config.max_training_epochs):
            loss = self.model.train(train_dataset)
            val_loss = self.model.evaluate(test_dataset)
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
            self.model.save(f'triplet_model_{epoch+1}.h5')
            print(f'Model saved at triplet_model_{epoch+1}.h5')
        return history

def plot_training_history(history):
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    config = Config()
    model = TripletModel(config, tokenizer)
    trainer = TripletTrainer(model, dataset_path, snippet_folder_path, tokenizer)
    history = trainer.train()
    plot_training_history(history)

if __name__ == "__main__":
    main()