import os
import random
import json
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

class Config:
    INSTANCE_ID_FIELD = 'instance_id'
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    NUM_NEGATIVES_PER_POSITIVE = 3
    EMBEDDING_DIM = 128
    FC_DIM = 64
    DROPOUT = 0.2
    LEARNING_RATE = 1e-5
    MAX_EPOCHS = 5

class TripletModel(tf.keras.Model):
    def __init__(self, tokenizer):
        super(TripletModel, self).__init__()
        self.tokenizer = tokenizer
        self.embedding = layers.Dense(Config.EMBEDDING_DIM, activation='relu')
        self.dropout1 = layers.Dropout(Config.DROPOUT)
        self.fc = layers.Dense(Config.FC_DIM, activation='relu')
        self.dropout2 = layers.Dropout(Config.DROPOUT)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)

    def encode(self, inputs):
        encoding = self.tokenizer.encode_plus(
            inputs, 
            max_length=Config.MAX_LENGTH, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='tf'
        )
        return encoding['input_ids'][:, 0, :]

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x = self.fc(x)
        x = self.dropout2(x)
        return x

    def loss(self, anchor, positive, negative):
        return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1) + 1.0, 0.0))

    def train_step(self, data):
        anchor, positive, negative = data
        with tf.GradientTape() as tape:
            anchor_embedding = self(anchor, training=True)
            positive_embedding = self(positive, training=True)
            negative_embedding = self(negative, training=True)
            loss = self.loss(anchor_embedding, positive_embedding, negative_embedding)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

class TripletDataset:
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.batch_size = Config.BATCH_SIZE
        self.indices = list(range(len(self.triplets)))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        triplet = self.triplets[index]
        anchor_encoding = self.tokenizer.encode_plus(
            triplet['anchor'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        positive_encoding = self.tokenizer.encode_plus(
            triplet['positive'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        negative_encoding = self.tokenizer.encode_plus(
            triplet['negative'],
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        return {
            'anchor': anchor_encoding['input_ids'][:, 0, :],
            'positive': positive_encoding['input_ids'][:, 0, :],
            'negative': negative_encoding['input_ids'][:, 0, :]
        }

    def shuffle(self):
        random.shuffle(self.indices)

    def batch(self):
        self.shuffle()
        for i in range(0, len(self), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.__getitem__(index) for index in batch_indices]
            anchors = np.stack([item['anchor'] for item in batch])
            positives = np.stack([item['positive'] for item in batch])
            negatives = np.stack([item['negative'] for item in batch])
            yield tf.data.Dataset.from_tensor_slices((anchors, positives, negatives)).batch(self.batch_size)

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

def separate_snippets(snippets):
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
    snippets = [load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
    bug_snippets, non_bug_snippets = zip(*[separate_snippets(snippet) for snippet in snippets])
    problem_statements = [instance_id_map.get(os.path.basename(folder_path)) for folder_path in folder_paths]
    triplets = [create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], Config.NUM_NEGATIVES_PER_POSITIVE) 
                for i, problem_statement in enumerate(problem_statements)]
    return [item for sublist in triplets for item in sublist]

def load_data(dataset_path, snippet_folder_path, tokenizer):
    triplets = create_triplet_dataset(dataset_path, snippet_folder_path)
    random.shuffle(triplets)
    train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
    train_dataset = TripletDataset(train_triplets, tokenizer)
    test_dataset = TripletDataset(test_triplets, tokenizer)
    return train_dataset, test_dataset

def train(model, dataset):
    total_loss = 0
    for batch in dataset.batch():
        loss = model.train_step(batch)
        total_loss += loss['loss']
    return total_loss / len(dataset)

def evaluate(model, dataset):
    total_loss = 0
    for batch in dataset.batch():
        anchor, positive, negative = batch
        anchor_embedding = model(anchor, training=False)
        positive_embedding = model(positive, training=False)
        negative_embedding = model(negative, training=False)
        loss = model.loss(anchor_embedding, positive_embedding, negative_embedding)
        total_loss += loss
    return total_loss / len(dataset)

def save_model(model, path):
    model.save_weights(path)

def load_model(path, tokenizer):
    model = TripletModel(tokenizer)
    model.load_weights(path)
    return model

def plot_history(history):
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset, test_dataset = load_data(dataset_path, snippet_folder_path, tokenizer)
    model = TripletModel(tokenizer)
    model_path = 'triplet_model.h5'
    history = {'loss': [], 'val_loss': []}
    for epoch in range(Config.MAX_EPOCHS):
        loss = train(model, train_dataset)
        val_loss = evaluate(model, test_dataset)
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        save_model(model, model_path)
        print(f'Model saved at {model_path}')
    plot_history(history)

if __name__ == "__main__":
    main()