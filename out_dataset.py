import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer
from typing import List, Tuple

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
    def __init__(self):
        super(TripletModel, self).__init__()
        self.distilbert = tf.keras.layers.Lambda(lambda x: tf.keras.applications.distilbert_encode(x, training=False))
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(Config.EMBEDDING_DIM),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(Config.DROPOUT),
            tf.keras.layers.Dense(Config.FC_DIM),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(Config.DROPOUT)
        ])

    def call(self, inputs):
        anchor, positive, negative = inputs
        anchor_output = self.distilbert([anchor['input_ids'], anchor['attention_mask']])
        positive_output = self.distilbert([positive['input_ids'], positive['attention_mask']])
        negative_output = self.distilbert([negative['input_ids'], negative['attention_mask']])
        anchor_embedding = self.embedding(anchor_output)
        positive_embedding = self.embedding(positive_output)
        negative_embedding = self.embedding(negative_output)
        return anchor_embedding, positive_embedding, negative_embedding

    def triplet_loss(self, anchor, positive, negative):
        loss = tf.maximum(tf.zeros_like(anchor), tf.reduce_sum((anchor - positive) ** 2) - tf.reduce_sum((anchor - negative) ** 2) + 1.0)
        return tf.reduce_mean(loss)

class TripletDataset:
    def __init__(self, triplets, tokenizer, batch_size):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.triplets) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_triplets = self.triplets[start:end]
        anchor_encodings = []
        positive_encodings = []
        negative_encodings = []
        for triplet in batch_triplets:
            anchor_encoding = self.tokenizer.encode_plus(
                triplet['anchor'],
                max_length=Config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            positive_encoding = self.tokenizer.encode_plus(
                triplet['positive'],
                max_length=Config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            negative_encoding = self.tokenizer.encode_plus(
                triplet['negative'],
                max_length=Config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            anchor_encodings.append({
                'input_ids': anchor_encoding['input_ids'].flatten(),
                'attention_mask': anchor_encoding['attention_mask'].flatten()
            })
            positive_encodings.append({
                'input_ids': positive_encoding['input_ids'].flatten(),
                'attention_mask': positive_encoding['attention_mask'].flatten()
            })
            negative_encodings.append({
                'input_ids': negative_encoding['input_ids'].flatten(),
                'attention_mask': negative_encoding['attention_mask'].flatten()
            })
        return {
            'anchor': np.array(anchor_encodings),
            'positive': np.array(positive_encodings),
            'negative': np.array(negative_encodings)
        }

    def on_epoch_end(self):
        random.shuffle(self.triplets)

def load_json_file(file_path: str) -> List:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

def separate_snippets(snippets: List) -> Tuple[List, List]:
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement: str, positive_snippets: List, negative_snippets: List, num_negatives_per_positive: int) -> List:
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
            for positive_doc in positive_snippets
            for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def create_triplet_dataset(dataset_path: str, snippet_folder_path: str) -> List:
    dataset = np.load(dataset_path, allow_pickle=True)
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
    bug_snippets, non_bug_snippets = zip(*[separate_snippets(snippet) for snippet in snippets])
    problem_statements = [instance_id_map.get(os.path.basename(folder_path)) for folder_path in folder_paths]
    triplets = [create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], Config.NUM_NEGATIVES_PER_POSITIVE) 
                for i, problem_statement in enumerate(problem_statements)]
    return [item for sublist in triplets for item in sublist]

def load_data(dataset_path: str, snippet_folder_path: str, tokenizer):
    triplets = create_triplet_dataset(dataset_path, snippet_folder_path)
    random.shuffle(triplets)
    train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
    train_dataset = TripletDataset(train_triplets, tokenizer, batch_size=Config.BATCH_SIZE)
    test_dataset = TripletDataset(test_triplets, tokenizer, batch_size=Config.BATCH_SIZE)
    return train_dataset, test_dataset

def train(model, dataset, optimizer):
    total_loss = 0
    for batch in dataset:
        anchor, positive, negative = model([batch['anchor'], batch['positive'], batch['negative']])
        loss = model.triplet_loss(anchor, positive, negative)
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss
    return total_loss / len(dataset)

def evaluate(model, dataset):
    total_loss = 0
    for batch in dataset:
        anchor, positive, negative = model([batch['anchor'], batch['positive'], batch['negative']])
        loss = model.triplet_loss(anchor, positive, negative)
        total_loss += loss
    return total_loss / len(dataset)

def save_model(model, path):
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset, test_dataset = load_data(dataset_path, snippet_folder_path, tokenizer)
    model = TripletModel()
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model_path = 'triplet_model.h5'
    for epoch in range(Config.MAX_EPOCHS):
        train_dataset.on_epoch_end()
        loss = train(model, train_dataset, optimizer)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        save_model(model, model_path)
        print(f'Model saved at {model_path}')
    model = load_model(model_path)
    loss = evaluate(model, test_dataset)
    print(f'Test Loss: {loss:.4f}')

if __name__ == "__main__":
    main()