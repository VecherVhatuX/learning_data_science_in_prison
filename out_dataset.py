import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import pickle
import json

# Constants
INSTANCE_ID_FIELD = 'instance_id'
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_NEGATIVES_PER_POSITIVE = 3
EMBEDDING_DIM = 128
FC_DIM = 64
DROPOUT = 0.2
LEARNING_RATE = 1e-5
MAX_EPOCHS = 5

def load_dataset(dataset_path):
    try:
        return np.load(dataset_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return []

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
        return []

def get_subfolder_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

def separate_snippets(snippets):
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
            for positive_doc in positive_snippets
            for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

def encode_triplet(triplet):
    vectorizer = TfidfVectorizer(max_features=MAX_LENGTH)
    encoded_triplet = {}
    for key, text in triplet.items():
        encoded_text = vectorizer.fit_transform([text])
        encoded_triplet[f"{key}_input_ids"] = encoded_text.toarray().flatten()
    return encoded_triplet

class TripletDataset:
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.dataset = load_dataset(dataset_path)
        self.instance_id_map = {item['instance_id']: item['problem_statement'] for item in self.dataset}
        self.folder_paths = get_subfolder_paths(snippet_folder_path)
        self.snippets = [load_json_file(os.path.join(folder_path, 'snippet.json')) for folder_path in self.folder_paths]
        self.bug_snippets, self.non_bug_snippets = zip(*[separate_snippets(snippet) for snippet in self.snippets])
        self.problem_statements = [self.instance_id_map.get(os.path.basename(folder_path)) for folder_path in self.folder_paths]
        self.triplets = [create_triplets(problem_statement, bug_snippets, non_bug_snippets, NUM_NEGATIVES_PER_POSITIVE) for problem_statement, bug_snippets, non_bug_snippets in zip(self.problem_statements, self.bug_snippets, self.non_bug_snippets)]

    def __len__(self):
        return len(self.dataset) * NUM_NEGATIVES_PER_POSITIVE

    def __getitem__(self, index):
        folder_index = index // NUM_NEGATIVES_PER_POSITIVE
        triplet_index = index % NUM_NEGATIVES_PER_POSITIVE
        return encode_triplet(self.triplets[folder_index][triplet_index])

class TripletModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        input_anchor = Input(shape=(MAX_LENGTH,), name='anchor_input_ids')
        input_positive = Input(shape=(MAX_LENGTH,), name='positive_input_ids')
        input_negative = Input(shape=(MAX_LENGTH,), name='negative_input_ids')

        embedding = Dense(EMBEDDING_DIM, activation='relu')(input_anchor)
        embedding = Dropout(DROPOUT)(embedding)
        embedding = Dense(FC_DIM, activation='relu')(embedding)
        anchor_embedding = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(embedding)

        embedding = Dense(EMBEDDING_DIM, activation='relu')(input_positive)
        embedding = Dropout(DROPOUT)(embedding)
        embedding = Dense(FC_DIM, activation='relu')(embedding)
        positive_embedding = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(embedding)

        embedding = Dense(EMBEDDING_DIM, activation='relu')(input_negative)
        embedding = Dropout(DROPOUT)(embedding)
        embedding = Dense(FC_DIM, activation='relu')(embedding)
        negative_embedding = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(embedding)

        return Model(inputs=[input_anchor, input_positive, input_negative], outputs=[anchor_embedding, positive_embedding, negative_embedding])

    def _triplet_loss(self, y_true, y_pred):
        anchor, positive, negative = y_pred
        loss = tf.maximum(0.0, tf.reduce_mean(tf.square(anchor - positive)) - tf.reduce_mean(tf.square(anchor - negative)) + 1.0)
        return loss

    def train(self, dataset, batch_size, epochs):
        checkpoint = ModelCheckpoint('triplet_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')
        self.model.compile(loss=self._triplet_loss, optimizer=Adam(lr=LEARNING_RATE))
        self.model.fit([np.array([item['anchor_input_ids'] for item in dataset]), 
                        np.array([item['positive_input_ids'] for item in dataset]), 
                        np.array([item['negative_input_ids'] for item in dataset])], 
                       epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping])

    def evaluate(self, dataset, batch_size):
        self.model.compile(loss=self._triplet_loss, optimizer=Adam(lr=LEARNING_RATE))
        loss = self.model.evaluate([np.array([item['anchor_input_ids'] for item in dataset]), 
                                    np.array([item['positive_input_ids'] for item in dataset]), 
                                    np.array([item['negative_input_ids'] for item in dataset])], 
                                   epochs=1, batch_size=batch_size)
        return loss

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    dataset = TripletDataset(dataset_path, snippet_folder_path)
    dataset_list = [dataset[i] for i in range(len(dataset))]
    train_dataset, test_dataset = train_test_split(dataset_list, test_size=0.2, random_state=42)
    model = TripletModel()
    model.train(train_dataset, BATCH_SIZE, MAX_EPOCHS)
    loss = model.evaluate(test_dataset, BATCH_SIZE)
    print(f"Test loss: {loss}")

if __name__ == "__main__":
    main()