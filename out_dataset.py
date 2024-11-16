import os
import random
import json
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

class Config:
    instance_id_key = 'instance_id'
    max_sequence_length = 512
    minibatch_size = 16
    negative_samples_per_positive = 3
    embedding_size = 128
    fully_connected_size = 64
    dropout_rate = 0.2
    learning_rate_value = 1e-5
    max_training_epochs = 5

class DataHelper:
    @staticmethod
    def load_json_data(path: str) -> list:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {path}, error: {str(e)}")
            return []

    @staticmethod
    def separate_code_snippets(snippets: list) -> tuple:
        bug_snippets = []
        non_bug_snippets = []
        for item in snippets:
            if item.get('is_bug', False) and item.get('snippet'):
                bug_snippets.append(item['snippet'])
            elif not item.get('is_bug', False) and item.get('snippet'):
                non_bug_snippets.append(item['snippet'])
        return bug_snippets, non_bug_snippets

    @staticmethod
    def create_triplets(problem_statement: str, positive_snippets: list, negative_snippets: list, num_negatives_per_positive: int) -> list:
        triplets = []
        for _ in range(min(num_negatives_per_positive, len(negative_snippets))):
            for positive_doc in positive_snippets:
                triplets.append({
                    'anchor': problem_statement,
                    'positive': positive_doc,
                    'negative': random.choice(negative_snippets)
                })
        return triplets

    @staticmethod
    def create_triplet_dataset(dataset_path: str, snippet_folder_path: str) -> list:
        dataset = np.load(dataset_path, allow_pickle=True)
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
        folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
        snippets = [DataHelper.load_json_data(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
        triplets = []
        for i, folder_path in enumerate(folder_paths):
            bug_snippets, non_bug_snippets = DataHelper.separate_code_snippets(snippets[i])
            problem_statement = instance_id_map.get(os.path.basename(folder_path))
            triplets.extend(DataHelper.create_triplets(problem_statement, bug_snippets, non_bug_snippets, 3))
        return triplets

    @staticmethod
    def encode_text(inputs: str, tokenizer, max_sequence_length: int) -> tf.Tensor:
        encoding = tokenizer.encode_plus(
            inputs, 
            max_length=max_sequence_length, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='tf'
        )
        return encoding['input_ids'][:, 0, :]

class TripletModel(tf.keras.Model):
    def __init__(self, config: Config):
        super(TripletModel, self).__init__()
        self.config = config
        self.tokenizer = None
        self.embedding = layers.Embedding(input_dim=1000, output_dim=config.embedding_size, input_length=config.max_sequence_length)
        self.pooling = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(config.embedding_size, activation='relu')
        self.dropout1 = layers.Dropout(config.dropout_rate)
        self.dense2 = layers.Dense(config.fully_connected_size, activation='relu')
        self.dropout2 = layers.Dropout(config.dropout_rate)
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate_value), loss=self.triplet_loss)

    def set_tokenizer(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def create_dataset(self, triplets: list) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((triplets))
        dataset = dataset.map(self.encode_triplet, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(self.config.minibatch_size).prefetch(tf.data.AUTOTUNE)

    def encode_triplet(self, triplet: dict) -> tuple:
        anchor = DataHelper.encode_text(triplet['anchor'], self.tokenizer, self.config.max_sequence_length)
        positive = DataHelper.encode_text(triplet['positive'], self.tokenizer, self.config.max_sequence_length)
        negative = DataHelper.encode_text(triplet['negative'], self.tokenizer, self.config.max_sequence_length)
        return anchor, positive, negative

    def call(self, inputs: tuple) -> tuple:
        anchor, positive, negative = inputs
        anchor_embedding = self.dropout2(self.dense2(self.dropout1(self.dense1(self.pooling(self.embedding(anchor))))))
        positive_embedding = self.dropout2(self.dense2(self.dropout1(self.dense1(self.pooling(self.embedding(positive))))))
        negative_embedding = self.dropout2(self.dense2(self.dropout1(self.dense1(self.pooling(self.embedding(negative))))))
        return anchor_embedding, positive_embedding, negative_embedding

    @staticmethod
    def triplet_loss(y_true: tf.Tensor, y_pred: tuple) -> tf.Tensor:
        anchor, positive, negative = y_pred
        return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1) + 1.0, 0.0))

class TripletTrainer:
    def __init__(self, model: TripletModel, dataset_path: str, snippet_folder_path: str):
        self.model = model
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path

    def train(self) -> dict:
        triplets = DataHelper.create_triplet_dataset(self.dataset_path, self.snippet_folder_path)
        random.shuffle(triplets)
        train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
        train_dataset = self.model.create_dataset(train_triplets)
        test_dataset = self.model.create_dataset(test_triplets)
        history = {'loss': [], 'val_loss': []}
        for epoch in range(self.model.config.max_training_epochs):
            loss = self.model.fit(train_dataset, epochs=1).history['loss'][0]
            val_loss = self.model.evaluate(test_dataset)
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
            self.model.save_weights(f'triplet_model_{epoch+1}.h5')
            print(f'Model saved at triplet_model_{epoch+1}.h5')
        return history

def plot_training_history(history: dict):
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    config = Config()
    model = TripletModel(config)
    model.set_tokenizer(tokenizer)
    trainer = TripletTrainer(model, dataset_path, snippet_folder_path)
    history = trainer.train()
    plot_training_history(history)

if __name__ == "__main__":
    main()