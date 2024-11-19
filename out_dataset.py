import os
import random
import json
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

class TripletModel:
    def __init__(self, embedding_size=128, fully_connected_size=64, dropout_rate=0.2, max_sequence_length=512, learning_rate_value=1e-5):
        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.learning_rate_value = learning_rate_value

    def load_json_data(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def load_dataset(self, file_path):
        return np.load(file_path, allow_pickle=True)

    def load_snippets(self, folder_path):
        return [(os.path.join(folder_path, f), os.path.join(folder_path, f, 'snippet.json')) 
                for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    def separate_code_snippets(self, snippets):
        return tuple(map(list, zip(*[
            ((snippet_data['snippet'], True) if snippet_data.get('is_bug', False) else (snippet_data['snippet'], False)) 
            for folder_path, snippet_file_path in snippets 
            for snippet_data in [self.load_json_data(snippet_file_path)]
            if snippet_data.get('snippet')
        ])))

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)} 
                for positive_doc in positive_snippets 
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

    def create_triplet_dataset(self, dataset_path, snippet_folder_path):
        dataset = self.load_dataset(dataset_path)
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
        snippets = self.load_snippets(snippet_folder_path)
        return [
            (problem_statement, bug_snippet, non_bug_snippet) 
            for folder_path, _ in snippets 
            for bug_snippets, non_bug_snippets in [self.separate_code_snippets([(folder_path, os.path.join(folder_path, 'snippet.json'))])]
            for problem_statement in [instance_id_map.get(os.path.basename(folder_path))] 
            for bug_snippet, non_bug_snippet in [(bug, non_bug) for bug in bug_snippets for non_bug in non_bug_snippets]
        ]

    def shuffle_samples(self, samples):
        random.shuffle(samples)
        return samples

    def encode_triplet(self, triplet, tokenizer):
        return tuple(map(
            lambda text: tf.squeeze(tokenizer.encode_plus(text, 
                                                           max_length=self.max_sequence_length, 
                                                           padding='max_length', 
                                                           truncation=True, 
                                                           return_attention_mask=True, 
                                                           return_tensors='tf')['input_ids']),
            [triplet['anchor'], triplet['positive'], triplet['negative']]
        ))

    def create_dataset(self, triplets, minibatch_size, tokenizer):
        triplets = self.shuffle_samples(triplets)
        anchor_docs, positive_docs, negative_docs = tuple(map(list, zip(*[
            self.encode_triplet(triplet, tokenizer) 
            for triplet in triplets
        ])))
        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_docs)
        positive_dataset = tf.data.Dataset.from_tensor_slices(positive_docs)
        negative_dataset = tf.data.Dataset.from_tensor_slices(negative_docs)
        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        return dataset.batch(minibatch_size).prefetch(tf.data.AUTOTUNE)

    def create_model(self):
        model = tf.keras.Sequential([
            layers.Embedding(input_dim=1000, output_dim=self.embedding_size, input_length=self.max_sequence_length),
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.embedding_size, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.fully_connected_size, activation='relu'),
            layers.Dropout(self.dropout_rate)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate_value), loss=lambda y_true, y_pred: tf.reduce_mean(tf.maximum(tf.norm(y_pred[:, 0] - y_pred[:, 1], axis=1) - tf.norm(y_pred[:, 0] - y_pred[:, 2], axis=1) + 1.0, 0.0)))
        return model

    def train_model(self, model, train_dataset, test_dataset, max_training_epochs):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="triplet_model_{epoch:02d}.h5",
            save_weights_only=True,
            save_freq="epoch",
            verbose=1
        )
        history = model.fit(train_dataset, epochs=max_training_epochs, 
                            validation_data=test_dataset, callbacks=[checkpoint_callback])
        return history.history

    def plot_results(self, history):
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    def pipeline(self, dataset_path, snippet_folder_path, num_negatives_per_positive=1, minibatch_size=16, max_training_epochs=5):
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        triplets = self.create_triplet_dataset(dataset_path, snippet_folder_path)
        train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
        train_dataset = self.create_dataset(train_triplets, minibatch_size, tokenizer)
        test_dataset = self.create_dataset(test_triplets, minibatch_size, tokenizer)
        model = self.create_model()
        history = self.train_model(model, train_dataset, test_dataset, max_training_epochs)
        self.plot_results(history)


if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    TripletModel().pipeline(dataset_path, snippet_folder_path)