import os
import random
import json
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

class TripletModel:
    def __init__(self, dataset_path, snippet_folder_path):
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

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
        bug_snippets = []
        non_bug_snippets = []
        for folder_path, snippet_file_path in snippets:
            snippet_data = self.load_json_data(snippet_file_path)
            if snippet_data.get('is_bug', False) and snippet_data.get('snippet'):
                bug_snippets.append(snippet_data['snippet'])
            elif not snippet_data.get('is_bug', False) and snippet_data.get('snippet'):
                non_bug_snippets.append(snippet_data['snippet'])
        return bug_snippets, non_bug_snippets

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)} 
                for positive_doc in positive_snippets 
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

    def create_triplet_dataset(self):
        dataset = self.load_dataset(self.dataset_path)
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
        snippets = self.load_snippets(self.snippet_folder_path)
        triplets = []
        for folder_path, _ in snippets:
            bug_snippets, non_bug_snippets = self.separate_code_snippets([snippets[snippets.index((folder_path, os.path.join(folder_path, 'snippet.json')))]])
            problem_statement = instance_id_map.get(os.path.basename(folder_path))
            triplets.extend(self.create_triplets(problem_statement, bug_snippets, non_bug_snippets, 3))
        return triplets

    def shuffle_samples(self, samples):
        random.shuffle(samples)
        return samples

    def encode_triplet(self, triplet, max_sequence_length):
        anchor = tf.squeeze(self.tokenizer.encode_plus(triplet['anchor'], 
                                                       max_length=max_sequence_length, 
                                                       padding='max_length', 
                                                       truncation=True, 
                                                       return_attention_mask=True, 
                                                       return_tensors='tf')['input_ids'])
        positive = tf.squeeze(self.tokenizer.encode_plus(triplet['positive'], 
                                                         max_length=max_sequence_length, 
                                                         padding='max_length', 
                                                         truncation=True, 
                                                         return_attention_mask=True, 
                                                         return_tensors='tf')['input_ids'])
        negative = tf.squeeze(self.tokenizer.encode_plus(triplet['negative'], 
                                                         max_length=max_sequence_length, 
                                                         padding='max_length', 
                                                         truncation=True, 
                                                         return_attention_mask=True, 
                                                         return_tensors='tf')['input_ids'])
        return anchor, positive, negative

    def create_dataset(self, triplets, max_sequence_length, minibatch_size):
        triplets = self.shuffle_samples(triplets)
        anchor_docs = []
        positive_docs = []
        negative_docs = []
        for triplet in triplets:
            anchor, positive, negative = self.encode_triplet(triplet, max_sequence_length)
            anchor_docs.append(anchor)
            positive_docs.append(positive)
            negative_docs.append(negative)

        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_docs)
        positive_dataset = tf.data.Dataset.from_tensor_slices(positive_docs)
        negative_dataset = tf.data.Dataset.from_tensor_slices(negative_docs)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        return dataset.batch(minibatch_size).prefetch(tf.data.AUTOTUNE)

    def create_model(self, embedding_size, fully_connected_size, dropout_rate, max_sequence_length, learning_rate_value):
        model = tf.keras.Sequential([
            layers.Embedding(input_dim=1000, output_dim=embedding_size, input_length=max_sequence_length),
            layers.GlobalAveragePooling1D(),
            layers.Dense(embedding_size, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(fully_connected_size, activation='relu'),
            layers.Dropout(dropout_rate)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_value), loss=self.triplet_loss_function)
        return model

    def triplet_loss_function(self, y_true, y_pred):
        anchor, positive, negative = y_pred
        return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1) + 1.0, 0.0))

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

    def run(self):
        triplets = self.create_triplet_dataset()
        train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
        
        train_dataset = self.create_dataset(train_triplets, max_sequence_length=512, minibatch_size=16)
        test_dataset = self.create_dataset(test_triplets, max_sequence_length=512, minibatch_size=16)
        
        model = self.create_model(embedding_size=128, fully_connected_size=64, dropout_rate=0.2, max_sequence_length=512, learning_rate_value=1e-5)
        
        history = self.train_model(model, train_dataset, test_dataset, max_training_epochs=5)
        self.plot_results(history)

if __name__ == "__main__":
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    model = TripletModel(dataset_path, snippet_folder_path)
    model.run()