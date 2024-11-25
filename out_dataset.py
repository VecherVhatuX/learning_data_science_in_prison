import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import json
import os
from sklearn.model_selection import train_test_split

# Hyperparameters
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_SIZE = 128
FULLY_CONNECTED_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE_VALUE = 1e-5
EPOCHS = 5
BATCH_SIZE = 32
NUM_NEGATIVES_PER_POSITIVE = 1

class DataProcessor:
    """
    Class responsible for data processing and preparation.
    """
    def __init__(self, dataset_path, snippet_folder_path, num_negatives_per_positive):
        """
        Initialize the DataProcessor object.
        
        Parameters:
        dataset_path (str): The path to the dataset file.
        snippet_folder_path (str): The path to the snippet folder.
        num_negatives_per_positive (int): The number of negative samples per positive sample.
        """
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.num_negatives_per_positive = num_negatives_per_positive

    def load_data(self, file_path):
        """
        Load the data from a file.
        
        Parameters:
        file_path (str): The path to the file.
        
        Returns:
        The loaded data.
        """
        if file_path.endswith('.npy'):
            return np.load(file_path, allow_pickle=True)
        else:
            return json.load(open(file_path, 'r', encoding='utf-8'))

    def load_snippets(self):
        """
        Load the snippets from the snippet folder.
        
        Returns:
        A list of tuples containing the folder path and snippet file path.
        """
        return [(os.path.join(self.snippet_folder_path, folder), os.path.join(self.snippet_folder_path, folder, 'snippet.json')) 
                for folder in os.listdir(self.snippet_folder_path) if os.path.isdir(os.path.join(self.snippet_folder_path, folder))]

    def separate_snippets(self, snippets):
        """
        Separate the snippets into bug snippets and non-bug snippets.
        
        Parameters:
        snippets (list): A list of snippets.
        
        Returns:
        A tuple containing the bug snippets and non-bug snippets.
        """
        bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                        for snippet_data in [self.load_data(snippet_file_path)] if snippet_data.get('is_bug', False)]
        non_bug_snippets = [snippet_data['snippet'] for _, snippet_file_path in snippets 
                            for snippet_data in [self.load_data(snippet_file_path)] if not snippet_data.get('is_bug', False)]
        return bug_snippets, non_bug_snippets

    def create_triplets(self, instance_id_map, snippets):
        """
        Create triplets from the snippets.
        
        Parameters:
        instance_id_map (dict): A dictionary mapping instance IDs to problem statements.
        snippets (list): A list of snippets.
        
        Returns:
        A list of triplets.
        """
        bug_snippets, non_bug_snippets = self.separate_snippets(snippets)
        return [{'anchor': instance_id_map[os.path.basename(folder_path)], 'positive': positive_doc, 'negative': random.choice(non_bug_snippets)} 
                for folder_path, _ in snippets 
                for positive_doc in bug_snippets 
                for _ in range(min(self.num_negatives_per_positive, len(non_bug_snippets)))]

    def prepare_data(self):
        """
        Prepare the data for training and testing.
        
        Returns:
        A tuple containing the training data and testing data.
        """
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in self.load_data(self.dataset_path)}
        snippets = self.load_snippets()
        triplets = self.create_triplets(instance_id_map, snippets)
        return np.array_split(np.array(triplets), 2)


class DataTokenizer:
    """
    Class responsible for tokenizing the data.
    """
    def __init__(self, max_sequence_length):
        """
        Initialize the DataTokenizer object.
        
        Parameters:
        max_sequence_length (int): The maximum sequence length.
        """
        self.max_sequence_length = max_sequence_length

    def tokenize_triplets(self, triplets):
        """
        Tokenize the triplets.
        
        Parameters:
        triplets (list): A list of triplets.
        
        Returns:
        A numpy array containing the tokenized triplets.
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([triplet['anchor'] for triplet in triplets] + 
                                [triplet['positive'] for triplet in triplets] + 
                                [triplet['negative'] for triplet in triplets])
        anchor_sequences = tokenizer.texts_to_sequences([triplet['anchor'] for triplet in triplets])
        positive_sequences = tokenizer.texts_to_sequences([triplet['positive'] for triplet in triplets])
        negative_sequences = tokenizer.texts_to_sequences([triplet['negative'] for triplet in triplets])
        anchor_padded = pad_sequences(anchor_sequences, maxlen=self.max_sequence_length, padding='post')
        positive_padded = pad_sequences(positive_sequences, maxlen=self.max_sequence_length, padding='post')
        negative_padded = pad_sequences(negative_sequences, maxlen=self.max_sequence_length, padding='post')
        return np.stack((anchor_padded, positive_padded, negative_padded), axis=1)


class ModelTrainer:
    """
    Class responsible for training the model.
    """
    def __init__(self, learning_rate_value, epochs, batch_size):
        """
        Initialize the ModelTrainer object.
        
        Parameters:
        learning_rate_value (float): The learning rate value.
        epochs (int): The number of epochs.
        batch_size (int): The batch size.
        """
        self.learning_rate_value = learning_rate_value
        self.epochs = epochs
        self.batch_size = batch_size

    def create_model(self):
        """
        Create the model.
        
        Returns:
        The created model.
        """
        return tf.keras.Sequential([
            layers.Embedding(input_dim=10000, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQUENCE_LENGTH),
            layers.Bidirectional(layers.LSTM(FULLY_CONNECTED_SIZE, dropout=DROPOUT_RATE)),
            layers.Dense(FULLY_CONNECTED_SIZE, activation='relu'),
            layers.Dense(EMBEDDING_SIZE)
        ])

    def calculate_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        """
        Calculate the loss.
        
        Parameters:
        anchor_embeddings (tensor): The anchor embeddings.
        positive_embeddings (tensor): The positive embeddings.
        negative_embeddings (tensor): The negative embeddings.
        
        Returns:
        The calculated loss.
        """
        return tf.reduce_mean((anchor_embeddings - positive_embeddings) ** 2) + tf.keras.backend.maximum(tf.reduce_mean((anchor_embeddings - negative_embeddings) ** 2) - tf.reduce_mean((anchor_embeddings - positive_embeddings) ** 2), 0)

    def train(self, model, dataset):
        """
        Train the model.
        
        Parameters:
        model (tf.keras.Model): The model to train.
        dataset (tf.data.Dataset): The dataset to train on.
        
        Returns:
        The total loss.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_value)
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                anchor_embeddings = model(batch[:, 0])
                positive_embeddings = model(batch[:, 1])
                negative_embeddings = model(batch[:, 2])
                loss = self.calculate_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
        return total_loss / len(dataset)

    def evaluate(self, model, dataset):
        """
        Evaluate the model.
        
        Parameters:
        model (tf.keras.Model): The model to evaluate.
        dataset (tf.data.Dataset): The dataset to evaluate on.
        
        Returns:
        The accuracy.
        """
        total_correct = 0
        for batch in dataset:
            anchor_embeddings = model(batch[:, 0])
            positive_embeddings = model(batch[:, 1])
            negative_embeddings = model(batch[:, 2])
            for i in range(len(anchor_embeddings)):
                similarity_positive = tf.reduce_sum(anchor_embeddings[i] * positive_embeddings[i]) / (tf.norm(anchor_embeddings[i]) * tf.norm(positive_embeddings[i]))
                similarity_negative = tf.reduce_sum(anchor_embeddings[i] * negative_embeddings[i]) / (tf.norm(anchor_embeddings[i]) * tf.norm(negative_embeddings[i]))
                total_correct += int(similarity_positive > similarity_negative)
        return total_correct / len(dataset)


def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    data_processor = DataProcessor(dataset_path, snippet_folder_path, NUM_NEGATIVES_PER_POSITIVE)
    train_triplets, test_triplets = data_processor.prepare_data()
    data_tokenizer = DataTokenizer(MAX_SEQUENCE_LENGTH)
    train_tokenized_triplets = data_tokenizer.tokenize_triplets(train_triplets)
    test_tokenized_triplets = data_tokenizer.tokenize_triplets(test_triplets)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_tokenized_triplets).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_tokenized_triplets).batch(BATCH_SIZE)
    model_trainer = ModelTrainer(LEARNING_RATE_VALUE, EPOCHS, BATCH_SIZE)
    model = model_trainer.create_model()
    for epoch in range(EPOCHS):
        loss = model_trainer.train(model, train_dataset)
        print(f'Epoch {epoch+1}, Loss: {loss}')
    print(f'Test Accuracy: {model_trainer.evaluate(model, test_dataset)}')


if __name__ == "__main__":
    main()