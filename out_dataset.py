import os
import random
import json
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

# constants
INSTANCE_ID_KEY = 'instance_id'
MAX_SEQUENCE_LENGTH = 512
MINIBATCH_SIZE = 16
NEGATIVE_SAMPLES_PER_POSITIVE = 3
EMBEDDING_SIZE = 128
FULLY_CONNECTED_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE_VALUE = 1e-5
MAX_TRAINING_EPOCHS = 5

class TripletModel:
    def __init__(self, config, tokenizer):
        """
        Initialize the TripletModel with the given configuration and tokenizer.
        
        Parameters:
        config (dict): The configuration dictionary.
        tokenizer: The tokenizer instance.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model = tf.keras.models.Sequential([
            layers.Dense(config['EMBEDDING_SIZE'], activation='relu', name='embedding_layer'),
            layers.Dropout(config['DROPOUT_RATE'], name='dropout_layer_1'),
            layers.Dense(config['FULLY_CONNECTED_SIZE'], activation='relu', name='fully_connected_layer'),
            layers.Dropout(config['DROPOUT_RATE'], name='dropout_layer_2')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['LEARNING_RATE_VALUE']))

    def encode_text(self, inputs):
        """
        Encode the input text using the tokenizer.
        
        Parameters:
        inputs (str): The input text.
        
        Returns:
        encoded_input_ids (tf.Tensor): The encoded input IDs.
        """
        encoding = self.tokenizer.encode_plus(
            inputs, 
            max_length=MAX_SEQUENCE_LENGTH, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='tf'
        )
        return encoding['input_ids'][:, 0, :]

    def triplet_loss(self, anchor, positive, negative):
        """
        Calculate the triplet loss.
        
        Parameters:
        anchor (tf.Tensor): The anchor embedding.
        positive (tf.Tensor): The positive embedding.
        negative (tf.Tensor): The negative embedding.
        
        Returns:
        loss (tf.Tensor): The triplet loss.
        """
        return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1) + 1.0, 0.0))

    def train_step(self, data):
        """
        Perform a training step.
        
        Parameters:
        data (tf.data.Dataset): The training data.
        
        Returns:
        loss (tf.Tensor): The loss value.
        """
        anchor, positive, negative = data
        with tf.GradientTape() as tape:
            anchor_embedding = self.model(anchor, training=True)
            positive_embedding = self.model(positive, training=True)
            negative_embedding = self.model(negative, training=True)
            loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    def create_dataset(self, triplets):
        """
        Create a dataset from the given triplets.
        
        Parameters:
        triplets (list): The list of triplets.
        
        Returns:
        dataset (list): The created dataset.
        """
        return list(map(lambda triplet: {
            'anchor': self.encode_text(triplet['anchor']),
            'positive': self.encode_text(triplet['positive']),
            'negative': self.encode_text(triplet['negative'])
        }, triplets))

    def load_json_data(self, file_path):
        """
        Load JSON data from the given file path.
        
        Parameters:
        file_path (str): The file path.
        
        Returns:
        data (list): The loaded JSON data.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file: {file_path}, error: {str(e)}")
            return []

    def separate_code_snippets(self, snippets):
        """
        Separate code snippets into bug and non-bug snippets.
        
        Parameters:
        snippets (list): The list of snippets.
        
        Returns:
        bug_snippets (list): The list of bug snippets.
        non_bug_snippets (list): The list of non-bug snippets.
        """
        return (
            [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
            [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
        )

    def create_triplets(self, problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
        """
        Create triplets from the given problem statement, positive snippets, and negative snippets.
        
        Parameters:
        problem_statement (str): The problem statement.
        positive_snippets (list): The list of positive snippets.
        negative_snippets (list): The list of negative snippets.
        num_negatives_per_positive (int): The number of negatives per positive.
        
        Returns:
        triplets (list): The created triplets.
        """
        return [{'anchor': problem_statement, 'positive': positive_doc, 'negative': random.choice(negative_snippets)}
                for positive_doc in positive_snippets
                for _ in range(min(num_negatives_per_positive, len(negative_snippets)))]

    def create_triplet_dataset(self, dataset_path, snippet_folder_path):
        """
        Create a triplet dataset from the given dataset path and snippet folder path.
        
        Parameters:
        dataset_path (str): The dataset path.
        snippet_folder_path (str): The snippet folder path.
        
        Returns:
        triplets (list): The created triplets.
        """
        dataset = np.load(dataset_path, allow_pickle=True)
        instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
        folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
        snippets = [self.load_json_data(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]
        bug_snippets, non_bug_snippets = zip(*[self.separate_code_snippets(snippet) for snippet in snippets])
        problem_statements = [instance_id_map.get(os.path.basename(folder_path)) for folder_path in folder_paths]
        triplets = [self.create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], NEGATIVE_SAMPLES_PER_POSITIVE) 
                    for i, problem_statement in enumerate(problem_statements)]
        return [item for sublist in triplets for item in sublist]

    def load_data(self, dataset_path, snippet_folder_path):
        """
        Load data from the given dataset path and snippet folder path.
        
        Parameters:
        dataset_path (str): The dataset path.
        snippet_folder_path (str): The snippet folder path.
        
        Returns:
        train_dataset (list): The training dataset.
        test_dataset (list): The testing dataset.
        """
        triplets = self.create_triplet_dataset(dataset_path, snippet_folder_path)
        random.shuffle(triplets)
        train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
        return self.create_dataset(train_triplets), self.create_dataset(test_triplets)

    def batch_data(self, data):
        """
        Batch the given data.
        
        Parameters:
        data (list): The data to be batched.
        
        Yields:
        batch (tf.data.Dataset): The batched data.
        """
        random.shuffle(data)
        for i in range(0, len(data), MINIBATCH_SIZE):
            batch = data[i:i + MINIBATCH_SIZE]
            anchors = np.stack([item['anchor'] for item in batch])
            positives = np.stack([item['positive'] for item in batch])
            negatives = np.stack([item['negative'] for item in batch])
            yield tf.data.Dataset.from_tensor_slices((anchors, positives, negatives)).batch(MINIBATCH_SIZE)

    def train(self, dataset):
        """
        Train the model on the given dataset.
        
        Parameters:
        dataset (list): The training dataset.
        
        Returns:
        total_loss (float): The total loss.
        """
        total_loss = 0
        for batch in self.batch_data(dataset):
            loss = self.train_step(batch)[['loss']]
            total_loss += loss
        return total_loss / len(dataset)

    def evaluate(self, dataset):
        """
        Evaluate the model on the given dataset.
        
        Parameters:
        dataset (list): The testing dataset.
        
        Returns:
        total_loss (float): The total loss.
        """
        total_loss = 0
        for batch in self.batch_data(dataset):
            anchor, positive, negative = batch
            anchor_embedding = self.model(anchor, training=False)
            positive_embedding = self.model(positive, training=False)
            negative_embedding = self.model(negative, training=False)
            loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            total_loss += loss
        return total_loss / len(dataset)

    def save(self, path):
        """
        Save the model to the given path.
        
        Parameters:
        path (str): The path to save the model.
        """
        self.model.save_weights(path)

    def load(self, path):
        """
        Load the model from the given path.
        
        Parameters:
        path (str): The path to load the model.
        """
        self.model.load_weights(path)

class TripletTrainer:
    def __init__(self, model, dataset_path, snippet_folder_path, tokenizer):
        """
        Initialize the TripletTrainer with the given model, dataset path, snippet folder path, and tokenizer.
        
        Parameters:
        model (TripletModel): The TripletModel instance.
        dataset_path (str): The dataset path.
        snippet_folder_path (str): The snippet folder path.
        tokenizer: The tokenizer instance.
        """
        self.model = model
        self.dataset_path = dataset_path
        self.snippet_folder_path = snippet_folder_path
        self.tokenizer = tokenizer

    def train(self):
        """
        Train the model.
        
        Returns:
        history (dict): The training history.
        """
        train_dataset, test_dataset = self.model.load_data(self.dataset_path, self.snippet_folder_path)
        history = {'loss': [], 'val_loss': []}
        for epoch in range(MAX_TRAINING_EPOCHS):
            loss = self.model.train(train_dataset)
            val_loss = self.model.evaluate(test_dataset)
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
            self.model.save(f'triplet_model_{epoch+1}.h5')
            print(f'Model saved at triplet_model_{epoch+1}.h5')
        return history

def create_config():
    """
    Create a configuration dictionary.
    
    Returns:
    config (dict): The configuration dictionary.
    """
    return {
        'INSTANCE_ID_KEY': INSTANCE_ID_KEY,
        'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH,
        'MINIBATCH_SIZE': MINIBATCH_SIZE,
        'NEGATIVE_SAMPLES_PER_POSITIVE': NEGATIVE_SAMPLES_PER_POSITIVE,
        'EMBEDDING_SIZE': EMBEDDING_SIZE,
        'FULLY_CONNECTED_SIZE': FULLY_CONNECTED_SIZE,
        'DROPOUT_RATE': DROPOUT_RATE,
        'LEARNING_RATE_VALUE': LEARNING_RATE_VALUE,
        'MAX_TRAINING_EPOCHS': MAX_TRAINING_EPOCHS
    }

def plot_training_history(history):
    """
    Plot the training history.
    
    Parameters:
    history (dict): The training history.
    """
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    config = create_config()
    model = TripletModel(config, tokenizer)
    trainer = TripletTrainer(model, dataset_path, snippet_folder_path, tokenizer)
    history = trainer.train()
    plot_training_history(history)

if __name__ == "__main__":
    main()