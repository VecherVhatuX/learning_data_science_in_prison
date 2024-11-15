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

def create_config():
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

def create_model(config, tokenizer):
    model = tf.keras.models.Sequential([
        layers.Dense(config['EMBEDDING_SIZE'], activation='relu'),
        layers.Dropout(config['DROPOUT_RATE']),
        layers.Dense(config['FULLY_CONNECTED_SIZE'], activation='relu'),
        layers.Dropout(config['DROPOUT_RATE'])
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['LEARNING_RATE_VALUE']))
    model.tokenizer = tokenizer
    return model

def encode_text(tokenizer, inputs):
    encoding = tokenizer.encode_plus(
        inputs, 
        max_length=MAX_SEQUENCE_LENGTH, 
        padding='max_length', 
        truncation=True, 
        return_attention_mask=True, 
        return_tensors='tf'
    )
    return encoding['input_ids'][:, 0, :]

def triplet_loss(anchor, positive, negative):
    return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1) + 1.0, 0.0))

def train_step(model, data):
    anchor, positive, negative = data
    with tf.GradientTape() as tape:
        anchor_embedding = model(anchor, training=True)
        positive_embedding = model(positive, training=True)
        negative_embedding = model(negative, training=True)
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return {"loss": loss}

def create_dataset(triplets, tokenizer):
    return list(map(lambda triplet: {
        'anchor': encode_text(tokenizer, triplet['anchor']),
        'positive': encode_text(tokenizer, triplet['positive']),
        'negative': encode_text(tokenizer, triplet['negative'])
    }, triplets))

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
    triplets = [create_triplets(problem_statement, bug_snippets[i], non_bug_snippets[i], NEGATIVE_SAMPLES_PER_POSITIVE) 
                for i, problem_statement in enumerate(problem_statements)]
    return [item for sublist in triplets for item in sublist]

def load_data(dataset_path, snippet_folder_path, tokenizer):
    triplets = create_triplet_dataset(dataset_path, snippet_folder_path)
    random.shuffle(triplets)
    train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
    return create_dataset(train_triplets, tokenizer), create_dataset(test_triplets, tokenizer)

def batch_data(data):
    random.shuffle(data)
    for i in range(0, len(data), MINIBATCH_SIZE):
        batch = data[i:i + MINIBATCH_SIZE]
        anchors = np.stack([item['anchor'] for item in batch])
        positives = np.stack([item['positive'] for item in batch])
        negatives = np.stack([item['negative'] for item in batch])
        yield tf.data.Dataset.from_tensor_slices((anchors, positives, negatives)).batch(MINIBATCH_SIZE)

def train_model(model, dataset):
    total_loss = 0
    for batch in batch_data(dataset):
        loss = train_step(model, batch)[['loss']]
        total_loss += loss
    return total_loss / len(dataset)

def evaluate_model(model, dataset):
    total_loss = 0
    for batch in batch_data(dataset):
        anchor, positive, negative = batch
        anchor_embedding = model(anchor, training=False)
        positive_embedding = model(positive, training=False)
        negative_embedding = model(negative, training=False)
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        total_loss += loss
    return total_loss / len(dataset)

def save_model_weights(model, path):
    model.save_weights(path)

def load_model_weights(path, tokenizer):
    model = create_model(create_config(), tokenizer)
    model.load_weights(path)
    return model

def plot_training_history(history):
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset, test_dataset = load_data(dataset_path, snippet_folder_path, tokenizer)
    model = create_model(create_config(), tokenizer)
    model_path = 'triplet_model.h5'
    history = {'loss': [], 'val_loss': []}
    for epoch in range(MAX_TRAINING_EPOCHS):
        loss = train_model(model, train_dataset)
        val_loss = evaluate_model(model, test_dataset)
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        save_model_weights(model, model_path)
        print(f'Model saved at {model_path}')
    plot_training_history(history)

if __name__ == "__main__":
    main()