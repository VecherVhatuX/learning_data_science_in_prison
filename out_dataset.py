import os
import random
import json
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

def load_json_data(path: str) -> list:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file: {path}, error: {str(e)}")
        return []

def load_dataset(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)

def load_snippets(snippet_folder_path: str) -> list:
    folder_paths = [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    return [load_json_data(os.path.join(folder_path, 'snippet.json')) for folder_path in folder_paths]

def separate_code_snippets(snippets: list) -> tuple:
    bug_snippets = []
    non_bug_snippets = []
    for item in snippets:
        if item.get('is_bug', False) and item.get('snippet'):
            bug_snippets.append(item['snippet'])
        elif not item.get('is_bug', False) and item.get('snippet'):
            non_bug_snippets.append(item['snippet'])
    return bug_snippets, non_bug_snippets

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

def create_triplet_dataset(dataset_path: str, snippet_folder_path: str) -> list:
    dataset = load_dataset(dataset_path)
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    snippets = load_snippets(snippet_folder_path)
    triplets = []
    for i, folder_path in enumerate([os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]):
        bug_snippets, non_bug_snippets = separate_code_snippets(snippets[i])
        problem_statement = instance_id_map.get(os.path.basename(folder_path))
        triplets.extend(create_triplets(problem_statement, bug_snippets, non_bug_snippets, 3))
    return triplets

def shuffle_samples(samples: list) -> None:
    random.shuffle(samples)

def encode_triplet(triplet: dict, max_sequence_length: int, tokenizer: AutoTokenizer) -> tuple:
    anchor = tf.squeeze(tokenizer.encode_plus(triplet['anchor'], 
                                               max_length=max_sequence_length, 
                                               padding='max_length', 
                                               truncation=True, 
                                               return_attention_mask=True, 
                                               return_tensors='tf')['input_ids'])
    positive = tf.squeeze(tokenizer.encode_plus(triplet['positive'], 
                                                 max_length=max_sequence_length, 
                                                 padding='max_length', 
                                                 truncation=True, 
                                                 return_attention_mask=True, 
                                                 return_tensors='tf')['input_ids'])
    negative = tf.squeeze(tokenizer.encode_plus(triplet['negative'], 
                                                 max_length=max_sequence_length, 
                                                 padding='max_length', 
                                                 truncation=True, 
                                                 return_attention_mask=True, 
                                                 return_tensors='tf')['input_ids'])
    return anchor, positive, negative

def create_dataset(triplets: list, max_sequence_length: int, minibatch_size: int, tokenizer: AutoTokenizer) -> tf.data.Dataset:
    shuffle_samples(triplets)
    anchor_docs = []
    positive_docs = []
    negative_docs = []
    for triplet in triplets:
        anchor, positive, negative = encode_triplet(triplet, max_sequence_length, tokenizer)
        anchor_docs.append(anchor)
        positive_docs.append(positive)
        negative_docs.append(negative)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_docs)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_docs)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_docs)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    return dataset.batch(minibatch_size).prefetch(tf.data.AUTOTUNE)

def create_model(embedding_size: int, fully_connected_size: int, dropout_rate: int, max_sequence_length: int, learning_rate_value: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=1000, output_dim=embedding_size, input_length=max_sequence_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(embedding_size, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(fully_connected_size, activation='relu'),
        layers.Dropout(dropout_rate)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_value), loss=triplet_loss)
    return model

def triplet_loss(y_true: tf.Tensor, y_pred: tuple) -> tf.Tensor:
    anchor, positive, negative = y_pred
    return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.norm(anchor - negative, axis=1) + 1.0, 0.0))

def train_model(model: tf.keras.Model, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, max_training_epochs: int) -> dict:
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="triplet_model_{epoch:02d}.h5",
        save_weights_only=True,
        save_freq="epoch",
        verbose=1
    )
    history = model.fit(train_dataset, epochs=max_training_epochs, 
                        validation_data=test_dataset, callbacks=[checkpoint_callback])
    return history.history

def plot_results(history: dict) -> None:
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    triplets = create_triplet_dataset(dataset_path, snippet_folder_path)
    random.shuffle(triplets)
    train_triplets, test_triplets = triplets[:int(0.8 * len(triplets))], triplets[int(0.8 * len(triplets)):]
    
    train_dataset = create_dataset(train_triplets, max_sequence_length=512, minibatch_size=16, tokenizer=tokenizer)
    test_dataset = create_dataset(test_triplets, max_sequence_length=512, minibatch_size=16, tokenizer=tokenizer)
    
    model = create_model(embedding_size=128, fully_connected_size=64, dropout_rate=0.2, max_sequence_length=512, learning_rate_value=1e-5)
    
    history = train_model(model, train_dataset, test_dataset, max_training_epochs=5)
    plot_results(history)

if __name__ == "__main__":
    main()