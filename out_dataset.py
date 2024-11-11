import json
import random
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

nltk.download('punkt')

def create_swebench_dict(swebench_dataset):
    return {item['instance_id']: item['problem_statement'] for item in swebench_dataset}

def load_swebench_dataset(dataset_path):
    return np.load(dataset_path, allow_pickle=True)

def load_snippet_file(snippet_file):
    try:
        with open(snippet_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load snippet file: {snippet_file}, error: {str(e)}")
        return []

def load_triplet_data(snippet_folder_path):
    return [os.path.join(snippet_folder_path, f) for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]

def separate_snippets(snippets):
    return (
        [item['snippet'] for item in snippets if item.get('is_bug', False) and item.get('snippet')],
        [item['snippet'] for item in snippets if not item.get('is_bug', False) and item.get('snippet')]
    )

def create_triplets(problem_statement, positive_snippets, negative_snippets, num_negatives_per_positive):
    return [
        {'anchor': problem_statement, 'positive': positive_doc, 'negative': negative_doc}
        for positive_doc in positive_snippets
        for negative_doc in (negative_snippets if len(negative_snippets) <= num_negatives_per_positive else random.sample(negative_snippets, num_negatives_per_positive))
    ]

def encode_data(tokenizer, data, max_length):
    return {
        'anchor_input_ids': tokenizer.encode_plus(
            text=data['anchor'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten(),
        'anchor_attention_mask': tokenizer.encode_plus(
            text=data['anchor'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten(),
        'positive_input_ids': tokenizer.encode_plus(
            text=data['positive'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten(),
        'positive_attention_mask': tokenizer.encode_plus(
            text=data['positive'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten(),
        'negative_input_ids': tokenizer.encode_plus(
            text=data['negative'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['input_ids'].flatten(),
        'negative_attention_mask': tokenizer.encode_plus(
            text=data['negative'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )['attention_mask'].flatten()
    }

def create_triplet_dataset(swebench_dataset_path, snippet_folder_path, instance_id_field='instance_id', num_negatives_per_positive=3):
    swebench_dataset = load_swebench_dataset(swebench_dataset_path)
    swebench_dict = create_swebench_dict(swebench_dataset)
    triplet_data = []
    for folder in os.listdir(snippet_folder_path):
        folder_path = os.path.join(snippet_folder_path, folder)
        if os.path.isdir(folder_path):
            snippet_file = os.path.join(folder_path, 'snippet.json')
            snippets = load_snippet_file(snippet_file)
            if snippets:
                bug_snippets, non_bug_snippets = separate_snippets(snippets)
                problem_statement = swebench_dict.get(folder)
                if problem_statement:
                    triplets = create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive)
                    triplet_data.extend(triplets)
    print(f"Number of triplets: {len(triplet_data)}")
    return triplet_data

def create_triplet_dataset_generator(swebench_dataset_path, snippet_folder_path, instance_id_field='instance_id', num_negatives_per_positive=3, tokenizer=None, max_length=512):
    swebench_dataset = load_swebench_dataset(swebench_dataset_path)
    swebench_dict = create_swebench_dict(swebench_dataset)
    for folder in os.listdir(snippet_folder_path):
        folder_path = os.path.join(snippet_folder_path, folder)
        if os.path.isdir(folder_path):
            snippet_file = os.path.join(folder_path, 'snippet.json')
            snippets = load_snippet_file(snippet_file)
            if snippets:
                bug_snippets, non_bug_snippets = separate_snippets(snippets)
                problem_statement = swebench_dict.get(folder)
                if problem_statement:
                    triplets = create_triplets(problem_statement, bug_snippets, non_bug_snippets, num_negatives_per_positive)
                    for triplet in triplets:
                        yield encode_data(tokenizer, triplet, max_length)

def create_model():
    embedding = tf.keras.layers.Embedding(input_dim=30522, output_dim=128, input_length=512)
    dropout = tf.keras.layers.Dropout(0.2)
    fc = tf.keras.layers.Dense(64, activation='relu')
    inputs = tf.keras.Input(shape=(512,))
    attention_mask = tf.keras.Input(shape=(512,))
    outputs = embedding(inputs)
    outputs = dropout(outputs)
    outputs = tf.reduce_mean(outputs, axis=1)
    outputs = fc(outputs)
    return tf.keras.Model(inputs=[inputs, attention_mask], outputs=outputs)

def train(model, dataset, epochs):
    loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.norm(y_pred[:64] - y_pred[64:128], axis=1) - tf.norm(y_pred[:64] - y_pred[128:], axis=1) + 1)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            anchor_input_ids = batch['anchor_input_ids'].numpy()
            anchor_attention_mask = batch['anchor_attention_mask'].numpy()
            positive_input_ids = batch['positive_input_ids'].numpy()
            positive_attention_mask = batch['positive_attention_mask'].numpy()
            negative_input_ids = batch['negative_input_ids'].numpy()
            negative_attention_mask = batch['negative_attention_mask'].numpy()

            anchor_inputs = (anchor_input_ids, anchor_attention_mask)
            positive_inputs = (positive_input_ids, positive_attention_mask)
            negative_inputs = (negative_input_ids, negative_attention_mask)

            anchor_output = model(anchor_inputs, training=True)
            positive_output = model(positive_inputs, training=True)
            negative_output = model(negative_inputs, training=True)

            with tf.GradientTape() as tape:
                outputs = tf.concat([anchor_output, positive_output, negative_output], axis=0)
                loss = loss_fn(tf.zeros((anchor_output.shape[0]*3,)), outputs)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

def main(swebench_dataset_path, snippet_folder_path):
    dataset = create_triplet_dataset(swebench_dataset_path, snippet_folder_path)
    if not dataset:
        print("No available triplets to create the dataset.")
        return

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset_generator = create_triplet_dataset_generator(swebench_dataset_path, snippet_folder_path, tokenizer=tokenizer)
    dataset = tf.data.Dataset.from_generator(lambda: dataset_generator, 
                                             output_types={'anchor_input_ids': tf.int32, 
                                                           'anchor_attention_mask': tf.int32, 
                                                           'positive_input_ids': tf.int32, 
                                                           'positive_attention_mask': tf.int32, 
                                                           'negative_input_ids': tf.int32, 
                                                           'negative_attention_mask': tf.int32}).batch(16).prefetch(tf.data.AUTOTUNE)

    model = create_model()

    train(model, dataset, epochs=5)

if __name__ == "__main__":
    swebench_dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'

    main(swebench_dataset_path, snippet_folder_path)