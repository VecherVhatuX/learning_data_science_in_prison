import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def load_data(dataset_path, snippet_folder_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    instance_id_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    snippet_directories = [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
                           for f in os.listdir(snippet_folder_path) if os.path.isdir(os.path.join(snippet_folder_path, f))]
    snippets = [(folder, json.load(open(snippet_file))) for folder, snippet_file in snippet_directories]
    return instance_id_map, snippets

def create_triplets(instance_id_map, snippets):
    bug_snippets = []
    non_bug_snippets = []
    for _, snippet_file in snippets:
        snippet_data = snippet_file
        if snippet_data.get('is_bug', False):
            bug_snippets.append(snippet_data['snippet'])
        else:
            non_bug_snippets.append(snippet_data['snippet'])
    bug_snippets = [snippet for snippet in bug_snippets if snippet]
    non_bug_snippets = [snippet for snippet in non_bug_snippets if snippet]
    return [{'anchor': instance_id_map[os.path.basename(folder)], 
             'positive': positive_doc, 
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for positive_doc in bug_snippets 
            for _ in range(min(1, len(non_bug_snippets)))]

def create_model(vocab_size, embedding_dim, max_sequence_length):
    anchor_input = Input(shape=(max_sequence_length,), name='anchor_input')
    positive_input = Input(shape=(max_sequence_length,), name='positive_input')
    negative_input = Input(shape=(max_sequence_length,), name='negative_input')

    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    anchor_embedding = embedding(anchor_input)
    positive_embedding = embedding(positive_input)
    negative_embedding = embedding(negative_input)

    anchor_pooling = GlobalMaxPooling1D()(anchor_embedding)
    positive_pooling = GlobalMaxPooling1D()(positive_embedding)
    negative_pooling = GlobalMaxPooling1D()(negative_embedding)

    anchor_dense = Dense(128, activation='relu')(anchor_pooling)
    positive_dense = Dense(128, activation='relu')(positive_pooling)
    negative_dense = Dense(128, activation='relu')(negative_pooling)

    model = Model(inputs=[anchor_input, positive_input, negative_input], 
                  outputs=[anchor_dense, positive_dense, negative_dense])

    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error')
    return model

def calculate_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeddings - positive_embeddings, axis=1) - tf.norm(anchor_embeddings - negative_embeddings, axis=1), 0))

def train(model, train_data_loader, test_data_loader, epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(1, epochs+1):
        train_data_loader.dataset.triplets = np.random.permutation(train_data_loader.dataset.triplets)
        total_loss = 0
        for batch in train_data_loader:
            anchor_sequences = tf.keras.preprocessing.sequence.pad_sequences([item['anchor_sequence'] for item in batch], maxlen=512)
            positive_sequences = tf.keras.preprocessing.sequence.pad_sequences([item['positive_sequence'] for item in batch], maxlen=512)
            negative_sequences = tf.keras.preprocessing.sequence.pad_sequences([item['negative_sequence'] for item in batch], maxlen=512)
            
            anchor_output, positive_output, negative_output = model.predict([anchor_sequences, positive_sequences, negative_sequences])
            
            batch_loss = calculate_triplet_loss(anchor_output, positive_output, negative_output)
            model.trainable = True
            with tf.GradientTape() as tape:
                anchor_output, positive_output, negative_output = model([anchor_sequences, positive_sequences, negative_sequences], training=True)
                batch_loss = calculate_triplet_loss(anchor_output, positive_output, negative_output)
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += batch_loss
        print(f'Epoch {epoch}, Train Loss: {total_loss/len(train_data_loader)}')
        train_losses.append(total_loss/len(train_data_loader))
        
        total_loss = 0
        total_correct = 0
        for batch in test_data_loader:
            anchor_sequences = tf.keras.preprocessing.sequence.pad_sequences([item['anchor_sequence'] for item in batch], maxlen=512)
            positive_sequences = tf.keras.preprocessing.sequence.pad_sequences([item['positive_sequence'] for item in batch], maxlen=512)
            negative_sequences = tf.keras.preprocessing.sequence.pad_sequences([item['negative_sequence'] for item in batch], maxlen=512)
            
            anchor_output, positive_output, negative_output = model.predict([anchor_sequences, positive_sequences, negative_sequences])
            
            batch_loss = calculate_triplet_loss(anchor_output, positive_output, negative_output)
            total_loss += batch_loss
            positive_similarity = tf.reduce_sum(tf.multiply(anchor_output, positive_output), axis=1)
            negative_similarity = tf.reduce_sum(tf.multiply(anchor_output, negative_output), axis=1)
            total_correct += tf.reduce_sum((positive_similarity > negative_similarity))
        accuracy = total_correct / len(test_data_loader.dataset)
        print(f'Test Loss: {total_loss/len(test_data_loader)}')
        print(f'Test Accuracy: {accuracy}')
        test_losses.append(total_loss/len(test_data_loader))
        test_accuracies.append(accuracy)
        train_accuracies.append(total_correct / len(train_data_loader.dataset))
    return train_losses, test_losses, train_accuracies, test_accuracies

def plot(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    dataset_path = 'datasets/SWE-bench_oracle.npy'
    snippet_folder_path = 'datasets/10_10_after_fix_pytest'
    
    instance_id_map, snippets = load_data(dataset_path, snippet_folder_path)
    triplets = create_triplets(instance_id_map, snippets)
    train_triplets, test_triplets = np.array_split(np.array(triplets), 2)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([item['anchor'] for item in triplets] + [item['positive'] for item in triplets] + [item['negative'] for item in triplets])
    
    train_dataset = [{'anchor_sequence': tf.keras.preprocessing.text.text_to_word_sequence(triplet['anchor']), 
                      'positive_sequence': tf.keras.preprocessing.text.text_to_word_sequence(triplet['positive']), 
                      'negative_sequence': tf.keras.preprocessing.text.text_to_word_sequence(triplet['negative'])} 
                     for triplet in train_triplets]
    test_dataset = [{'anchor_sequence': tf.keras.preprocessing.text.text_to_word_sequence(triplet['anchor']), 
                     'positive_sequence': tf.keras.preprocessing.text.text_to_word_sequence(triplet['positive']), 
                     'negative_sequence': tf.keras.preprocessing.text.text_to_word_sequence(triplet['negative'])} 
                    for triplet in test_triplets]
    
    train_data_loader = tf.data.Dataset.from_tensor_slices(train_dataset).batch(32)
    test_data_loader = tf.data.Dataset.from_tensor_slices(test_dataset).batch(32)
    
    model = create_model(len(tokenizer.word_index) + 1, 128, 512)
    train_losses, test_losses, train_accuracies, test_accuracies = train(model, train_data_loader, test_data_loader, 5)
    plot(train_losses, test_losses, train_accuracies, test_accuracies)

if __name__ == "__main__":
    main()