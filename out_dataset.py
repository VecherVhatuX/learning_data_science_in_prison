import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, GlobalMaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def load_json_data(data_path, folder_path):
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return ({item['instance_id']: item['problem_statement'] for item in dataset},
            [(os.path.join(folder, f), os.path.join(folder, f, 'snippet.json')) 
             for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])

def generate_triplets(instance_map, snippets):
    bug_snippets, non_bug_snippets = [], []
    for _, snippet_file in snippets:
        snippet_content = json.load(open(snippet_file))
        (bug_snippets if snippet_content.get('is_bug', False) else non_bug_snippets).append(snippet_content['snippet'])
    
    bug_snippets = [s for s in bug_snippets if s]
    non_bug_snippets = [s for s in non_bug_snippets if s]
    
    return [{'anchor': instance_map[os.path.basename(folder)],
             'positive': pos_doc,
             'negative': random.choice(non_bug_snippets)} 
            for folder, _ in snippets 
            for pos_doc in bug_snippets]

class TripletDataset:
    def __init__(self, triplet_data, batch_size, max_length):
        self.triplet_data = triplet_data
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = Tokenizer()
        self._fit_tokenizer()
        self.epoch_counter = 0

    def _fit_tokenizer(self):
        texts = [item['anchor'] for item in self.triplet_data] + \
                [item['positive'] for item in self.triplet_data] + \
                [item['negative'] for item in self.triplet_data]
        self.tokenizer.fit_on_texts(texts)
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def _prepare_sequences(self, item):
        sequences = [pad_sequences([self.tokenizer.texts_to_sequences([item[key]])[0]], 
                                    maxlen=self.max_length)[0] for key in ['anchor', 'positive', 'negative']]
        return {'anchor_seq': sequences[0], 'positive_seq': sequences[1], 'negative_seq': sequences[2]}

    def create_tf_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.triplet_data).map(
            lambda x: tf.py_function(self._prepare_sequences, [x], [tf.int32, tf.int32, tf.int32]), 
            num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size)

    def shuffle_data(self):
        random.shuffle(self.triplet_data)

    def next_epoch(self):
        self.epoch_counter += 1
        self.shuffle_data()

def build_model(vocab_size, embed_dim, seq_length):
    anchor_input = Input(shape=(seq_length,), name='anchor_input')
    positive_input = Input(shape=(seq_length,), name='positive_input')
    negative_input = Input(shape=(seq_length,), name='negative_input')

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)
    anchor_embedded = embedding_layer(anchor_input)
    positive_embedded = embedding_layer(positive_input)
    negative_embedded = embedding_layer(negative_input)

    pooling_layer = GlobalMaxPooling1D()
    anchor_pooled = pooling_layer(anchor_embedded)
    positive_pooled = pooling_layer(positive_embedded)
    negative_pooled = pooling_layer(negative_embedded)

    dense_layer = Dense(128, activation='relu')
    batch_norm_layer = BatchNormalization()
    dropout_layer = Dropout(0.2)

    anchor_dense = dropout_layer(batch_norm_layer(dense_layer(anchor_pooled)))
    positive_dense = dropout_layer(batch_norm_layer(dense_layer(positive_pooled)))
    negative_dense = dropout_layer(batch_norm_layer(dense_layer(negative_pooled)))

    model = Model(inputs=[anchor_input, positive_input, negative_input], 
                  outputs=[anchor_dense, positive_dense, negative_dense])
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error')
    return model

def triplet_loss(anchor_embeds, positive_embeds, negative_embeds):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) - 
                                     tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train_model(model, train_loader, test_loader, num_epochs):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(1, num_epochs + 1):
        train_loader.next_epoch()
        train_data = train_loader.create_tf_dataset().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        
        epoch_loss = 0
        for batch in train_data:
            anchor_seq, positive_seq, negative_seq = batch['anchor_seq'].numpy(), batch['positive_seq'].numpy(), batch['negative_seq'].numpy()
            with tf.GradientTape() as tape:
                anchor_out, positive_out, negative_out = model([anchor_seq, positive_seq, negative_seq], training=True)
                batch_loss = triplet_loss(anchor_out, positive_out, negative_out)
            grads = tape.gradient(batch_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += batch_loss

        print(f'Epoch {epoch}, Train Loss: {epoch_loss / len(train_data)}')
        train_losses.append(epoch_loss / len(train_data))

        test_loss, correct_preds = 0, 0
        for batch in test_loader.create_tf_dataset().batch(32).prefetch(tf.data.AUTOTUNE):
            anchor_seq, positive_seq, negative_seq = batch['anchor_seq'].numpy(), batch['positive_seq'].numpy(), batch['negative_seq'].numpy()
            anchor_out, positive_out, negative_out = model([anchor_seq, positive_seq, negative_seq])
            batch_loss = triplet_loss(anchor_out, positive_out, negative_out)
            test_loss += batch_loss

            pos_similarity = tf.reduce_sum(anchor_out * positive_out, axis=1)
            neg_similarity = tf.reduce_sum(anchor_out * negative_out, axis=1)
            correct_preds += tf.reduce_sum(tf.cast(pos_similarity > neg_similarity, tf.float32))

        acc = correct_preds / len(list(test_loader.create_tf_dataset()))
        print(f'Test Loss: {test_loss / len(list(test_loader.create_tf_dataset()))}')
        print(f'Test Accuracy: {acc}')
        test_losses.append(test_loss / len(list(test_loader.create_tf_dataset())))
        test_accs.append(acc)
        train_accs.append(correct_preds / len(list(train_loader.create_tf_dataset())))

    return train_losses, test_losses, train_accs, test_accs

def plot_results(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    data_path = 'datasets/SWE-bench_oracle.npy'
    snippet_path = 'datasets/10_10_after_fix_pytest'
    
    instance_map, snippets = load_json_data(data_path, snippet_path)
    triplets = generate_triplets(instance_map, snippets)
    train_triplets, valid_triplets = np.array_split(np.array(triplets), 2)
    
    train_loader = TripletDataset(train_triplets.tolist(), batch_size=32, max_length=512)
    test_loader = TripletDataset(valid_triplets.tolist(), batch_size=32, max_length=512)
    
    model = build_model(train_loader.vocab_size, embed_dim=128, seq_length=512)
    checkpoint_path = "training_1/cp.ckpt"
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

    train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, num_epochs=5)
    plot_results(train_losses, test_losses, train_accs, test_accs)

if __name__ == "__main__":
    main()