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

class TripletDataset:
    def __init__(self, triplet_data, batch_size, max_length):
        self.triplet_data = triplet_data
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self._gather_texts())
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def _gather_texts(self):
        return [item[key] for item in self.triplet_data for key in ['anchor', 'positive', 'negative']]

    def _convert_to_sequences(self, item):
        return {
            'anchor_seq': pad_sequences([self.tokenizer.texts_to_sequences([item['anchor']])[0]], maxlen=self.max_length)[0],
            'positive_seq': pad_sequences([self.tokenizer.texts_to_sequences([item['positive']])[0]], maxlen=self.max_length)[0],
            'negative_seq': pad_sequences([self.tokenizer.texts_to_sequences([item['negative']])[0]], maxlen=self.max_length)[0]
        }

    def create_tf_dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.triplet_data).map(
            lambda x: tf.py_function(self._convert_to_sequences, [x], [tf.int32, tf.int32, tf.int32]),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(self.batch_size)

    def shuffle_data(self):
        random.shuffle(self.triplet_data)

    def next_epoch(self):
        self.shuffle_data()

def load_json_data(data_path, folder_path):
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    instance_map = {item['instance_id']: item['problem_statement'] for item in dataset}
    snippets = [
        (folder, os.path.join(folder, 'snippet.json'))
        for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))
    ]
    return instance_map, snippets

def generate_triplets(instance_map, snippets):
    bug_snippets, non_bug_snippets = zip(*(map(lambda snippet_file: json.load(open(snippet_file)), snippets)))
    bug_snippets = [s['snippet'] for s in bug_snippets if s.get('is_bug') and s['snippet']]
    non_bug_snippets = [s['snippet'] for s in non_bug_snippets if not s.get('is_bug') and s['snippet']]
    return create_triplet_structure(instance_map, snippets, bug_snippets, non_bug_snippets)

def create_triplet_structure(instance_map, snippets, bug_snippets, non_bug_snippets):
    return [
        {
            'anchor': instance_map[os.path.basename(folder)],
            'positive': pos_doc,
            'negative': random.choice(non_bug_snippets)
        }
        for folder, _ in snippets
        for pos_doc in bug_snippets
    ]

def build_model(vocab_size, embed_dim, seq_length):
    anchor_input, positive_input, negative_input = create_model_inputs(seq_length)
    anchor_embedded, positive_embedded, negative_embedded = create_embedding_layers(anchor_input, positive_input, negative_input, vocab_size, embed_dim)
    anchor_dense, positive_dense, negative_dense = create_dense_layers(anchor_embedded, positive_embedded, negative_embedded)

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=[anchor_dense, positive_dense, negative_dense])
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mean_squared_error')
    return model

def create_model_inputs(seq_length):
    return (Input(shape=(seq_length,), name='anchor_input'),
            Input(shape=(seq_length,), name='positive_input'),
            Input(shape=(seq_length,), name='negative_input'))

def create_embedding_layers(anchor_input, positive_input, negative_input, vocab_size, embed_dim):
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)
    return (embedding_layer(anchor_input), embedding_layer(positive_input), embedding_layer(negative_input))

def create_dense_layers(anchor_embedded, positive_embedded, negative_embedded):
    pooling_layer = GlobalMaxPooling1D()
    dense_layer = Dense(128, activation='relu')
    batch_norm_layer = BatchNormalization()
    dropout_layer = Dropout(0.2)

    return (
        dropout_layer(batch_norm_layer(dense_layer(pooling_layer(anchor_embedded)))),
        dropout_layer(batch_norm_layer(dense_layer(pooling_layer(positive_embedded)))),
        dropout_layer(batch_norm_layer(dense_layer(pooling_layer(negative_embedded))))
    )

def triplet_loss(anchor_embeds, positive_embeds, negative_embeds):
    return tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) - 
                                     tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

def train_model(model, train_loader, test_loader, num_epochs):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(1, num_epochs + 1):
        train_loader.next_epoch()
        train_data = train_loader.create_tf_dataset().shuffle(1000).prefetch(tf.data.AUTOTUNE)

        epoch_loss = train_epoch(model, train_data)
        train_losses.append(epoch_loss / len(train_data))
        print(f'Epoch {epoch}, Train Loss: {train_losses[-1]}')

        test_loss, acc = evaluate_model(model, test_loader)
        test_losses.append(test_loss)
        test_accs.append(acc)
        print(f'Test Loss: {test_loss}, Test Accuracy: {acc}')
        train_accs.append(acc)

    return train_losses, test_losses, train_accs, test_accs

def train_epoch(model, train_data):
    epoch_loss = 0
    for batch in train_data:
        anchor_seq, positive_seq, negative_seq = batch['anchor_seq'].numpy(), batch['positive_seq'].numpy(), batch['negative_seq'].numpy()
        with tf.GradientTape() as tape:
            anchor_out, positive_out, negative_out = model([anchor_seq, positive_seq, negative_seq], training=True)
            batch_loss = triplet_loss(anchor_out, positive_out, negative_out)
        grads = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += batch_loss
    return epoch_loss

def evaluate_model(model, test_loader):
    test_loss, correct_preds = 0, 0
    for batch in test_loader.create_tf_dataset().batch(32).prefetch(tf.data.AUTOTUNE):
        anchor_seq, positive_seq, negative_seq = batch['anchor_seq'].numpy(), batch['positive_seq'].numpy(), batch['negative_seq'].numpy()
        anchor_out, positive_out, negative_out = model([anchor_seq, positive_seq, negative_seq])
        batch_loss = triplet_loss(anchor_out, positive_out, negative_out)
        test_loss += batch_loss

        pos_similarity = tf.reduce_sum(anchor_out * positive_out, axis=1)
        neg_similarity = tf.reduce_sum(anchor_out * negative_out, axis=1)
        correct_preds += tf.reduce_sum(tf.cast(pos_similarity > neg_similarity, tf.float32))

    acc = correct_preds / len(test_loader.triplet_data)
    return test_loss / len(test_loader.triplet_data), acc

def plot_results(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Rate')
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