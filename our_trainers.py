import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.utils import to_categorical
import numpy as np

def create_triplet_network(num_embeddings, embedding_dim, margin):
    inputs = layers.Input(shape=(None,), name='input_ids')
    embedding = layers.Embedding(num_embeddings, embedding_dim)(inputs)
    pooling = layers.GlobalAveragePooling1D()(embedding)
    dense = layers.Dense(embedding_dim)(pooling)
    normalize = layers.BatchNormalization()(dense)
    outputs = normalize / tf.norm(normalize, axis=-1, keepdims=True)
    return Model(inputs, outputs)

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    anchor_positive_distance = tf.norm(anchor_embeddings - positive_embeddings, axis=-1)
    anchor_negative_distance = tf.norm(anchor_embeddings[:, tf.newaxis] - negative_embeddings, axis=-1)
    min_anchor_negative_distance = tf.reduce_min(anchor_negative_distance, axis=-1)
    return tf.reduce_mean(tf.maximum(anchor_positive_distance - min_anchor_negative_distance + margin, 0))

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    def __len__():
        return -(-len(samples) // batch_size)

    def __getitem__(idx):
        start_idx = idx * batch_size
        end_idx = min((idx + 1) * batch_size, len(samples))
        anchor_idx = np.arange(start_idx, end_idx)
        anchor_labels = labels[anchor_idx]

        positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_labels])

        return {
            'anchor_input_ids': samples[anchor_idx],
            'positive_input_ids': samples[positive_idx],
            'negative_input_ids': samples[negative_idx]
        }

    return type('TripletDataset', (), {
        '__len__': __len__,
        '__getitem__': __getitem__
    })()

def train(network, dataset, margin, lr, epochs):
    optimizer = optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            anchor_inputs = data['anchor_input_ids']
            positive_inputs = data['positive_input_ids']
            negative_inputs = data['negative_input_ids']

            with tf.GradientTape() as tape:
                anchor_embeddings = network(anchor_inputs, training=True)
                positive_embeddings = network(positive_inputs, training=True)
                negative_embeddings = network(negative_inputs, training=True)
                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            gradients = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, network.trainable_variables))
            total_loss += loss
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

def evaluate(network, dataset, margin):
    total_loss = 0.0
    for i, data in enumerate(dataset):
        anchor_inputs = data['anchor_input_ids']
        positive_inputs = data['positive_input_ids']
        negative_inputs = data['negative_input_ids']

        anchor_embeddings = network(anchor_inputs, training=False)
        positive_embeddings = network(positive_inputs, training=False)
        negative_embeddings = network(negative_inputs, training=False)

        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
        total_loss += loss
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict(network, input_ids):
    return network(input_ids, training=False)

def save_model(network, path):
    network.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)

def main():
    np.random.seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    network = create_triplet_network(num_embeddings, embedding_dim, margin)
    dataset = create_triplet_dataset(samples, labels, batch_size, num_negatives)
    train(network, dataset, margin, lr, epochs)
    input_ids = np.array([1, 2, 3, 4, 5])[None, :]
    output = predict(network, input_ids)
    print(output)
    save_model(network, "triplet_model.h5")
    loaded_network = load_model("triplet_model.h5")
    print("Model saved and loaded successfully.")

if __name__ == "__main__":
    main()