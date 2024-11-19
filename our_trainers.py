import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

class TripletNetwork(keras.Model):
    def __init__(self, num_embeddings, embedding_dim, margin):
        super(TripletNetwork, self).__init__()
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(embedding_dim)
        self.normalize = layers.BatchNormalization()
        self.margin = margin

    def call(self, inputs):
        embedding = self.embedding(inputs)
        pooling = self.pooling(embedding)
        dense = self.dense(pooling)
        normalize = self.normalize(dense)
        outputs = normalize / tf.norm(normalize, axis=1, keepdims=True)
        return outputs

    def triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return tf.reduce_mean(tf.maximum(
            tf.norm(anchor_embeddings - positive_embeddings, axis=1) 
            - tf.reduce_min(tf.norm(anchor_embeddings[:, None] - negative_embeddings, axis=2), axis=1) + self.margin, 0))

class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        anchor_idx = np.arange(start_idx, end_idx)
        anchor_labels = self.labels[anchor_idx]

        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_labels])

        return {
            'anchor_input_ids': self.samples[anchor_idx],
            'positive_input_ids': self.samples[positive_idx],
            'negative_input_ids': self.samples[negative_idx]
        }

def train_triplet_network(network, dataset, epochs, learning_rate):
    optimizer = keras.optimizers.Adam(learning_rate)
    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            with tf.GradientTape() as tape:
                anchor_embeddings = network(data['anchor_input_ids'])
                positive_embeddings = network(data['positive_input_ids'])
                negative_embeddings = network(data['negative_input_ids'])
                loss = network.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            gradients = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, network.trainable_variables))
            total_loss += loss
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

def evaluate_triplet_network(network, dataset):
    total_loss = 0.0
    for i, data in enumerate(dataset):
        anchor_embeddings = network(data['anchor_input_ids'])
        positive_embeddings = network(data['positive_input_ids'])
        negative_embeddings = network(data['negative_input_ids'])
        loss = network.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        total_loss += loss
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict_with_triplet_network(network, input_ids):
    return network(input_ids)

def save_triplet_model(network, path):
    network.save(path)

def load_triplet_model(path):
    return keras.models.load_model(path)

def calculate_distance(embedding1, embedding2):
    return tf.norm(embedding1 - embedding2, axis=1)

def calculate_similarity(embedding1, embedding2):
    return tf.reduce_sum(embedding1 * embedding2, axis=1) / (tf.norm(embedding1, axis=1) * tf.norm(embedding2, axis=1))

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
    learning_rate = 1e-4

    network = TripletNetwork(num_embeddings, embedding_dim, margin)
    dataset = TripletDataset(samples, labels, batch_size, num_negatives)
    train_triplet_network(network, dataset, epochs, learning_rate)
    input_ids = np.array([1, 2, 3, 4, 5])[None, :]
    output = predict_with_triplet_network(network, input_ids)
    print(output)
    save_triplet_model(network, "triplet_model.h5")
    loaded_network = load_triplet_model("triplet_model.h5")
    print("Model saved and loaded successfully.")

    evaluate_triplet_network(network, dataset)

    predicted_embeddings = predict_with_triplet_network(network, np.array([1, 2, 3, 4, 5])[None, :])
    print(predicted_embeddings)

    distance = calculate_distance(predicted_embeddings, predicted_embeddings)
    print(distance)

    similarity = calculate_similarity(predicted_embeddings, predicted_embeddings)
    print(similarity)

if __name__ == "__main__":
    main()