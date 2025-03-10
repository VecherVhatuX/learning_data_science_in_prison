import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

class TripletModel(keras.Model):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(embedding_dim, num_features)
        self.pooling = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(num_features, activation='relu')
        self.bn = layers.BatchNormalization()
        self.ln = layers.LayerNormalization()

    def call(self, x):
        x = self.embedding(x)
        x = self.pooling(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.ln(x)
        return x

class TripletDataset(tf.data.Dataset):
    def __new__(cls, samples, labels, num_negatives):
        dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
        return dataset.map(lambda sample, label: cls.get_triplet(sample, label, labels, num_negatives))

    @staticmethod
    def get_triplet(anchor, anchor_label, labels, num_negatives):
        positive_idx = random.choice(np.where(labels == anchor_label.numpy())[0].tolist())
        negative_idx = random.sample(np.where(labels != anchor_label.numpy())[0].tolist(), num_negatives)
        positive = labels[positive_idx]
        negatives = [labels[i] for i in negative_idx]
        return anchor, positive, negatives

def create_triplet_loss(margin=1.0):
    def triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred
        d_ap = tf.norm(anchor - positive, axis=1)
        d_an = tf.norm(anchor[:, tf.newaxis] - tf.stack(negative, axis=1), axis=2)
        loss = tf.maximum(d_ap - tf.reduce_min(d_an, axis=1) + margin, 0.0)
        return tf.reduce_mean(loss)
    return triplet_loss

def train_model(model, dataset, epochs):
    model.compile(optimizer=keras.optimizers.Adam(), loss=create_triplet_loss())
    model.fit(dataset.batch(32), epochs=epochs)

def validate_model(model, samples, labels, k=5):
    predicted_embeddings = model(samples).numpy()
    print("Validation KNN Accuracy:", knn_accuracy(predicted_embeddings, labels, k))
    print("Validation KNN Precision:", knn_precision(predicted_embeddings, labels, k))
    print("Validation KNN Recall:", knn_recall(predicted_embeddings, labels, k))
    print("Validation KNN F1-score:", knn_f1(predicted_embeddings, labels, k))
    visualize_embeddings(predicted_embeddings, labels)

def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))

def create_dataset(samples, labels, num_negatives):
    return TripletDataset(samples, labels, num_negatives)

def create_model(embedding_dim, num_features):
    return TripletModel(embedding_dim, num_features)

def save_model(model, path):
    model.save_weights(path)

def get_predicted_embeddings(model, samples):
    return model(samples).numpy()

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
    plt.colorbar()
    plt.show()

def knn_accuracy(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    return np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))

def knn_precision(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    true_positive = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    return np.mean(true_positive / k)

def knn_recall(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k+1]
    true_positive = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    return np.mean(true_positive / np.sum(labels == labels[:, np.newaxis], axis=1))

def knn_f1(embeddings, labels, k=5):
    precision = knn_precision(embeddings, labels, k)
    recall = knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)

def run_pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features, size):
    samples, labels = generate_data(size)
    dataset = create_dataset(samples, labels, num_negatives)
    model = create_model(embedding_dim, num_features)
    train_model(model, dataset, epochs)
    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape((1, 10))
    output = model(input_ids)
    save_model(model, "triplet_model.h5")
    predicted_embeddings = get_predicted_embeddings(model, samples)
    validate_model(model, samples, labels)

if __name__ == "__main__":
    run_pipeline(1e-4, 32, 10, 5, 101, 10, 100)