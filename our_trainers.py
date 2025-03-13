import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers

def create_embedding_model(embedding_dim, feature_dim):
    return tf.keras.Sequential([
        layers.Embedding(embedding_dim, feature_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(feature_dim),
        layers.BatchNormalization(),
        layers.LayerNormalization()
    ])

def generate_random_dataset(size):
    return (np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, size))

def create_triplet_data(samples, labels, negative_count):
    def map_fn(anchor, label):
        positive = tf.convert_to_tensor(random.choice(samples[labels == label.numpy()]))
        negatives = tf.convert_to_tensor(random.sample(samples[labels != label.numpy()].tolist(), negative_count))
        return anchor, positive, negatives
    return tf.data.Dataset.from_tensor_slices((samples, labels)).map(map_fn)

def triplet_loss_function(margin=1.0):
    def loss_fn(anchor, positive, negative):
        pos_dist = tf.norm(anchor - positive, axis=1)
        neg_dist = tf.reduce_min(tf.norm(anchor[:, tf.newaxis] - negative, axis=2), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss_fn

def train_embedding_model(model, dataset, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = triplet_loss_function()
    loss_history = []

    for _ in range(epochs):
        epoch_loss = 0
        for batch in dataset.batch(32):
            anchors, positives, negatives = batch
            with tf.GradientTape() as tape:
                loss = loss_fn(anchors, positives, negatives)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        loss_history.append(epoch_loss / len(dataset))
    return loss_history

def evaluate_model(model, samples, labels, k=5):
    embeddings = model(tf.convert_to_tensor(samples, dtype=tf.int32)).numpy()
    metrics = calculate_knn_metrics(embeddings, labels, k)
    display_metrics(metrics)
    plot_embeddings(embeddings, labels)

def display_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_model(model, filepath):
    model.save_weights(filepath)

def load_model(model_class, filepath):
    model = model_class()
    model.load_weights(filepath)
    return model

def get_embeddings(model, input_data):
    return model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()

def plot_embeddings(embeddings, labels):
    reduced_data = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def calculate_knn_metrics(embeddings, labels, k=5):
    dist_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]
    true_positives = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def execute_training_pipeline(learning_rate, batch_size, epochs, negative_count, embedding_dim, feature_dim, data_size):
    samples, labels = generate_random_dataset(data_size)
    triplet_data = create_triplet_data(samples, labels, negative_count)
    model = create_embedding_model(embedding_dim, feature_dim)
    loss_history = train_embedding_model(model, triplet_data, epochs, learning_rate)
    save_model(model, "triplet_model.h5")
    plot_training_loss(loss_history)
    evaluate_model(model, samples, labels)

def display_model_summary(model):
    model.summary()

if __name__ == "__main__":
    execute_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_summary(create_embedding_model(101, 10))