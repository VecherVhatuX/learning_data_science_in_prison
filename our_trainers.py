import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers

def build_embedder(emb_dim, feat_dim):
    return tf.keras.Sequential([
        layers.Embedding(emb_dim, feat_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(feat_dim),
        layers.BatchNormalization(),
        layers.LayerNormalization()
    ])

def create_random_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, size)

def generate_triplet_dataset(samples, labels, neg_count):
    return tf.data.Dataset.from_tensor_slices((samples, labels)).map(
        lambda anchor, label: (
            anchor,
            tf.convert_to_tensor(random.choice(samples[labels == label.numpy()]), dtype=tf.float32),
            tf.convert_to_tensor(random.sample(samples[labels != label.numpy()].tolist(), neg_count), dtype=tf.float32)
        )
    )

def create_triplet_loss(margin=1.0):
    return lambda anchor, positive, negative: tf.reduce_mean(
        tf.maximum(tf.norm(anchor - positive, axis=1) - tf.reduce_min(tf.norm(anchor[:, tf.newaxis] - negative, axis=2), axis=1) + margin, 0.0)
    )

def train_model(model, dataset, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = create_triplet_loss()
    losses = []

    for _ in range(epochs):
        epoch_loss = 0
        for batch in dataset.batch(32):
            anchors, positives, negatives = batch
            with tf.GradientTape() as tape:
                loss = loss_fn(anchors, positives, negatives)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        losses.append(epoch_loss / len(dataset))
    return losses

def assess_model(model, samples, labels, k=5):
    model.evaluate(samples, labels)
    embeddings = model(tf.convert_to_tensor(samples, dtype=tf.int32)).numpy()
    metrics = compute_knn_metrics(embeddings, labels, k)
    print_metrics(metrics)
    visualize_embeddings(embeddings, labels)

def print_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_model_weights(model, filepath):
    model.save_weights(filepath)

def load_model_weights(model_class, filepath):
    model = model_class()
    model.load_weights(filepath)
    return model

def extract_embeddings(model, input_data):
    return model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()

def visualize_embeddings(embeddings, labels):
    reduced_data = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def compute_knn_metrics(embeddings, labels, k=5):
    dist_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]
    true_positives = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training_pipeline(learning_rate, batch_size, epochs, neg_count, emb_dim, feat_dim, data_size):
    samples, labels = create_random_data(data_size)
    triplet_data = generate_triplet_dataset(samples, labels, neg_count)
    model = build_embedder(emb_dim, feat_dim)
    loss_history = train_model(model, triplet_data, epochs, learning_rate)
    save_model_weights(model, "triplet_model.h5")
    plot_loss(loss_history)
    assess_model(model, samples, labels)

def show_model_summary(model):
    model.summary()

if __name__ == "__main__":
    run_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    show_model_summary(build_embedder(101, 10))