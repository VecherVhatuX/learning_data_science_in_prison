import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, optimizers


def create_embedder(emb_dim, feat_dim):
    model = tf.keras.Sequential([
        layers.Embedding(emb_dim, feat_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(feat_dim),
        layers.BatchNormalization(),
        layers.LayerNormalization()
    ])
    return model


def create_triplet_data(samples, labels, neg_count):
    dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
    return dataset.map(lambda anchor_sample, anchor_label: (
        anchor_sample, 
        tf.convert_to_tensor(random.choice(samples[np.where(labels == anchor_label)[0]])),
        tf.convert_to_tensor(random.sample(samples[np.where(labels != anchor_label)[0]].tolist(), neg_count))
    ))


def triplet_loss(margin=1.0):
    def loss_fn(anchor, positive, negative):
        pos_distance = tf.norm(anchor - positive, axis=1)
        neg_distance = tf.norm(anchor[:, tf.newaxis] - negative, axis=2)
        return tf.reduce_mean(tf.maximum(pos_distance - tf.reduce_min(neg_distance, axis=1) + margin, 0.0))
    return loss_fn


def train(model, dataset, epochs, lr):
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fn = triplet_loss()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for anchors, positives, negatives in dataset.batch(32):
            with tf.GradientTape() as tape:
                loss = loss_fn(anchors, positives, negatives)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
        loss_history.append(epoch_loss / len(dataset))
    return loss_history


def evaluate_model(model, samples, labels, k=5):
    model.evaluate(samples, labels)
    embeddings = get_embeddings(model, samples)
    metrics = calculate_knn_metrics(embeddings, labels, k)
    display_metrics(metrics)
    plot_embeddings(embeddings, labels)


def display_metrics(metrics):
    print(f"KNN Accuracy: {metrics[0]:.4f}")
    print(f"KNN Precision: {metrics[1]:.4f}")
    print(f"KNN Recall: {metrics[2]:.4f}")
    print(f"KNN F1-score: {metrics[3]:.4f}")


def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, size)


def save_model(model, path):
    model.save_weights(path)


def load_model(model_class, path):
    model = model_class()
    model.load_weights(path)
    return model


def get_embeddings(model, input_data):
    return model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()


def plot_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.show()


def calculate_knn_metrics(embeddings, labels, k=5):
    distance_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1:k + 1]
    
    tp_count = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    precision = np.mean(tp_count / k)
    recall = np.mean(tp_count / np.sum(labels == labels[:, np.newaxis], axis=1))
    
    accuracy = np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def execute_training_pipeline(lr, batch_size, epochs, neg_count, emb_dim, feat_dim, data_size):
    samples, labels = generate_data(data_size)
    triplet_data = create_triplet_data(samples, labels, neg_count)
    model = create_embedder(emb_dim, feat_dim)
    loss_history = train(model, triplet_data, epochs, lr)
    save_model(model, "triplet_model.h5")
    plot_training_loss(loss_history)
    evaluate_model(model, samples, labels)


def visualize_model_architecture(model):
    model.summary()


if __name__ == "__main__":
    execute_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    visualize_model_architecture(create_embedder(101, 10))