import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random


class NeuralEmbedder(tf.keras.Model):
    def __init__(self, embedding_dim, feature_dim):
        super(NeuralEmbedder, self).__init__()
        self.embedding = layers.Embedding(embedding_dim, feature_dim)
        self.pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(feature_dim)
        self.batch_norm = layers.BatchNormalization()
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.pooling(x)
        x = self.dense(x)
        x = self.batch_norm(x)
        return self.layer_norm(x)


class CustomTripletDataset(tf.keras.utils.Sequence):
    def __init__(self, data_samples, data_labels, num_negatives, batch_size=32):
        self.data_samples = data_samples
        self.data_labels = data_labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.indices = np.arange(len(self.data_samples))

    def __len__(self):
        return int(np.ceil(len(self.data_samples) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        anchors, positives, negatives = [], [], []
        
        for anchor_idx in batch_indices:
            anchor = self.data_samples[anchor_idx]
            anchor_label = self.data_labels[anchor_idx]
            pos_idx = random.choice(np.where(self.data_labels == anchor_label)[0])
            negative_indices = random.sample(np.where(self.data_labels != anchor_label)[0].tolist(), self.num_negatives)
            anchors.append(anchor)
            positives.append(self.data_samples[pos_idx])
            negatives.append([self.data_samples[i] for i in negative_indices])

        return np.array(anchors), np.array(positives), np.array(negatives)

    def shuffle_data(self):
        np.random.shuffle(self.indices)


def build_triplet_dataset(data_samples, data_labels, num_negatives, batch_size=32):
    return CustomTripletDataset(data_samples, data_labels, num_negatives, batch_size)


def compute_triplet_loss(margin_value=1.0):
    def loss_fn(anchor, positive, negative_samples):
        pos_dist = tf.norm(anchor - positive, axis=1)
        neg_dist = tf.norm(tf.expand_dims(anchor, axis=1) - tf.convert_to_tensor(negative_samples), axis=2)
        return tf.reduce_mean(tf.maximum(pos_dist - tf.reduce_min(neg_dist, axis=1) + margin_value, 0.0))
    return loss_fn


def train_neural_network(model, data_set, epochs):
    optimizer = tf.keras.optimizers.Adam()
    loss_function = compute_triplet_loss()
    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: loss_function(y_pred[0], y_pred[1], y_pred[2]))

    for epoch in range(epochs):
        data_set.shuffle_data()
        for anchors, positives, negatives in data_set:
            with tf.GradientTape() as tape:
                loss_value = model([anchors, positives, negatives])
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def evaluate_model(model, data_samples, data_labels, k_neighbors=5):
    embeddings = model.predict(data_samples)
    metrics = compute_knn_metrics(embeddings, data_labels, k_neighbors)
    display_evaluation_metrics(metrics)
    visualize_embeddings(embeddings, data_labels)


def display_evaluation_metrics(metrics):
    print(f"Validation KNN Accuracy: {metrics[0]}")
    print(f"Validation KNN Precision: {metrics[1]}")
    print(f"Validation KNN Recall: {metrics[2]}")
    print(f"Validation KNN F1-score: {metrics[3]}")


def create_random_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, (size,))


def save_model(model, file_path):
    model.save(file_path)


def extract_model_embeddings(model, data_samples):
    return model.predict(data_samples)


def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()


def compute_knn_metrics(embeddings, labels, k_neighbors=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_indices = np.argsort(distances, axis=1)[:, 1:k_neighbors + 1]
    
    true_positive_count = np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1)
    precision = np.mean(true_positive_count / k_neighbors)
    recall = np.mean(true_positive_count / np.sum(labels == labels[:, np.newaxis], axis=1))
    
    accuracy = np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def run_training_pipeline(learning_rate, batch_size, num_epochs, num_negatives, embedding_dim, feature_dim, data_size):
    samples, labels = create_random_data(data_size)
    triplet_dataset = build_triplet_dataset(samples, labels, num_negatives, batch_size)
    model = NeuralEmbedder(embedding_dim, feature_dim)
    train_neural_network(model, triplet_dataset, num_epochs)
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    model(input_ids)
    save_model(model, "triplet_model.h5")
    predicted_embeddings = extract_model_embeddings(model, samples)
    evaluate_model(model, samples, labels)

def load_model(file_path):
    return tf.keras.models.load_model(file_path)

def generate_additional_data(size):
    """Generates additional data samples for model evaluation."""
    return np.random.randn(size, 10), np.random.randint(0, 2, size)

if __name__ == "__main__":
    run_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)