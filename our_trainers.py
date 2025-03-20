import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, models, optimizers, losses

class WordEmbeddingNetwork(models.Sequential):
    def __init__(self, vocabulary_size, embedding_dimension):
        super().__init__([
            layers.Embedding(vocabulary_size, embedding_dimension),
            layers.GlobalAveragePooling1D(),
            layers.Dense(embedding_dimension),
            layers.BatchNormalization(),
            layers.LayerNormalization()
        ])

class TripletDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_data, target_labels, negative_samples):
        self.input_data = input_data
        self.target_labels = target_labels
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        anchor = self.input_data[index]
        positive = random.choice(self.input_data[self.target_labels == self.target_labels[index]])
        negative = random.sample(self.input_data[self.target_labels != self.target_labels[index]].tolist(), self.negative_samples)
        return anchor, positive, negative

def compute_triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin=1.0):
    positive_distance = tf.norm(anchor_embedding - positive_embedding, axis=1)
    negative_distance = tf.reduce_min(tf.norm(tf.expand_dims(anchor_embedding, 1) - negative_embedding, axis=2), axis=1)
    return tf.reduce_mean(tf.maximum(positive_distance - negative_distance + margin, 0.0))

def train_embedding_network(model, data_generator, num_epochs, learning_rate):
    optimizer = optimizers.Adam(learning_rate)
    learning_rate_scheduler = optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=30, decay_rate=0.1)
    loss_tracker = []

    for _ in range(num_epochs):
        batch_losses = []
        for anchor, positive, negative in data_generator:
            with tf.GradientTape() as tape:
                loss = compute_triplet_loss(model(anchor), model(positive), model(negative)) + 0.01 * sum(tf.norm(param, ord=2) for param in model.trainable_variables)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            batch_losses.append(loss.numpy())
        learning_rate_scheduler.step()
        loss_tracker.append(np.mean(batch_losses))
    return loss_tracker

def evaluate_embedding_network(model, input_data, target_labels, top_k=5):
    embeddings = model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()
    distance_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_neighbors = np.argsort(distance_matrix, axis=1)[:, 1:top_k+1]
    correct_predictions = np.sum(target_labels[nearest_neighbors] == target_labels[:, np.newaxis], axis=1)

    accuracy = np.mean(np.any(target_labels[nearest_neighbors] == target_labels[:, np.newaxis], axis=1))
    precision = np.mean(correct_predictions / top_k)
    recall = np.mean(correct_predictions / np.sum(target_labels == target_labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=target_labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def save_embedding_network(model, file_path):
    model.save_weights(file_path)

def load_embedding_network(model_class, file_path, vocabulary_size, embedding_dimension):
    model = model_class(vocabulary_size, embedding_dimension)
    model.load_weights(file_path)
    return model

def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def generate_synthetic_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

def visualize_embedding_space(model, input_data, target_labels):
    embeddings = model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()
    tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=target_labels, cmap='viridis')
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {target_labels[np.argmin(np.linalg.norm(tsne_embeddings - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def visualize_similarity_matrix(model, input_data):
    embeddings = model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()
    cosine_similarity = np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings, axis=1))

    plt.figure(figsize=(8, 8))
    plt.imshow(cosine_similarity, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')
    plt.show()

def visualize_embedding_distribution(model, input_data):
    embeddings = model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()
    plt.figure(figsize=(8, 8))
    plt.hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Embedding Value Distribution')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.show()

def visualize_learning_rate_schedule(optimizer, scheduler, num_epochs):
    learning_rate_history = []
    for _ in range(num_epochs):
        learning_rate_history.append(optimizer.learning_rate.numpy())
        scheduler.step()
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rate_history, label='Learning Rate', color='red')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

def visualize_embedding_clusters(model, input_data, target_labels, n_clusters=5):
    from sklearn.cluster import KMeans
    embeddings = model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar()
    plt.title('Embedding Clusters')
    plt.show()

if __name__ == "__main__":
    synthetic_data, synthetic_labels = generate_synthetic_data(100)
    data_generator = TripletDataGenerator(synthetic_data, synthetic_labels, 5)
    data_loader = tf.data.Dataset.from_generator(lambda: data_generator, output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    ).batch(32).shuffle(1000)
    embedding_model = WordEmbeddingNetwork(101, 10)
    model_optimizer = optimizers.Adam(1e-4)
    learning_rate_scheduler = optimizers.schedules.ExponentialDecay(1e-4, decay_steps=30, decay_rate=0.1)
    training_loss_history = train_embedding_network(embedding_model, data_loader, 10, 1e-4)
    save_embedding_network(embedding_model, "embedding_model.h5")
    plot_training_loss(training_loss_history)
    evaluate_embedding_network(embedding_model, synthetic_data, synthetic_labels)
    visualize_embedding_space(load_embedding_network(WordEmbeddingNetwork, "embedding_model.h5", 101, 10), *generate_synthetic_data(100))
    visualize_similarity_matrix(embedding_model, synthetic_data)
    visualize_embedding_distribution(embedding_model, synthetic_data)
    visualize_learning_rate_schedule(model_optimizer, learning_rate_scheduler, 10)
    visualize_embedding_clusters(embedding_model, synthetic_data, synthetic_labels)