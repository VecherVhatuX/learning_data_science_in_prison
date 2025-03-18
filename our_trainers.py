import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, optimizers, losses

class WordVectorGenerator(tf.keras.Sequential):
    def __init__(self, vocab_size, embedding_size):
        super().__init__([
            layers.Embedding(vocab_size, embedding_size),
            layers.GlobalAveragePooling1D(),
            layers.Dense(embedding_size),
            layers.BatchNormalization(),
            layers.LayerNormalization()
        ])

class TripletDataLoader:
    def __init__(self, data, labels, negative_samples):
        self.data = data
        self.labels = labels
        self.negative_samples = negative_samples

    def __iter__(self):
        for i in range(len(self.data)):
            anchor = tf.convert_to_tensor(self.data[i], dtype=tf.int32)
            positive = tf.convert_to_tensor(random.choice(self.data[self.labels == self.labels[i]]), dtype=tf.int32)
            negatives = tf.convert_to_tensor(random.sample(self.data[self.labels != self.labels[i]].tolist(), self.negative_samples), dtype=tf.int32)
            yield anchor, positive, negatives

def calculate_triplet_loss(anchor, positive, negative, margin=1.0):
    return tf.reduce_mean(tf.maximum(tf.norm(anchor - positive, axis=1) - tf.reduce_min(tf.norm(tf.expand_dims(anchor, 1) - negative, axis=2), axis=1) + margin, 0.0)

def train_vector_generator(model, dataset, num_epochs, learning_rate):
    optimizer = tf.optimizers.Adam(learning_rate)
    for _ in range(num_epochs):
        for anchor, positive, negative in dataset:
            with tf.GradientTape() as tape:
                loss = calculate_triplet_loss(model(anchor), model(positive), model(negative))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def evaluate_model(model, data, labels, k=5):
    embeddings = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    metrics = compute_performance_metrics(embeddings, labels, k)
    show_metrics(metrics)
    plot_embeddings(embeddings, labels)

def show_metrics(metrics):
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1-score: {metrics[3]:.4f}")

def save_model(model, file_path):
    model.save_weights(file_path)

def load_model(model_class, file_path, vocab_size, embedding_size):
    model = model_class(vocab_size, embedding_size)
    model.load_weights(file_path)
    return model

def generate_embeddings(model, data):
    return model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()

def plot_embeddings(embeddings, labels):
    plt.figure(figsize=(8, 8))
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def compute_performance_metrics(embeddings, labels, k=5):
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    nearest_neighbors = np.argsort(distances, axis=1)[:, 1:k + 1]
    true_positives = np.sum(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)
    accuracy = np.mean(np.any(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training(learning_rate, batch_size, num_epochs, negative_samples, vocab_size, embedding_size, data_size):
    data, labels = generate_data(data_size)
    model = WordVectorGenerator(vocab_size, embedding_size)
    save_model(model, "word_vector_generator.h5")
    dataset = TripletDataLoader(data, labels, negative_samples)
    loss_history = train_vector_generator(model, dataset, num_epochs, learning_rate)
    plot_loss_history(loss_history)
    evaluate_model(model, data, labels)

def display_model_architecture(model):
    model.summary()

def train_with_early_termination(model, dataset, num_epochs, learning_rate, patience=5):
    optimizer = tf.optimizers.Adam(learning_rate)
    best_loss = float('inf')
    no_improvement = 0
    for epoch in range(num_epochs):
        avg_loss = sum(calculate_triplet_loss(model(anchor), model(positive), model(negative)) for anchor, positive, negative in dataset) / len(dataset)
        optimizer.apply_gradients(zip(tf.GradientTape().gradient(avg_loss, model.trainable_variables), model.trainable_variables))
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

def generate_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

def visualize_embeddings_interactive(model, data, labels):
    embeddings = generate_embeddings(model, data)
    tsne_result = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne_result - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def add_custom_regularization(model, lambda_reg=0.01):
    return lambda_reg * sum(tf.norm(param, ord=2) for param in model.trainable_variables)

def add_learning_rate_scheduler(optimizer, step_size=30, gamma=0.1):
    return optimizers.schedules.ExponentialDecay(initial_learning_rate=optimizer.learning_rate, decay_steps=step_size, decay_rate=gamma)

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_architecture(WordVectorGenerator(101, 10))
    visualize_embeddings_interactive(load_model(WordVectorGenerator, "word_vector_generator.h5", 101, 10), *generate_data(100))