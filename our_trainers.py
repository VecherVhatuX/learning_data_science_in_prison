import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, optimizers, losses

class WordVectorGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super(WordVectorGenerator, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_size)
        self.pooling_layer = layers.GlobalAveragePooling1D()
        self.projection_layer = layers.Dense(embedding_size)
        self.batch_norm = layers.BatchNormalization()
        self.layer_norm = layers.LayerNormalization()

    def call(self, input_ids):
        embeddings = self.embedding_layer(input_ids)
        pooled = self.pooling_layer(embeddings)
        projected = self.projection_layer(pooled)
        normalized = self.batch_norm(projected)
        return self.layer_norm(normalized)

class TripletDataLoader(tf.keras.utils.Sequence):
    def __init__(self, data, labels, negative_samples):
        self.data = data
        self.labels = labels
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor = self.data[index]
        anchor_label = self.labels[index]
        positive = random.choice(self.data[self.labels == anchor_label])
        negatives = random.sample(self.data[self.labels != anchor_label].tolist(), self.negative_samples)
        return tf.convert_to_tensor(anchor, dtype=tf.int32), tf.convert_to_tensor(positive, dtype=tf.int32), tf.convert_to_tensor(negatives, dtype=tf.int32)

def calculate_triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = tf.norm(anchor - positive, axis=1)
    negative_distance = tf.reduce_min(tf.norm(tf.expand_dims(anchor, 1) - negative, axis=2), axis=1)
    return tf.reduce_mean(tf.maximum(positive_distance - negative_distance + margin, 0.0))

def train_vector_generator(model, dataset, num_epochs, learning_rate):
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for anchor, positive, negative in dataset:
            with tf.GradientTape() as tape:
                anchor_vector = model(anchor)
                positive_vector = model(positive)
                negative_vector = model(negative)
                loss = calculate_triplet_loss(anchor_vector, positive_vector, negative_vector)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        loss_history.append(epoch_loss / len(dataset))
    return loss_history

def evaluate_model(model, data, labels, k=5):
    embeddings = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    show_metrics(compute_performance_metrics(embeddings, labels, k))
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
    tsne_result = TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
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
    dataset = TripletDataLoader(data, labels, negative_samples)
    model = WordVectorGenerator(vocab_size, embedding_size)
    save_model(model, "word_vector_generator.h5")
    loss_history = train_vector_generator(model, dataset, num_epochs, learning_rate)
    plot_loss_history(loss_history)
    evaluate_model(model, data, labels)

def display_model_architecture(model):
    model.summary()

def train_with_early_termination(model, dataset, num_epochs, learning_rate, patience=5):
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss_history = []
    best_loss = float('inf')
    no_improvement = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for anchor, positive, negative in dataset:
            with tf.GradientTape() as tape:
                anchor_vector = model(anchor)
                positive_vector = model(positive)
                negative_vector = model(negative)
                loss = calculate_triplet_loss(anchor_vector, positive_vector, negative_vector)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        avg_loss = epoch_loss / len(dataset))
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return loss_history

def generate_data(data_size):
    data = np.random.randint(0, 100, (data_size, 10))
    labels = np.random.randint(0, 10, data_size)
    return data, labels

def visualize_embeddings_interactive(model, data, labels):
    embeddings = generate_embeddings(model, data)
    tsne_result = TSNE(n_components=2).fit_transform(embeddings)
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    
    def on_click(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            distances = np.linalg.norm(tsne_result - np.array([x, y]), axis=1)
            closest_index = np.argmin(distances)
            print(f"Clicked on point with label: {labels[closest_index]}")
    
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def add_custom_regularization(model, lambda_reg=0.01):
    regularization_loss = 0
    for param in model.trainable_variables:
        regularization_loss += tf.norm(param, ord=2)
    return lambda_reg * regularization_loss

def add_learning_rate_scheduler(optimizer, step_size=30, gamma=0.1):
    return optimizers.schedules.ExponentialDecay(initial_learning_rate=optimizer.learning_rate, decay_steps=step_size, decay_rate=gamma)

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_architecture(WordVectorGenerator(101, 10))
    model = load_model(WordVectorGenerator, "word_vector_generator.h5", 101, 10)
    data, labels = generate_data(100)
    visualize_embeddings_interactive(model, data, labels)