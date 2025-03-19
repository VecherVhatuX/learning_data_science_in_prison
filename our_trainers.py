import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, models, optimizers, losses

class EmbeddingModel(models.Sequential):
    def __init__(self, vocab_size, embed_dim):
        super().__init__([
            layers.Embedding(vocab_size, embed_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(embed_dim),
            layers.BatchNormalization(),
            layers.LayerNormalization()
        ])

class TripletData(tf.keras.utils.Sequence):
    def __init__(self, data, labels, neg_samples):
        self.data = data
        self.labels = labels
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        pos = random.choice(self.data[self.labels == self.labels[idx]])
        neg = random.sample(self.data[self.labels != self.labels[idx]].tolist(), self.neg_samples)
        return anchor, pos, neg

def triplet_loss(anchor, pos, neg, margin=1.0):
    pos_dist = tf.norm(anchor - pos, axis=1)
    neg_dist = tf.reduce_min(tf.norm(tf.expand_dims(anchor, 1) - neg, axis=2), axis=1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

def train_model(model, loader, epochs, lr):
    optimizer = optimizers.Adam(lr)
    scheduler = optimizers.schedules.ExponentialDecay(lr, decay_steps=30, decay_rate=0.1)
    losses = []

    for _ in range(epochs):
        epoch_loss = []
        for anchor, pos, neg in loader:
            with tf.GradientTape() as tape:
                loss = triplet_loss(model(anchor), model(pos), model(neg)) + 0.01 * sum(tf.norm(p, ord=2) for p in model.trainable_variables)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss.append(loss.numpy())
        scheduler.step()
        losses.append(np.mean(epoch_loss))
    return losses

def evaluate(model, data, labels, k=5):
    embeddings = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]
    true_positives = np.sum(labels[neighbors] == labels[:, np.newaxis], axis=1)

    accuracy = np.mean(np.any(labels[neighbors] == labels[:, np.newaxis], axis=1))
    precision = np.mean(true_positives / k)
    recall = np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    tsne = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def save(model, path):
    model.save_weights(path)

def load(model_class, path, vocab_size, embed_dim):
    model = model_class(vocab_size, embed_dim)
    model.load_weights(path)
    return model

def plot_loss_history(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def generate_data(data_size):
    return np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

def visualize(model, data, labels):
    embeddings = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    tsne = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def display_similarity(model, data):
    embeddings = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    cosine_sim = np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings, axis=1))

    plt.figure(figsize=(8, 8))
    plt.imshow(cosine_sim, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')
    plt.show()

def plot_distribution(model, data):
    embeddings = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    plt.figure(figsize=(8, 8))
    plt.hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Embedding Value Distribution')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.show()

def plot_learning_rate(optimizer, scheduler, epochs):
    lr_history = []
    for _ in range(epochs):
        lr_history.append(optimizer.learning_rate.numpy())
        scheduler.step()
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, label='Learning Rate', color='red')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data, labels = generate_data(100)
    dataset = TripletData(data, labels, 5)
    loader = tf.data.Dataset.from_generator(lambda: dataset, output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    ).batch(32).shuffle(1000)
    model = EmbeddingModel(101, 10)
    optimizer = optimizers.Adam(1e-4)
    scheduler = optimizers.schedules.ExponentialDecay(1e-4, decay_steps=30, decay_rate=0.1)
    loss_history = train_model(model, loader, 10, 1e-4)
    save(model, "embedding_model.h5")
    plot_loss_history(loss_history)
    evaluate(model, data, labels)
    visualize(load(EmbeddingModel, "embedding_model.h5", 101, 10), *generate_data(100))
    display_similarity(model, data)
    plot_distribution(model, data)
    plot_learning_rate(optimizer, scheduler, 10)