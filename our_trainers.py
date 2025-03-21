import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, models, optimizers
from sklearn.cluster import KMeans

class EmbeddingModel(models.Sequential):
    def __init__(self, vocab_size, embed_dim):
        super().__init__([
            layers.Embedding(vocab_size, embed_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(embed_dim),
            layers.BatchNormalization(),
            layers.LayerNormalization()
        ])

class TripletGenerator(tf.keras.utils.Sequence):
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

def train_model(model, generator, epochs, lr):
    opt = optimizers.Adam(lr)
    lr_scheduler = optimizers.schedules.ExponentialDecay(lr, decay_steps=30, decay_rate=0.1)
    loss_history = []

    for _ in range(epochs):
        batch_loss = []
        for a, p, n in generator:
            with tf.GradientTape() as tape:
                loss = triplet_loss(model(a), model(p), model(n)) + 0.01 * sum(tf.norm(param, ord=2) for param in model.trainable_variables)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            batch_loss.append(loss.numpy())
        lr_scheduler.step()
        loss_history.append(np.mean(batch_loss))
    return loss_history

def evaluate_model(model, data, labels, k=5):
    embeds = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    dist_matrix = np.linalg.norm(embeds[:, np.newaxis] - embeds, axis=2)
    nn = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
    correct = np.sum(labels[nn] == labels[:, np.newaxis], axis=1)

    acc = np.mean(np.any(labels[nn] == labels[:, np.newaxis], axis=1))
    prec = np.mean(correct / k)
    rec = np.mean(correct / np.sum(labels == labels[:, np.newaxis], axis=1))
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    tsne = TSNE(n_components=2).fit_transform(embeds)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()

def save_model(model, path):
    model.save_weights(path)

def load_model(model_class, path, vocab_size, embed_dim):
    model = model_class(vocab_size, embed_dim)
    model.load_weights(path)
    return model

def plot_loss(loss_hist):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def generate_data(size):
    return np.random.randint(0, 100, (size, 10)), np.random.randint(0, 10, size)

def visualize_embeddings(model, data, labels):
    embeds = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    tsne = TSNE(n_components=2).fit_transform(embeds)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None)
    plt.show()

def visualize_similarity(model, data):
    embeds = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    cos_sim = np.dot(embeds, embeds.T) / (np.linalg.norm(embeds, axis=1)[:, np.newaxis] * np.linalg.norm(embeds, axis=1))

    plt.figure(figsize=(8, 8))
    plt.imshow(cos_sim, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Cosine Similarity Matrix')
    plt.show()

def visualize_distribution(model, data):
    embeds = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    plt.figure(figsize=(8, 8))
    plt.hist(embeds.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Embedding Value Distribution')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.show()

def visualize_lr_schedule(opt, scheduler, epochs):
    lr_hist = []
    for _ in range(epochs):
        lr_hist.append(opt.learning_rate.numpy())
        scheduler.step()
    plt.figure(figsize=(10, 5))
    plt.plot(lr_hist, label='Learning Rate', color='red')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()

def visualize_clusters(model, data, labels, n_clusters=5):
    embeds = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeds)

    plt.figure(figsize=(8, 8))
    plt.scatter(embeds[:, 0], embeds[:, 1], c=cluster_labels, cmap='viridis')
    plt.colorbar()
    plt.title('Embedding Clusters')
    plt.show()

def visualize_histogram(model, data):
    embeds = model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()
    plt.figure(figsize=(8, 8))
    plt.hist(embeds.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Embedding Value Histogram')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    data, labels = generate_data(100)
    generator = TripletGenerator(data, labels, 5)
    loader = tf.data.Dataset.from_generator(lambda: generator, output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    ).batch(32).shuffle(1000)
    model = EmbeddingModel(101, 10)
    opt = optimizers.Adam(1e-4)
    lr_scheduler = optimizers.schedules.ExponentialDecay(1e-4, decay_steps=30, decay_rate=0.1)
    loss_hist = train_model(model, loader, 10, 1e-4)
    save_model(model, "embedding_model.h5")
    plot_loss(loss_hist)
    evaluate_model(model, data, labels)
    visualize_embeddings(load_model(EmbeddingModel, "embedding_model.h5", 101, 10), *generate_data(100))
    visualize_similarity(model, data)
    visualize_distribution(model, data)
    visualize_lr_schedule(opt, lr_scheduler, 10)
    visualize_clusters(model, data, labels)
    visualize_histogram(model, data)