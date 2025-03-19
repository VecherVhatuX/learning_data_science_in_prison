import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, models, optimizers, losses

EmbeddingModel = lambda vocab_size, embed_dim: models.Sequential([
    layers.Embedding(vocab_size, embed_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dense(embed_dim),
    layers.BatchNormalization(),
    layers.LayerNormalization()
])

TripletData = lambda data, labels, neg_samples: type('TripletData', (tf.keras.utils.Sequence,), {
    '__init__': lambda self, data, labels, neg_samples: (setattr(self, 'data', data), setattr(self, 'labels', labels), setattr(self, 'neg_samples', neg_samples),
    '__len__': lambda self: len(self.data),
    '__getitem__': lambda self, idx: (
        self.data[idx],
        random.choice(self.data[self.labels == self.labels[idx]]),
        random.sample(self.data[self.labels != self.labels[idx]].tolist(), self.neg_samples)
    )
})(data, labels, neg_samples)

triplet_loss = lambda anchor, pos, neg, margin=1.0: tf.reduce_mean(tf.maximum(
    tf.norm(anchor - pos, axis=1) - tf.reduce_min(tf.norm(tf.expand_dims(anchor, 1) - neg, axis=2), axis=1) + margin, 0.0
)

train_model = lambda model, loader, epochs, lr: (
    lambda optimizer, scheduler: [
        (lambda epoch_loss: [
            (lambda loss: (loss.backward(), optimizer.apply_gradients(zip(loss.gradients, model.trainable_variables)), epoch_loss.append(loss.numpy())))(
                triplet_loss(model(anchor), model(pos), model(neg)) + 0.01 * sum(tf.norm(p, ord=2) for p in model.trainable_variables
            for anchor, pos, neg in loader
        ], scheduler.step(), losses.append(np.mean(epoch_loss))
        for _ in range(epochs)
    ](optimizers.Adam(lr), optimizers.schedules.ExponentialDecay(lr, decay_steps=30, decay_rate=0.1), []
)[2]

evaluate = lambda model, data, labels, k=5: (
    lambda embeddings, distances, neighbors, true_positives: (
        print(f"Accuracy: {np.mean(np.any(labels[neighbors] == labels[:, np.newaxis], axis=1)):.4f}"),
        print(f"Precision: {np.mean(true_positives / k):.4f}"),
        print(f"Recall: {np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)):.4f}"),
        print(f"F1-score: {2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0:.4f}"),
        plt.figure(figsize=(8, 8)),
        plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis'),
        plt.colorbar(),
        plt.show()
    )
)(model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy(), np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), np.argsort(distances, axis=1)[:, 1:k+1], np.sum(labels[neighbors] == labels[:, np.newaxis], axis=1))

save = lambda model, path: model.save_weights(path)

load = lambda model_class, path, vocab_size, embed_dim: (lambda model: (model.load_weights(path), model)[1])(model_class(vocab_size, embed_dim))

plot_loss_history = lambda losses: (plt.figure(figsize=(10, 5)), plt.plot(losses, label='Loss', color='blue'), plt.title('Training Loss Over Epochs'), plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.legend(), plt.show())

generate_data = lambda data_size: (np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 10, data_size)

visualize = lambda model, data, labels: (
    lambda embeddings, tsne: (
        plt.figure(figsize=(8, 8)),
        plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis'),
        plt.colorbar(),
        plt.gcf().canvas.mpl_connect('button_press_event', lambda event: print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None),
        plt.show()
    )
)(model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy(), TSNE(n_components=2).fit_transform(embeddings))

display_similarity = lambda model, data: (
    lambda embeddings, cosine_sim: (
        plt.figure(figsize=(8, 8)),
        plt.imshow(cosine_sim, cmap='viridis', vmin=0, vmax=1),
        plt.colorbar(),
        plt.title('Cosine Similarity Matrix'),
        plt.show()
    )
)(model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy(), np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings, axis=1)))

plot_distribution = lambda model, data: (
    lambda embeddings: (
        plt.figure(figsize=(8, 8)),
        plt.hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7),
        plt.title('Embedding Value Distribution'),
        plt.xlabel('Embedding Value'),
        plt.ylabel('Frequency'),
        plt.show()
    )
)(model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy())

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