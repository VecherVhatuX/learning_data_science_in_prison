import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers

Embedder = lambda emb_dim, feat_dim: tf.keras.Sequential([
    layers.Embedding(emb_dim, feat_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dense(feat_dim),
    layers.BatchNormalization(),
    layers.LayerNormalization()
])

create_random_data = lambda size: (np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, size)

generate_triplet_dataset = lambda samples, labels, neg_count: tf.data.Dataset.from_tensor_slices((samples, labels)).map(
    lambda anchor, label: (
        anchor,
        tf.convert_to_tensor(random.choice(samples[labels == label.numpy()]), dtype=tf.float32),
        tf.convert_to_tensor(random.sample(samples[labels != label.numpy()].tolist(), neg_count), dtype=tf.float32)
    )
)

create_triplet_loss = lambda margin=1.0: lambda anchor, positive, negative: tf.reduce_mean(
    tf.maximum(tf.norm(anchor - positive, axis=1) - tf.reduce_min(tf.norm(anchor[:, tf.newaxis] - negative, axis=2), axis=1) + margin, 0.0
)

train_model = lambda model, dataset, epochs, learning_rate: (
    lambda optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss_fn=create_triplet_loss(): [
        (lambda epoch_loss=0: [
            (lambda loss=loss_fn(anchors, positives, negatives): [
                optimizer.apply_gradients(zip(tf.GradientTape().gradient(loss, model.trainable_variables), model.trainable_variables)),
                epoch_loss.__setitem__(0, epoch_loss + loss.numpy())
            ])(*batch) for batch in dataset.batch(32)
        ]() or losses.append(epoch_loss / len(dataset)) for _ in range(epochs)
    ]() or losses
)(losses=[])

assess_model = lambda model, samples, labels, k=5: (
    model.evaluate(samples, labels),
    (lambda embeddings=model(tf.convert_to_tensor(samples, dtype=tf.int32)).numpy(): (
        (lambda metrics=compute_knn_metrics(embeddings, labels, k): (
            print_metrics(metrics),
            visualize_embeddings(embeddings, labels)
        ))()
    ))()
)

print_metrics = lambda metrics: (
    print(f"Accuracy: {metrics[0]:.4f}"),
    print(f"Precision: {metrics[1]:.4f}"),
    print(f"Recall: {metrics[2]:.4f}"),
    print(f"F1-score: {metrics[3]:.4f}")
)

save_model_weights = lambda model, filepath: model.save_weights(filepath)

load_model_weights = lambda model_class, filepath: (lambda model=model_class(): model.load_weights(filepath) or model)()

extract_embeddings = lambda model, input_data: model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()

visualize_embeddings = lambda embeddings, labels: (
    (lambda reduced_data=TSNE(n_components=2).fit_transform(embeddings): (
        plt.figure(figsize=(8, 8)),
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis'),
        plt.colorbar(),
        plt.show()
    )
)()

compute_knn_metrics = lambda embeddings, labels, k=5: (
    (lambda dist_matrix=np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), nearest_indices=np.argsort(dist_matrix, axis=1)[:, 1:k + 1]: (
        (lambda true_positives=np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1): (
            np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1)),
            np.mean(true_positives / k),
            np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)),
            2 * (np.mean(true_positives / k) * np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)) / (np.mean(true_positives / k) + np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))) if (np.mean(true_positives / k) + np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)) > 0 else 0
        ))()
    ))()
)

plot_loss = lambda loss_history: (
    plt.figure(figsize=(10, 5)),
    plt.plot(loss_history, label='Loss', color='blue'),
    plt.title('Training Loss Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Loss'),
    plt.legend(),
    plt.show()
)

run_training_pipeline = lambda learning_rate, batch_size, epochs, neg_count, emb_dim, feat_dim, data_size: (
    (lambda samples, labels=create_random_data(data_size): (
        (lambda triplet_data=generate_triplet_dataset(samples, labels, neg_count), model=Embedder(emb_dim, feat_dim): (
            (lambda loss_history=train_model(model, triplet_data, epochs, learning_rate): (
                save_model_weights(model, "triplet_model.h5"),
                plot_loss(loss_history),
                assess_model(model, samples, labels)
            ))()
        ))()
    ))()
)

show_model_summary = lambda model: model.summary()

if __name__ == "__main__":
    run_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    show_model_summary(Embedder(101, 10))