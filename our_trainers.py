import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers

build_embedding_network = lambda embedding_size, feature_size: tf.keras.Sequential([
    layers.Embedding(embedding_size, feature_size),
    layers.GlobalAveragePooling1D(),
    layers.Dense(feature_size),
    layers.BatchNormalization(),
    layers.LayerNormalization()
])

generate_random_data = lambda data_size: (np.random.randint(0, 100, (data_size, 10)), np.random.randint(0, 2, data_size))

create_triplet_dataset = lambda data_samples, data_labels, num_negatives: tf.data.Dataset.from_tensor_slices((data_samples, data_labels)).map(
    lambda anchor_sample, anchor_label: (
        anchor_sample,
        tf.convert_to_tensor(random.choice(data_samples[data_labels == anchor_label.numpy()])),
        tf.convert_to_tensor(random.sample(data_samples[data_labels != anchor_label.numpy()].tolist(), num_negatives))
    )
)

compute_triplet_loss = lambda margin=1.0: lambda anchor, positive, negative: tf.reduce_mean(
    tf.maximum(tf.norm(anchor - positive, axis=1) - tf.reduce_min(tf.norm(anchor[:, tf.newaxis] - negative, axis=2), axis=1) + margin, 0.0)
)

train_embedding_network = lambda network, dataset, num_epochs, lr: (
    lambda optimizer, loss_function: [
        (lambda epoch_loss: [
            (lambda anchors, positives, negatives: (
                (lambda loss: optimizer.apply_gradients(zip(tf.GradientTape().gradient(loss, network.trainable_variables), network.trainable_variables))(
                    loss_function(anchors, positives, negatives)
                )
            )(*batch) for batch in dataset.batch(32)
        ] and loss_history.append(epoch_loss / len(dataset)) for _ in range(num_epochs)
    ])(tf.keras.optimizers.Adam(lr), compute_triplet_loss()), []
)

evaluate_network = lambda network, data_samples, data_labels, k=5: (
    lambda embeddings: (
        display_performance_metrics(compute_knn_metrics(embeddings, data_labels, k)),
        visualize_embeddings(embeddings, data_labels)
    )
)(network(tf.convert_to_tensor(data_samples, dtype=tf.int32)).numpy())

display_performance_metrics = lambda metrics: (
    print(f"Accuracy: {metrics[0]:.4f}"),
    print(f"Precision: {metrics[1]:.4f}"),
    print(f"Recall: {metrics[2]:.4f}"),
    print(f"F1-score: {metrics[3]:.4f}")
)

save_network = lambda network, filepath: network.save_weights(filepath)

load_network = lambda network_class, filepath: (lambda network: network.load_weights(filepath) or network)(network_class())

extract_embeddings = lambda network, input_data: network(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()

visualize_embeddings = lambda embeddings, labels: (
    plt.figure(figsize=(8, 8)),
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis'),
    plt.colorbar(),
    plt.show()
)

compute_knn_metrics = lambda embeddings, labels, k=5: (
    lambda distance_matrix, nearest_neighbors, true_positives: (
        np.mean(np.any(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)),
        np.mean(true_positives / k),
        np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)),
        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    )
)(np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), np.argsort(distance_matrix, axis=1)[:, 1:k + 1], np.sum(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1))

plot_loss_history = lambda loss_history: (
    plt.figure(figsize=(10, 5)),
    plt.plot(loss_history, label='Loss', color='blue'),
    plt.title('Training Loss Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Loss'),
    plt.legend(),
    plt.show()
)

run_training_pipeline = lambda lr, batch_size, num_epochs, num_negatives, embedding_size, feature_size, data_size: (
    lambda data_samples, data_labels, triplet_dataset, network: (
        save_network(network, "triplet_network.h5"),
        plot_loss_history(train_embedding_network(network, triplet_dataset, num_epochs, lr)),
        evaluate_network(network, data_samples, data_labels)
    )
)(*generate_random_data(data_size), create_triplet_dataset(data_samples, data_labels, num_negatives), build_embedding_network(embedding_size, feature_size))

display_network_summary = lambda network: network.summary()

if __name__ == "__main__":
    run_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    display_network_summary(build_embedding_network(101, 10))