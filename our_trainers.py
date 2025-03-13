import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers

create_embedding_model = lambda embedding_dim, feature_dim: tf.keras.Sequential([
    layers.Embedding(embedding_dim, feature_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dense(feature_dim),
    layers.BatchNormalization(),
    layers.LayerNormalization()
])

generate_random_dataset = lambda size: (np.random.randint(0, 100, (size, 10)), np.random.randint(0, 2, size))

create_triplet_data = lambda samples, labels, negative_count: tf.data.Dataset.from_tensor_slices((samples, labels)).map(
    lambda anchor, label: (
        anchor,
        tf.convert_to_tensor(random.choice(samples[labels == label.numpy()])),
        tf.convert_to_tensor(random.sample(samples[labels != label.numpy()].tolist(), negative_count))
    )
)

triplet_loss_function = lambda margin=1.0: lambda anchor, positive, negative: tf.reduce_mean(
    tf.maximum(tf.norm(anchor - positive, axis=1) - tf.reduce_min(tf.norm(anchor[:, tf.newaxis] - negative, axis=2), axis=1) + margin, 0.0)
)

train_embedding_model = lambda model, dataset, epochs, learning_rate: (
    lambda optimizer, loss_fn: [
        (lambda epoch_loss: [
            (lambda anchors, positives, negatives: (
                (lambda loss: optimizer.apply_gradients(zip(tf.GradientTape().gradient(loss, model.trainable_variables), model.trainable_variables))(
                    loss_fn(anchors, positives, negatives)
                )
            )(*batch) for batch in dataset.batch(32)
        ] and loss_history.append(epoch_loss / len(dataset)) for _ in range(epochs)
    ])(tf.keras.optimizers.Adam(learning_rate), triplet_loss_function()), []

evaluate_model = lambda model, samples, labels, k=5: (
    lambda embeddings: (
        display_metrics(calculate_knn_metrics(embeddings, labels, k)),
        plot_embeddings(embeddings, labels)
    )
)(model(tf.convert_to_tensor(samples, dtype=tf.int32)).numpy())

display_metrics = lambda metrics: (
    print(f"Accuracy: {metrics[0]:.4f}"),
    print(f"Precision: {metrics[1]:.4f}"),
    print(f"Recall: {metrics[2]:.4f}"),
    print(f"F1-score: {metrics[3]:.4f}")
)

save_model = lambda model, filepath: model.save_weights(filepath)

load_model = lambda model_class, filepath: (lambda model: model.load_weights(filepath) or model)(model_class())

get_embeddings = lambda model, input_data: model(tf.convert_to_tensor(input_data, dtype=tf.int32)).numpy()

plot_embeddings = lambda embeddings, labels: (
    plt.figure(figsize=(8, 8)),
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis'),
    plt.colorbar(),
    plt.show()
)

calculate_knn_metrics = lambda embeddings, labels, k=5: (
    lambda dist_matrix, nearest_indices, true_positives: (
        np.mean(np.any(labels[nearest_indices] == labels[:, np.newaxis], axis=1)),
        np.mean(true_positives / k),
        np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)),
        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    )
)(np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), np.argsort(dist_matrix, axis=1)[:, 1:k + 1], np.sum(labels[nearest_indices] == labels[:, np.newaxis], axis=1))

plot_training_loss = lambda loss_history: (
    plt.figure(figsize=(10, 5)),
    plt.plot(loss_history, label='Loss', color='blue'),
    plt.title('Training Loss Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Loss'),
    plt.legend(),
    plt.show()
)

execute_training_pipeline = lambda learning_rate, batch_size, epochs, negative_count, embedding_dim, feature_dim, data_size: (
    lambda samples, labels, triplet_data, model: (
        save_model(model, "triplet_model.h5"),
        plot_training_loss(train_embedding_model(model, triplet_data, epochs, learning_rate)),
        evaluate_model(model, samples, labels)
    )
)(*generate_random_dataset(data_size), create_triplet_data(samples, labels, negative_count), create_embedding_model(embedding_dim, feature_dim))

display_model_summary = lambda model: model.summary()

if __name__ == "__main__":
    execute_training_pipeline(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_summary(create_embedding_model(101, 10))