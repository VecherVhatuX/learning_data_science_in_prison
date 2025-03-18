import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tensorflow.keras import layers, optimizers, losses

WordVectorGenerator = lambda vocab_size, embedding_size: tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_size),
    layers.GlobalAveragePooling1D(),
    layers.Dense(embedding_size),
    layers.BatchNormalization(),
    layers.LayerNormalization()
])

TripletDataLoader = lambda data, labels, negative_samples: tf.data.Dataset.from_generator(
    lambda: ((anchor, positive, negatives) for anchor, positive, negatives in [
        (tf.convert_to_tensor(data[i], dtype=tf.int32),
        tf.convert_to_tensor(random.choice(data[labels == labels[i]]), dtype=tf.int32),
        tf.convert_to_tensor(random.sample(data[labels != labels[i]].tolist(), negative_samples), dtype=tf.int32)
    ] for i in range(len(data))),
    output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

calculate_triplet_loss = lambda anchor, positive, negative, margin=1.0: tf.reduce_mean(
    tf.maximum(tf.norm(anchor - positive, axis=1) - tf.reduce_min(tf.norm(tf.expand_dims(anchor, 1) - negative, axis=2), axis=1) + margin, 0.0)
)

train_vector_generator = lambda model, dataset, num_epochs, learning_rate: (
    lambda optimizer=tf.optimizers.Adam(learning_rate): [
        optimizer.apply_gradients(zip(
            tf.GradientTape().gradient(
                calculate_triplet_loss(model(anchor), model(positive), model(negative)),
                model.trainable_variables
            ),
            model.trainable_variables
        )) for anchor, positive, negative in dataset
    ] for _ in range(num_epochs)
)

evaluate_model = lambda model, data, labels, k=5: (
    show_metrics(compute_performance_metrics(model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy(), labels, k)),
    plot_embeddings(model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy(), labels)
)

show_metrics = lambda metrics: (
    print(f"Accuracy: {metrics[0]:.4f}"),
    print(f"Precision: {metrics[1]:.4f}"),
    print(f"Recall: {metrics[2]:.4f}"),
    print(f"F1-score: {metrics[3]:.4f}")
)

save_model = lambda model, file_path: model.save_weights(file_path)

load_model = lambda model_class, file_path, vocab_size, embedding_size: (
    lambda model=model_class(vocab_size, embedding_size): model.load_weights(file_path) or model
)

generate_embeddings = lambda model, data: model(tf.convert_to_tensor(data, dtype=tf.int32)).numpy()

plot_embeddings = lambda embeddings, labels: (
    plt.figure(figsize=(8, 8)),
    plt.scatter(TSNE(n_components=2).fit_transform(embeddings)[:, 0], TSNE(n_components=2).fit_transform(embeddings)[:, 1], c=labels, cmap='viridis'),
    plt.colorbar(),
    plt.show()
)

compute_performance_metrics = lambda embeddings, labels, k=5: (
    lambda distances=np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2), nearest_neighbors=np.argsort(distances, axis=1)[:, 1:k + 1], true_positives=np.sum(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1): (
        np.mean(np.any(labels[nearest_neighbors] == labels[:, np.newaxis], axis=1)),
        np.mean(true_positives / k),
        np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1)),
        2 * (np.mean(true_positives / k) * np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))) / (np.mean(true_positives / k) + np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))) if (np.mean(true_positives / k) + np.mean(true_positives / np.sum(labels == labels[:, np.newaxis], axis=1))) > 0 else 0
    )
)

plot_loss_history = lambda loss_history: (
    plt.figure(figsize=(10, 5)),
    plt.plot(loss_history, label='Loss', color='blue'),
    plt.title('Training Loss Over Epochs'),
    plt.xlabel('Epochs'),
    plt.ylabel('Loss'),
    plt.legend(),
    plt.show()
)

run_training = lambda learning_rate, batch_size, num_epochs, negative_samples, vocab_size, embedding_size, data_size: (
    lambda data, labels=generate_data(data_size): (
        save_model(WordVectorGenerator(vocab_size, embedding_size), "word_vector_generator.h5"),
        plot_loss_history(train_vector_generator(WordVectorGenerator(vocab_size, embedding_size), TripletDataLoader(data, labels, negative_samples), num_epochs, learning_rate)),
        evaluate_model(WordVectorGenerator(vocab_size, embedding_size), data, labels)
    )
)

display_model_architecture = lambda model: model.summary()

train_with_early_termination = lambda model, dataset, num_epochs, learning_rate, patience=5: (
    lambda optimizer=tf.optimizers.Adam(learning_rate), best_loss=float('inf'), no_improvement=0: [
        (lambda avg_loss=sum(calculate_triplet_loss(model(anchor), model(positive), model(negative)) for anchor, positive, negative in dataset) / len(dataset): (
            optimizer.apply_gradients(zip(tf.GradientTape().gradient(avg_loss, model.trainable_variables), model.trainable_variables)),
            best_loss if avg_loss >= best_loss else (no_improvement := 0, best_loss := avg_loss),
            no_improvement := no_improvement + 1 if avg_loss >= best_loss else no_improvement,
            print(f"Early stopping at epoch {epoch}") if no_improvement >= patience else None
        )) for epoch in range(num_epochs)
    ]
)

generate_data = lambda data_size: (
    np.random.randint(0, 100, (data_size, 10)),
    np.random.randint(0, 10, data_size)
)

visualize_embeddings_interactive = lambda model, data, labels: (
    lambda embeddings=generate_embeddings(model, data), tsne_result=TSNE(n_components=2).fit_transform(embeddings): (
        plt.figure(figsize=(8, 8)),
        scatter=plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis'),
        plt.colorbar(scatter),
        plt.gcf().canvas.mpl_connect('button_press_event', lambda event: (
            print(f"Clicked on point with label: {labels[np.argmin(np.linalg.norm(tsne_result - np.array([event.xdata, event.ydata]), axis=1))]}") if event.inaxes is not None else None
        )),
        plt.show()
    )
)

add_custom_regularization = lambda model, lambda_reg=0.01: lambda_reg * sum(tf.norm(param, ord=2) for param in model.trainable_variables)

add_learning_rate_scheduler = lambda optimizer, step_size=30, gamma=0.1: optimizers.schedules.ExponentialDecay(initial_learning_rate=optimizer.learning_rate, decay_steps=step_size, decay_rate=gamma)

if __name__ == "__main__":
    run_training(1e-4, 32, 10, 5, 101, 10, 100)
    display_model_architecture(WordVectorGenerator(101, 10))
    visualize_embeddings_interactive(load_model(WordVectorGenerator, "word_vector_generator.h5", 101, 10), *generate_data(100))