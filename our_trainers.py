import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Functional style model definition
def create_model(embedding_dim, num_features):
    inputs = keras.Input(shape=(10,), dtype='int32')
    x = layers.Embedding(embedding_dim, num_features)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_features)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LayerNormalization()(x)
    return keras.Model(inputs=inputs, outputs=x)

# Functional style loss function definition
def triplet_loss(y_true, y_pred, margin=1.0):
    anchor, positive, negative = y_pred
    d_ap = tf.norm(anchor - positive, axis=-1)
    d_an = tf.norm(anchor[:, tf.newaxis, :] - negative, axis=-1)
    loss = tf.maximum(d_ap - tf.reduce_min(d_an, axis=-1) + margin, tf.zeros_like(d_ap))
    return tf.reduce_mean(loss)

# Functional style dataset generator definition
def create_dataset(samples, labels, num_negatives, batch_size):
    def generator():
        while True:
            anchor_idx = np.random.choice(len(samples), size=batch_size, replace=False)
            anchor_label = labels[anchor_idx]
            positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_label])
            negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_label])
            yield samples[anchor_idx], samples[positive_idx], samples[negative_idx]
    return generator

# Functional style training function definition
def train(model, dataset, epochs, optimizer):
    loss_fn = lambda y_true, y_pred: triplet_loss(y_true, y_pred)
    for epoch in range(epochs):
        for batch in dataset():
            with tf.GradientTape() as tape:
                anchor, positive, negative = batch
                anchor_embeddings = model(anchor)
                positive_embeddings = model(positive)
                negative_embeddings = model(negative)
                loss = loss_fn(None, (anchor_embeddings, positive_embeddings, negative_embeddings))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f'Epoch: {epoch+1}, Loss: {loss.numpy()}')

# Functional style validation function definition
def validate(model, embeddings, labels, k=5):
    predicted_embeddings = model(embeddings)
    print("Validation KNN Accuracy:", knn_accuracy(predicted_embeddings, labels, k))
    print("Validation KNN Precision:", knn_precision(predicted_embeddings, labels, k))
    print("Validation KNN Recall:", knn_recall(predicted_embeddings, labels, k))
    print("Validation KNN F1-score:", knn_f1(predicted_embeddings, labels, k))
    embedding_visualization(predicted_embeddings, labels)

# Functional style distance calculation function definition
def calculate_distances(output):
    print(tf.norm(output - output, axis=-1).numpy())
    print(tf.reduce_sum(output * output, axis=-1) / (tf.norm(output, axis=-1) * tf.norm(output, axis=-1)).numpy())
    print(1 - tf.reduce_sum(output * output, axis=-1) / (tf.norm(output, axis=-1) * tf.norm(output, axis=-1)).numpy())

# Functional style nearest neighbors calculation function definition
def calculate_neighbors(predicted_embeddings, output, k=5):
    print(tf.argsort(tf.norm(predicted_embeddings - output, axis=-1), direction='ASCENDING')[:, :k].numpy())
    print(tf.argsort(tf.reduce_sum(predicted_embeddings * output, axis=-1) / (tf.norm(predicted_embeddings, axis=-1) * tf.norm(output, axis=-1)), direction='DESCENDING')[:, :k].numpy())

# Functional style pipeline definition
def pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features, size):
    samples = np.random.randint(0, 100, (size, 10))
    labels = np.random.randint(0, 2, (size,))
    model = create_model(embedding_dim, num_features)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    dataset = create_dataset(samples, labels, num_negatives, batch_size)
    train(model, dataset, epochs, optimizer)
    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).reshape((1, 10))
    output = model(input_ids)
    model.save_weights("triplet_model.h5")
    predicted_embeddings = model(samples)
    calculate_distances(output)
    calculate_neighbors(predicted_embeddings, output)
    validate(model, predicted_embeddings, labels)

# Functional style main function definition
def main():
    np.random.seed(42)
    pipeline(1e-4, 32, 10, 5, 101, 10, 100)

if __name__ == "__main__":
    main()

# Utility functions
def embedding_visualization(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings.numpy())
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
    plt.show()

def knn_accuracy(embeddings, labels, k=5):
    return tf.reduce_mean(tf.cast(tf.reduce_any(tf.equal(labels[tf.argsort(tf.norm(embeddings - embeddings, axis=1), axis=1)[:, 1:k+1]], labels[:, tf.newaxis]), tf.float32), axis=1))

def knn_precision(embeddings, labels, k=5):
    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(labels[tf.argsort(tf.norm(embeddings - embeddings, axis=1), axis=1)[:, 1:k+1]], labels[:, tf.newaxis]), tf.float32), axis=1) / k)

def knn_recall(embeddings, labels, k=5):
    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(labels[tf.argsort(tf.norm(embeddings - embeddings, axis=1), axis=1)[:, 1:k+1]], labels[:, tf.newaxis]), tf.float32), axis=1) / tf.reduce_sum(tf.cast(tf.equal(labels, labels[:, tf.newaxis]), tf.float32), axis=1))

def knn_f1(embeddings, labels, k=5):
    precision = knn_precision(embeddings, labels, k)
    recall = knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)