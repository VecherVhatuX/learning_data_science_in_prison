import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class TripletModel(tf.keras.Model):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(embedding_dim, num_features)
        self.avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_features)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.avg_pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        return x

class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, anchor, positive, negative):
        d_ap = tf.norm(anchor - positive, axis=-1)
        d_an = tf.norm(anchor[:, tf.newaxis, :] - negative, axis=-1)
        loss = tf.maximum(d_ap - tf.reduce_min(d_an, axis=-1) + self.margin, tf.zeros_like(d_ap))
        return tf.reduce_mean(loss)

def train(model, dataset, epochs, optimizer, criterion):
    for epoch in range(epochs):
        for batch in dataset:
            anchor, positive, negative = batch
            with tf.GradientTape() as tape:
                anchor_embeddings = model(anchor)
                positive_embeddings = model(positive)
                negative_embeddings = model(negative)
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f'Epoch: {epoch+1}, Loss: {loss.numpy()}')

def validate(model, embeddings, labels, k=5):
    predicted_embeddings = model(embeddings)
    print("Validation KNN Accuracy:", knn_accuracy(predicted_embeddings, labels, k))
    print("Validation KNN Precision:", knn_precision(predicted_embeddings, labels, k))
    print("Validation KNN Recall:", knn_recall(predicted_embeddings, labels, k))
    print("Validation KNN F1-score:", knn_f1(predicted_embeddings, labels, k))
    embedding_visualization(predicted_embeddings, labels)

def distance(embedding1, embedding2):
    return tf.norm(embedding1 - embedding2, axis=-1)

def similarity(embedding1, embedding2):
    return tf.reduce_sum(embedding1 * embedding2, axis=-1) / (tf.norm(embedding1, axis=-1) * tf.norm(embedding2, axis=-1))

def cosine_distance(embedding1, embedding2):
    return 1 - similarity(embedding1, embedding2)

def nearest_neighbors(embeddings, target_embedding, k=5):
    distances = distance(embeddings, target_embedding)
    return tf.argsort(distances, direction='ASCENDING')[:k]

def similar_embeddings(embeddings, target_embedding, k=5):
    similarities = similarity(embeddings, target_embedding)
    return tf.argsort(similarities, direction='DESCENDING')[:k]

def embedding_visualization(embeddings, labels):
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings.numpy())
    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
    plt.show()

def knn_accuracy(embeddings, labels, k=5):
    return tf.reduce_mean(tf.cast(tf.reduce_any(tf.equal(labels[tf.argsort(distance(embeddings, embeddings), axis=1)[:, 1:k+1]], labels[:, tf.newaxis]), tf.float32), axis=1))

def knn_precision(embeddings, labels, k=5):
    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(labels[tf.argsort(distance(embeddings, embeddings), axis=1)[:, 1:k+1]], labels[:, tf.newaxis]), tf.float32), axis=1) / k)

def knn_recall(embeddings, labels, k=5):
    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(labels[tf.argsort(distance(embeddings, embeddings), axis=1)[:, 1:k+1]], labels[:, tf.newaxis]), tf.float32), axis=1) / tf.reduce_sum(tf.cast(tf.equal(labels, labels[:, tf.newaxis]), tf.float32), axis=1))

def knn_f1(embeddings, labels, k=5):
    precision = knn_precision(embeddings, labels, k)
    recall = knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)

def build_dataset(samples, labels, num_negatives, batch_size, shuffle=True):
    dataset = []
    indices = np.arange(len(samples))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(len(samples) // batch_size):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        anchor_idx = np.random.choice(batch_indices, size=batch_size, replace=False)
        anchor_label = labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_label])
        dataset.append((samples[anchor_idx], samples[positive_idx], samples[negative_idx]))
    return dataset

def build_optimizer(model, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features):
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    model = TripletModel(embedding_dim, num_features)
    criterion = TripletLoss()
    optimizer = build_optimizer(model, learning_rate)
    dataset = build_dataset(samples, labels, num_negatives, batch_size)
    train(model, dataset, epochs, optimizer, criterion)
    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).reshape((1, 10))
    output = model(input_ids)
    model.save_weights("triplet_model.h5")
    predicted_embeddings = model(samples)
    print(distance(output, output).numpy())
    print(similarity(output, output).numpy())
    print(cosine_distance(output, output).numpy())
    print(nearest_neighbors(predicted_embeddings, output, k=5).numpy())
    print(similar_embeddings(predicted_embeddings, output, k=5).numpy())
    validate(model, predicted_embeddings, labels, k=5)

def main():
    np.random.seed(42)
    pipeline(1e-4, 32, 10, 5, 101, 10)

if __name__ == "__main__":
    main()