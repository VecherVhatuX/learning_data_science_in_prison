import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class TripletModel(keras.Model):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(embedding_dim, num_features)
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_features)
        self.batch_normalization = layers.BatchNormalization()
        self.layer_normalization = layers.LayerNormalization()

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.global_average_pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.batch_normalization(x)
        x = self.layer_normalization(x)
        return x

class TripletLoss(keras.losses.Loss):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        anchor, positive, negative = y_pred
        d_ap = tf.norm(anchor - positive, axis=-1)
        d_an = tf.norm(anchor[:, tf.newaxis, :] - negative, axis=-1)
        loss = tf.maximum(d_ap - tf.reduce_min(d_an, axis=-1) + self.margin, tf.zeros_like(d_ap))
        return tf.reduce_mean(loss)

class TripletDataset(tf.keras.utils.Sequence):
    def __init__(self, samples, labels, num_negatives, batch_size):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        anchor_idx = np.random.choice(len(self.samples), size=self.batch_size, replace=False)
        anchor_label = self.labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_label])
        return self.samples[anchor_idx], self.samples[positive_idx], self.samples[negative_idx]

def train(model, dataset, epochs, optimizer):
    loss_fn = TripletLoss()
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                anchor, positive, negative = batch
                anchor_embeddings = model(anchor)
                positive_embeddings = model(positive)
                negative_embeddings = model(negative)
                loss = loss_fn(None, (anchor_embeddings, positive_embeddings, negative_embeddings))
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

def calculate_distances(output):
    print(tf.norm(output - output, axis=-1).numpy())
    print(tf.reduce_sum(output * output, axis=-1) / (tf.norm(output, axis=-1) * tf.norm(output, axis=-1)).numpy())
    print(1 - tf.reduce_sum(output * output, axis=-1) / (tf.norm(output, axis=-1) * tf.norm(output, axis=-1)).numpy())

def calculate_neighbors(predicted_embeddings, output, k=5):
    print(tf.argsort(tf.norm(predicted_embeddings - output, axis=-1), direction='ASCENDING')[:, :k].numpy())
    print(tf.argsort(tf.reduce_sum(predicted_embeddings * output, axis=-1) / (tf.norm(predicted_embeddings, axis=-1) * tf.norm(output, axis=-1)), direction='DESCENDING')[:, :k].numpy())

def pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features, size):
    samples = np.random.randint(0, 100, (size, 10))
    labels = np.random.randint(0, 2, (size,))
    model = TripletModel(embedding_dim, num_features)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    dataset = TripletDataset(samples, labels, num_negatives, batch_size)
    train(model, dataset, epochs, optimizer)
    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).reshape((1, 10))
    output = model(input_ids)
    model.save_weights("triplet_model.h5")
    predicted_embeddings = model(samples)
    calculate_distances(output)
    calculate_neighbors(predicted_embeddings, output)
    validate(model, predicted_embeddings, labels)

def main():
    np.random.seed(42)
    pipeline(1e-4, 32, 10, 5, 101, 10, 100)

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

if __name__ == "__main__":
    main()