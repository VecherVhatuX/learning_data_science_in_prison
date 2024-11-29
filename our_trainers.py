import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class TripletModel(tf.keras.Model):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(embedding_dim, num_features)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_features)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.layer_norm(x)
        return x

class TripletLoss(tf.keras.losses.Loss):
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
    def __init__(self, samples, labels, num_negatives, batch_size, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(samples))
        self.on_epoch_end()

    def __len__(self):
        return len(self.samples) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.anchor_idx = np.random.choice(self.indices, size=self.batch_size, replace=False)
        self.anchor_label = self.labels[self.anchor_idx]
        self.positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in self.anchor_label])
        self.negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in self.anchor_label])

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.samples[batch_indices], self.samples[self.positive_idx], self.samples[self.negative_idx]

def train(model, dataset, epochs, optimizer):
    loss_fn = TripletLoss()
    for epoch in range(epochs):
        dataset.on_epoch_end()
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

def build_optimizer(model, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features):
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    model = TripletModel(embedding_dim, num_features)
    optimizer = build_optimizer(model, learning_rate)
    dataset = TripletDataset(samples, labels, num_negatives, batch_size)
    train(model, dataset, epochs, optimizer)
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