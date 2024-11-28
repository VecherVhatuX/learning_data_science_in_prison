import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Flatten, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

class TripletModel(Model):
    def __init__(self, embedding_dim, num_features):
        super(TripletModel, self).__init__()
        self.embedding = Embedding(embedding_dim, num_features, input_length=10)
        self.avg_pool = GlobalAveragePooling1D()
        self.flatten = Flatten()
        self.linear = Dense(num_features)
        self.batch_norm = BatchNormalization()
        self.instance_norm = LayerNormalization()

    def call(self, x):
        x = self.embedding(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.instance_norm(x)
        return x


class TripletLoss(tf.keras.layers.Layer):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, anchor, positive, negative):
        d_ap = tf.norm(anchor - positive, axis=-1)
        d_an = tf.norm(anchor[:, None] - negative, axis=-1)
        loss = tf.maximum(d_ap - tf.reduce_min(d_an, axis=-1) + self.margin, 0.0)
        return tf.reduce_mean(loss)


class TripletData(tf.data.Dataset):
    def __init__(self, samples, labels, num_negatives, batch_size, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.samples))

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        anchor_idx = np.random.choice(batch_indices, size=self.batch_size, replace=False)
        anchor_label = self.labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_label])
        return {
            'anchor': self.samples[anchor_idx],
            'positive': self.samples[positive_idx],
            'negative': self.samples[negative_idx]
        }


class InputData(tf.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]


def save_checkpoint(model, path):
    model.save_weights(path)


def load_checkpoint(model, path):
    model.load_weights(path)


def distance(embedding1, embedding2):
    return tf.norm(embedding1 - embedding2, axis=-1)


def similarity(embedding1, embedding2):
    return tf.reduce_sum(embedding1 * embedding2, axis=-1) / (tf.norm(embedding1, axis=-1) * tf.norm(embedding2, axis=-1))


def cosine_distance(embedding1, embedding2):
    return 1 - similarity(embedding1, embedding2)


def nearest_neighbors(embeddings, target_embedding, k=5):
    distances = distance(embeddings, target_embedding)
    return tf.argsort(distances)[:k]


def similar_embeddings(embeddings, target_embedding, k=5):
    similarities = similarity(embeddings, target_embedding)
    return tf.argsort(-similarities)[:k]


def knn_accuracy(embeddings, labels, k=5):
    correct = 0
    for i in range(len(embeddings)):
        distances = distance(embeddings, embeddings[i])
        indices = tf.argsort(distances)[1:k+1]
        if labels[i] in labels[indices]:
            correct += 1
    return correct / len(embeddings)


def knn_precision(embeddings, labels, k=5):
    precision = 0
    for i in range(len(embeddings)):
        distances = distance(embeddings, embeddings[i])
        indices = tf.argsort(distances)[1:k+1]
        precision += len(tf.where(labels[indices] == labels[i])) / k
    return precision / len(embeddings)


def knn_recall(embeddings, labels, k=5):
    recall = 0
    for i in range(len(embeddings)):
        distances = distance(embeddings, embeddings[i])
        indices = tf.argsort(distances)[1:k+1]
        recall += len(tf.where(labels[indices] == labels[i])) / len(tf.where(labels == labels[i]))
    return recall / len(embeddings)


def knn_f1(embeddings, labels, k=5):
    precision = knn_precision(embeddings, labels, k)
    recall = knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)


def train(model, dataset, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            anchor_embeddings = model(batch['anchor'])
            positive_embeddings = model(batch['positive'])
            negative_embeddings = model(batch['negative'])
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            optimizer.minimize(loss, model.trainable_variables)
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')


def evaluate(model, dataset):
    return model.predict(dataset.samples)


def build_model(embedding_dim, num_features):
    return TripletModel(embedding_dim, num_features)


def build_criterion(margin=1.0):
    return TripletLoss(margin)


def build_optimizer(model, learning_rate):
    return Adam(learning_rate=learning_rate)


def build_dataset(samples, labels, num_negatives, batch_size, shuffle=True):
    return TripletData(samples, labels, num_negatives, batch_size, shuffle)


def build_input_dataset(input_ids):
    return InputData(input_ids)


def main():
    np.random.seed(42)

    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    learning_rate = 1e-4
    embedding_dim = 101
    num_features = 10

    model = build_model(embedding_dim, num_features)
    criterion = build_criterion()
    optimizer = build_optimizer(model, learning_rate)
    dataset = build_dataset(samples, labels, num_negatives, batch_size)

    train(model, dataset, criterion, optimizer, epochs)

    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).reshape((1, 10))
    output = model.predict(input_ids)

    save_checkpoint(model, "triplet_model.h5")

    predicted_embeddings = evaluate(model, dataset)
    print(predicted_embeddings)

    distance = distance(output, output)
    print(distance)

    similarity = similarity(output, output)
    print(similarity)

    cosine_distance = cosine_distance(output, output)
    print(cosine_distance)

    all_embeddings = model.predict(dataset.samples)
    nearest_neighbors = nearest_neighbors(all_embeddings, output, k=5)
    print(nearest_neighbors)

    similar_embeddings = similar_embeddings(all_embeddings, output, k=5)
    print(similar_embeddings)

    print("KNN Accuracy:", knn_accuracy(all_embeddings, labels, k=5))

    print("KNN Precision:", knn_precision(all_embeddings, labels, k=5))

    print("KNN Recall:", knn_recall(all_embeddings, labels, k=5))

    print("KNN F1-score:", knn_f1(all_embeddings, labels, k=5))


if __name__ == "__main__":
    main()