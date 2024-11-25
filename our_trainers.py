from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Reshape, GlobalAveragePooling1D, Dense, Lambda, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class EmbeddingModel(Model):
    def __init__(self, embedding_dim, num_features):
        super(EmbeddingModel, self).__init__()
        self.model = Embedding(embedding_dim, num_features, input_length=10)
        self.reshape = Reshape((-1, num_features))
        self.average_pooling = GlobalAveragePooling1D()
        self.flatten = Lambda(lambda x: x)
        self.dense = Dense(num_features)
        self.batch_norm = BatchNormalization()
        self.l2_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))

    def call(self, x):
        x = self.model(x)
        x = self.reshape(x)
        x = self.average_pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.l2_norm(x)
        return x

class TripletLoss(Loss):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def call(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(anchor_embeddings - positive_embeddings), axis=-1)) - K.min(K.sqrt(K.sum(K.square(K.expand_dims(anchor_embeddings, axis=1) - negative_embeddings), axis=-1)), axis=1) + 1.0, 0.0))

class InputDataset:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]

class TripletDataset:
    def __init__(self, samples, labels, num_negatives, batch_size, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.samples))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        anchor_idx = np.random.choice(batch_indices, size=self.batch_size)
        anchor_label = self.labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_label])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_label])
        return {
            'anchor': self.samples[anchor_idx],
            'positive': self.samples[positive_idx],
            'negative': self.samples[negative_idx]
        }

def save_model(model, path):
    model.save(path)

def load_model(model, path):
    model.load_weights(path)

def calculate_distance(embedding1, embedding2):
    return K.sqrt(K.sum(K.square(embedding1 - embedding2), axis=-1))

def calculate_similarity(embedding1, embedding2):
    return K.sum(embedding1 * embedding2, axis=-1) / (K.sqrt(K.sum(K.square(embedding1), axis=-1)) * K.sqrt(K.sum(K.square(embedding2), axis=-1)))

def calculate_cosine_distance(embedding1, embedding2):
    return 1 - calculate_similarity(embedding1, embedding2)

def get_nearest_neighbors(embeddings, target_embedding, k=5):
    distances = calculate_distance(embeddings, target_embedding)
    return np.argsort(distances)[:k]

def get_similar_embeddings(embeddings, target_embedding, k=5):
    similarities = calculate_similarity(embeddings, target_embedding)
    return np.argsort(-similarities)[:k]

def calculate_knn_accuracy(embeddings, labels, k=5):
    correct = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = np.argsort(distances)[1:k+1]
        if labels[i] in labels[indices]:
            correct += 1
    return correct / len(embeddings)

def calculate_knn_precision(embeddings, labels, k=5):
    precision = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = np.argsort(distances)[1:k+1]
        precision += len(np.where(labels[indices] == labels[i])[0]) / k
    return precision / len(embeddings)

def calculate_knn_recall(embeddings, labels, k=5):
    recall = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = np.argsort(distances)[1:k+1]
        recall += len(np.where(labels[indices] == labels[i])[0]) / len(np.where(labels == labels[i])[0])
    return recall / len(embeddings)

def calculate_knn_f1(embeddings, labels, k=5):
    precision = calculate_knn_precision(embeddings, labels, k)
    recall = calculate_knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)

def main():
    np.random.seed(42)

    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    learning_rate = 1e-4

    model = EmbeddingModel(101, 10)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=TripletLoss())
    dataset = TripletDataset(samples, labels, num_negatives, batch_size)

    model.fit(dataset, epochs=epochs)

    input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10))
    input_dataset = InputDataset(input_ids)
    output = model.predict(input_dataset)

    save_model(model, "triplet_model.h5")

    predicted_embeddings = model.predict(dataset)
    print(predicted_embeddings)

    distance = calculate_distance(output, output)
    print(distance)

    similarity = calculate_similarity(output, output)
    print(similarity)

    cosine_distance = calculate_cosine_distance(output, output)
    print(cosine_distance)

    all_embeddings = model.predict(dataset)
    nearest_neighbors = get_nearest_neighbors(all_embeddings, output, k=5)
    print(nearest_neighbors)

    similar_embeddings = get_similar_embeddings(all_embeddings, output, k=5)
    print(similar_embeddings)

    print("KNN Accuracy:", calculate_knn_accuracy(all_embeddings, labels, k=5))

    print("KNN Precision:", calculate_knn_precision(all_embeddings, labels, k=5))

    print("KNN Recall:", calculate_knn_recall(all_embeddings, labels, k=5))

    print("KNN F1-score:", calculate_knn_f1(all_embeddings, labels, k=5))

if __name__ == "__main__":
    main()