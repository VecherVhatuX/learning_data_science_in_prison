import tensorflow as tf
import numpy as np

# Model Building
def build_model(embedding_dim, num_features):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=embedding_dim, output_dim=num_features, input_length=10),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_features),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization()
    ])

# Loss Function
def build_criterion(margin=1.0):
    def triplet_loss(anchor, positive, negative):
        d_ap = tf.norm(anchor - positive, axis=-1)
        d_an = tf.norm(anchor[:, None] - negative, axis=-1)
        loss = tf.maximum(d_ap - tf.reduce_min(d_an, axis=-1) + margin, 0.0)
        return tf.reduce_mean(loss)
    return triplet_loss

# Optimizer
def build_optimizer(model, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Dataset
def build_dataset(samples, labels, num_negatives, batch_size, shuffle=True):
    def generate_triplets():
        indices = np.arange(len(samples))
        if shuffle:
            np.random.shuffle(indices)
        for i in range(len(samples) // batch_size):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            anchor_idx = np.random.choice(batch_indices, size=batch_size, replace=False)
            anchor_label = labels[anchor_idx]
            positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_label])
            negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_label])
            yield (samples[anchor_idx], samples[positive_idx], samples[negative_idx])
    return tf.data.Dataset.from_generator(
        generate_triplets,
        output_types=(tf.int32, tf.int32, tf.int32),
        output_shapes=(
            tf.TensorShape([batch_size, 10]),
            tf.TensorShape([batch_size, 10]),
            tf.TensorShape([batch_size, num_negatives, 10])
        )
    )

# Distance and Similarity Metrics
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

# KNN Metrics
def knn_accuracy(embeddings, labels, k=5):
    return tf.reduce_mean(tf.map_fn(lambda x: tf.reduce_any(tf.equal(labels[x[1:k+1]], labels[x[0]])), (tf.argsort(distance(embeddings, embeddings), axis=1)), tf.float32))

def knn_precision(embeddings, labels, k=5):
    return tf.reduce_mean(tf.map_fn(lambda x: tf.reduce_sum(tf.equal(labels[x[1:k+1]], labels[x[0]])) / k, (tf.argsort(distance(embeddings, embeddings), axis=1)), tf.float32))

def knn_recall(embeddings, labels, k=5):
    return tf.reduce_mean(tf.map_fn(lambda x: tf.reduce_sum(tf.equal(labels[x[1:k+1]], labels[x[0]])) / tf.reduce_sum(tf.equal(labels, labels[x[0]])), (tf.argsort(distance(embeddings, embeddings), axis=1)), tf.float32))

def knn_f1(embeddings, labels, k=5):
    precision = knn_precision(embeddings, labels, k)
    recall = knn_recall(embeddings, labels, k)
    return 2 * (precision * recall) / (precision + recall)

# Training
def train(model, criterion, optimizer, dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            anchor, positive, negative = batch
            with tf.GradientTape() as tape:
                anchor_embeddings = model(anchor, training=True)
                positive_embeddings = model(positive, training=True)
                negative_embeddings = model(negative, training=True)
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f'Epoch: {epoch+1}, Loss: {loss.numpy()}')

def validate(model, embeddings, labels, k=5):
    predicted_embeddings = model.predict(embeddings)
    print("Validation KNN Accuracy:", knn_accuracy(predicted_embeddings, labels, k))
    print("Validation KNN Precision:", knn_precision(predicted_embeddings, labels, k))
    print("Validation KNN Recall:", knn_recall(predicted_embeddings, labels, k))
    print("Validation KNN F1-score:", knn_f1(predicted_embeddings, labels, k))

def pipeline(learning_rate, batch_size, epochs, num_negatives, embedding_dim, num_features):
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    model = build_model(embedding_dim, num_features)
    criterion = build_criterion()
    optimizer = build_optimizer(model, learning_rate)
    dataset = build_dataset(samples, labels, num_negatives, batch_size)
    train(model, criterion, optimizer, dataset, epochs)
    input_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32).reshape((1, 10))
    output = model.predict(input_ids)
    model.save_weights("triplet_model.h5")
    predicted_embeddings = model.predict(samples)
    print(distance(output, output))
    print(similarity(output, output))
    print(cosine_distance(output, output))
    print(nearest_neighbors(predicted_embeddings, output, k=5))
    print(similar_embeddings(predicted_embeddings, output, k=5))
    validate(model, samples, labels, k=5)

def main():
    np.random.seed(42)
    pipeline(1e-4, 32, 10, 5, 101, 10)

if __name__ == "__main__":
    main()