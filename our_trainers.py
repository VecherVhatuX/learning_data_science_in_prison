import tensorflow as tf
import numpy as np

def build_model(embedding_dim, num_features):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(embedding_dim, num_features, input_length=10),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_features),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LayerNormalization()
    ])

def build_criterion(margin=1.0):
    @tf.function
    def triplet_loss(anchor, positive, negative):
        d_ap = tf.norm(anchor - positive, axis=-1)
        d_an = tf.norm(anchor[:, None] - negative, axis=-1)
        loss = tf.maximum(d_ap - tf.reduce_min(d_an, axis=-1) + margin, 0.0)
        return tf.reduce_mean(loss)
    return triplet_loss

def build_optimizer(model, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def build_dataset(samples, labels, num_negatives, batch_size, shuffle=True):
    @tf.function
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
            anchor, positive, negative = batch
            with tf.GradientTape() as tape:
                anchor_embeddings = model(anchor, training=True)
                positive_embeddings = model(positive, training=True)
                negative_embeddings = model(negative, training=True)
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')

def evaluate(model, dataset):
    embeddings = []
    for batch in dataset:
        batch_embeddings = model(batch[0])
        embeddings.append(batch_embeddings)
    return tf.concat(embeddings, axis=0)

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

    model.save_weights("triplet_model.h5")

    predicted_embeddings = evaluate(model, dataset)
    print(predicted_embeddings)

    distance = distance(output, output)
    print(distance)

    similarity = similarity(output, output)
    print(similarity)

    cosine_distance = cosine_distance(output, output)
    print(cosine_distance)

    all_embeddings = model.predict(samples)
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