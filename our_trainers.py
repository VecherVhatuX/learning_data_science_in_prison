import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
from flax import linen as nn
from flax.training import train_state
import numpy as np
import tensorflow as tf
import optax

def create_triplet_model(num_embeddings, features):
    def model(x):
        x = nn.Embed(num_embeddings=num_embeddings, features=features)(x)
        x = x.transpose((0, 2, 1))
        x = nn.avg_pool(x, window_shape=(x.shape[-1],), strides=None, padding='VALID')
        x = x.squeeze(2)
        x = nn.Dense(features=features)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        return x / jnp.linalg.norm(x, axis=1, keepdims=True)
    return model

def create_triplet_loss_fn():
    def loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings):
        return jnp.mean(jnp.maximum(jnp.linalg.norm(anchor_embeddings - positive_embeddings, axis=1)
                                    - jnp.linalg.norm(anchor_embeddings[:, jnp.newaxis] - negative_embeddings, axis=2).min(axis=1)
                                    + 1.0, 0.0))
    return loss_fn

def create_dataset(samples, labels, num_negatives, batch_size, shuffle=True):
    def dataset():
        indices = np.arange(len(samples))
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(len(indices) // batch_size):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            anchor_idx = np.random.choice(batch_indices, size=batch_size)
            anchor_label = labels[anchor_idx]

            positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_label])
            negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_label])

            yield {
                'anchor_input_ids': np.array([samples[i] for i in anchor_idx], dtype=np.int32),
                'positive_input_ids': np.array([samples[i] for i in positive_idx], dtype=np.int32),
                'negative_input_ids': np.array([samples[i] for i in negative_idx], dtype=np.int32)
            }
    return dataset

def create_train_state(model, learning_rate, batch_size):
    tx = optax.adam(learning_rate=learning_rate)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.ones((1, 10), dtype=jnp.int32))
    return train_state.TrainState(params, tx)

def train_step(state, batch, model, loss_fn):
    def loss_fn_step(params, batch):
        anchor_embeddings = model.apply(params, batch['anchor_input_ids'])
        positive_embeddings = model.apply(params, batch['positive_input_ids'])
        negative_embeddings = model.apply(params, batch['negative_input_ids'])
        return loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

    grads = jax.grad(loss_fn_step, has_aux=False)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state

def train(model, dataset, loss_fn, state, epochs, batch_size):
    for epoch in range(epochs):
        total_loss = 0.0
        dataloader = tf.data.Dataset.from_generator(dataset, output_types={'anchor_input_ids': tf.int32, 'positive_input_ids': tf.int32, 'negative_input_ids': tf.int32})
        dataloader = dataloader.batch(batch_size)
        dataloader = dataloader.prefetch(tf.data.AUTOTUNE)

        for i, batch in enumerate(dataloader):
            batch = {k: jax.device_put(v.numpy()) for k, v in batch.items()}
            state = train_step(state, batch, model, loss_fn)
            total_loss += loss_fn(model.apply(state.params, batch['anchor_input_ids']), model.apply(state.params, batch['positive_input_ids']), model.apply(state.params, batch['negative_input_ids']))

        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

def evaluate(model, dataset, loss_fn, state, batch_size):
    total_loss = 0.0
    dataloader = tf.data.Dataset.from_generator(dataset, output_types={'anchor_input_ids': tf.int32, 'positive_input_ids': tf.int32, 'negative_input_ids': tf.int32})
    dataloader = dataloader.batch(batch_size)
    dataloader = dataloader.prefetch(tf.data.AUTOTUNE)

    for i, batch in enumerate(dataloader):
        batch = {k: jax.device_put(v.numpy()) for k, v in batch.items()}
        total_loss += loss_fn(model.apply(state.params, batch['anchor_input_ids']), model.apply(state.params, batch['positive_input_ids']), model.apply(state.params, batch['negative_input_ids']))

    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict(model, state, input_ids, batch_size):
    predictions = []
    dataloader = tf.data.Dataset.from_tensor_slices(input_ids)
    dataloader = dataloader.batch(batch_size)
    dataloader = dataloader.prefetch(tf.data.AUTOTUNE)

    for batch in dataloader:
        batch = jax.device_put(batch.numpy())
        output = model.apply(state.params, batch)
        predictions.extend(output)

    return np.array(predictions)

def save_model(params, path):
    jax2tf.convert(params, 'triplet_model')

def load_model(path):
    return jax2tf.checkpoint.load(path, target='')

def calculate_distance(embedding1, embedding2):
    return jnp.linalg.norm(embedding1 - embedding2, axis=1)

def calculate_similarity(embedding1, embedding2):
    return jnp.sum(embedding1 * embedding2, axis=1) / (jnp.linalg.norm(embedding1, axis=1) * jnp.linalg.norm(embedding2, axis=1))

def calculate_cosine_distance(embedding1, embedding2):
    return 1 - calculate_similarity(embedding1, embedding2)

def get_nearest_neighbors(embeddings, target_embedding, k=5):
    distances = calculate_distance(embeddings, target_embedding)
    indices = jnp.argsort(distances)[:k]
    return indices

def get_similar_embeddings(embeddings, target_embedding, k=5):
    similarities = calculate_similarity(embeddings, target_embedding)
    indices = jnp.argsort(similarities)[-k:]
    return indices

def calculate_knn_accuracy(embeddings, labels, k=5):
    correct = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = jnp.argsort(distances)[:k]
        nearest_labels = labels[indices]
        if labels[i] in nearest_labels:
            correct += 1
    return correct / len(embeddings)

def calculate_knn_precision(embeddings, labels, k=5):
    precision = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = jnp.argsort(distances)[:k]
        nearest_labels = labels[indices]
        precision += len(np.where(nearest_labels == labels[i])[0]) / k
    return precision / len(embeddings)

def calculate_knn_recall(embeddings, labels, k=5):
    recall = 0
    for i in range(len(embeddings)):
        distances = calculate_distance(embeddings, embeddings[i])
        indices = jnp.argsort(distances)[:k]
        nearest_labels = labels[indices]
        recall += len(np.where(nearest_labels == labels[i])[0]) / len(np.where(labels == labels[i])[0])
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

    model = create_triplet_model(num_embeddings=101, features=10)
    loss_fn = create_triplet_loss_fn()
    dataset = create_dataset(samples, labels, num_negatives, batch_size)
    state = create_train_state(model, learning_rate, batch_size)
    train(model, dataset, loss_fn, state, epochs, batch_size)

    input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10))
    output = predict(model, state, input_ids, batch_size=1)
    print(output)

    save_model(state.params, "triplet_model")
    loaded_params = load_model("triplet_model")

    evaluate(model, dataset, loss_fn, state, batch_size)

    predicted_embeddings = predict(model, state, np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10)), batch_size=1)
    print(predicted_embeddings)

    distance = calculate_distance(predicted_embeddings[0], predicted_embeddings[0])
    print(distance)

    similarity = calculate_similarity(predicted_embeddings[0], predicted_embeddings[0])
    print(similarity)

    cosine_distance = calculate_cosine_distance(predicted_embeddings[0], predicted_embeddings[0])
    print(cosine_distance)

    all_embeddings = predict(model, state, np.array(samples, dtype=np.int32), batch_size=32)
    nearest_neighbors = get_nearest_neighbors(all_embeddings, predicted_embeddings[0], k=5)
    print(nearest_neighbors)

    similar_embeddings = get_similar_embeddings(all_embeddings, predicted_embeddings[0], k=5)
    print(similar_embeddings)

    print("KNN Accuracy:", calculate_knn_accuracy(all_embeddings, labels, k=5))

    print("KNN Precision:", calculate_knn_precision(all_embeddings, labels, k=5))

    print("KNN Recall:", calculate_knn_recall(all_embeddings, labels, k=5))

    print("KNN F1-score:", calculate_knn_f1(all_embeddings, labels, k=5))

if __name__ == "__main__":
    main()