import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
from flax import linen as nn
from flax.training import train_state
import numpy as np
import tensorflow as tf
import optax

class TripletModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Embed(num_embeddings=101, features=10)(x)
        x = x.transpose((0, 2, 1))
        x = nn.avg_pool(x, window_shape=(x.shape[-1],), strides=None, padding='VALID')
        x = x.squeeze(2)
        x = nn.Dense(features=10)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        return x / jnp.linalg.norm(x, axis=1, keepdims=True)

class TripletLossModule(nn.Module):
    @nn.compact
    def __call__(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        return jnp.mean(jnp.maximum(jnp.linalg.norm(anchor_embeddings - positive_embeddings, axis=1)
                                    - jnp.linalg.norm(anchor_embeddings[:, jnp.newaxis] - negative_embeddings, axis=2).min(axis=1)
                                    + 1.0, 0.0))

class Dataset:
    def __init__(self, samples, labels, num_negatives, batch_size, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(samples))
        if shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        anchor_idx = self.indices[idx]
        anchor_label = self.labels[anchor_idx]

        positive_idx = np.random.choice(np.where(self.labels == anchor_label)[0], size=1)[0]
        negative_idx = np.random.choice(np.where(self.labels != anchor_label)[0], size=self.num_negatives, replace=False)

        return {
            'anchor_input_ids': np.array(self.samples[anchor_idx], dtype=np.int32),
            'positive_input_ids': np.array(self.samples[positive_idx], dtype=np.int32),
            'negative_input_ids': np.array(self.samples[negative_idx], dtype=np.int32)
        }

    def __len__(self):
        return len(self.samples)

class Trainer:
    def __init__(self, model, dataset, epochs, learning_rate, batch_size):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = TripletLossModule()
        self.state = self.create_train_state()

    def create_train_state(self):
        tx = optax.adam(learning_rate=self.learning_rate)
        params = self.model.init(jax.random.PRNGKey(42), jnp.ones((1, 10), dtype=jnp.int32))
        return train_state.TrainState(params, tx)

    def train_step(self, state, batch):
        def loss_fn(params, batch):
            anchor_embeddings = self.model.apply(params, batch['anchor_input_ids'])
            positive_embeddings = self.model.apply(params, batch['positive_input_ids'])
            negative_embeddings = self.model.apply(params, batch['negative_input_ids'])
            return self.loss_fn.apply({}, anchor_embeddings, positive_embeddings, negative_embeddings)

        grads = jax.grad(loss_fn, has_aux=False)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state

    def train(self):
        for epoch in range(self.epochs):
            total_loss = 0.0
            dataloader = tf.data.Dataset.from_tensor_slices(self.dataset)
            dataloader = dataloader.batch(self.batch_size)
            dataloader = dataloader.prefetch(tf.data.AUTOTUNE)

            for i, batch in enumerate(dataloader):
                batch = {k: jax.device_put(v.numpy()) for k, v in batch.items()}
                self.state = self.train_step(self.state, batch)
                total_loss += self.loss_fn.apply({}, self.model.apply(self.state.params, batch['anchor_input_ids']), 
                                                 self.model.apply(self.state.params, batch['positive_input_ids']), 
                                                 self.model.apply(self.state.params, batch['negative_input_ids']))

            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(self):
        total_loss = 0.0
        dataloader = tf.data.Dataset.from_tensor_slices(self.dataset)
        dataloader = dataloader.batch(self.batch_size)
        dataloader = dataloader.prefetch(tf.data.AUTOTUNE)

        for i, batch in enumerate(dataloader):
            batch = {k: jax.device_put(v.numpy()) for k, v in batch.items()}
            total_loss += self.loss_fn.apply({}, self.model.apply(self.state.params, batch['anchor_input_ids']), 
                                             self.model.apply(self.state.params, batch['positive_input_ids']), 
                                             self.model.apply(self.state.params, batch['negative_input_ids']))

        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def predict(self, input_ids, batch_size):
        predictions = []
        dataloader = tf.data.Dataset.from_tensor_slices(input_ids)
        dataloader = dataloader.batch(batch_size)
        dataloader = dataloader.prefetch(tf.data.AUTOTUNE)

        for batch in dataloader:
            batch = jax.device_put(batch.numpy())
            output = self.model.apply(self.state.params, batch)
            predictions.extend(output)

        return np.array(predictions)

    def save_model(self, path):
        jax2tf.convert(self.state.params, 'triplet_model')

    def load_model(self, path):
        params = jax2tf.checkpoint.load(path, target='')
        return params

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

    model = TripletModel()
    dataset = Dataset(samples, labels, num_negatives, batch_size)
    trainer = Trainer(model, dataset, epochs, learning_rate, batch_size)
    trainer.train()

    input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10))
    output = trainer.predict(input_ids, batch_size=1)
    print(output)

    trainer.save_model("triplet_model")
    loaded_params = trainer.load_model("triplet_model")

    trainer.evaluate()

    predicted_embeddings = trainer.predict(np.array([1, 2, 3, 4, 5], dtype=np.int32).reshape((1, 10)), batch_size=1)
    print(predicted_embeddings)

    distance = calculate_distance(predicted_embeddings[0], predicted_embeddings[0])
    print(distance)

    similarity = calculate_similarity(predicted_embeddings[0], predicted_embeddings[0])
    print(similarity)

    cosine_distance = calculate_cosine_distance(predicted_embeddings[0], predicted_embeddings[0])
    print(cosine_distance)

    all_embeddings = trainer.predict(np.array(samples, dtype=np.int32), batch_size=32)
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