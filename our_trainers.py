import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DataShuffler:
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = np.arange(len(samples))

    def shuffle(self):
        np.random.shuffle(self.indices)

    def get_batch(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        batch = self.indices[start_idx:end_idx]
        return batch


class IndexSampler:
    def __init__(self, labels, num_negatives):
        self.labels = labels
        self.num_negatives = num_negatives

    def sample_positive(self, anchor_idx):
        positive_idx = []
        for anchor in anchor_idx:
            idx = np.where(self.labels == self.labels[anchor])[0]
            positive_idx.append(np.random.choice(idx[idx != anchor]))
        return np.array(positive_idx)

    def sample_negative(self, anchor_idx):
        negative_indices = []
        for anchor in anchor_idx:
            idx = np.where(self.labels != self.labels[anchor])[0]
            negative_idx = np.setdiff1d(np.random.choice(idx, self.num_negatives, replace=False), [anchor])
            negative_indices.append(negative_idx)
        return negative_indices


class BatchCreator:
    def __init__(self, samples):
        self.samples = samples

    def create_batch(self, anchor_idx, positive_idx, negative_indices):
        anchor_input_ids = tf.constant(self.samples[anchor_idx])
        positive_input_ids = tf.constant(self.samples[positive_idx])
        negative_input_ids = tf.stack([tf.constant(self.samples[negative_idx]) for negative_idx in negative_indices])
        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }


class TripletData(tf.data.Dataset):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.data_shuffler = DataShuffler(samples, labels, batch_size, num_negatives)
        self.index_sampler = IndexSampler(labels, num_negatives)
        self.batch_creator = BatchCreator(samples)

    def __len__(self):
        return (len(self.data_shuffler.samples) + self.data_shuffler.batch_size - 1) // self.data_shuffler.batch_size

    def __getitem__(self, idx):
        self.data_shuffler.shuffle()
        batch = self.data_shuffler.get_batch(idx)
        positive_idx = self.index_sampler.sample_positive(batch)
        negative_indices = self.index_sampler.sample_negative(batch)
        return self.batch_creator.create_batch(batch, positive_idx, negative_indices)


class EmbeddingModel(tf.keras.Model):
    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        x = self.embedding(x, training=training)
        x = self.pooling(x)
        return x

    def standardize_vectors(self, embeddings):
        return embeddings / tf.norm(embeddings, axis=1, keepdims=True)


class TripletLossModel(tf.keras.Model):
    def __init__(self, model):
        super(TripletLossModel, self).__init__()
        self.model = model

    def triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings, margin):
        return tf.reduce_mean(tf.maximum(margin + 
                                         tf.reduce_sum((anchor_embeddings - positive_embeddings) ** 2, axis=1) - 
                                         tf.reduce_sum((anchor_embeddings - negative_embeddings[:, 0, :]) ** 2, axis=1), 
                                         0.0))

    def compile(self, optimizer, margin):
        self.optimizer = optimizer
        self.margin = margin
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            anchor_input_ids = batch["anchor_input_ids"]
            positive_input_ids = batch["positive_input_ids"]
            negative_input_ids = batch["negative_input_ids"]
            anchor_embeddings = self.model.standardize_vectors(self.model(anchor_input_ids, training=True))
            positive_embeddings = self.model.standardize_vectors(self.model(positive_input_ids, training=True))
            negative_embeddings = self.model.standardize_vectors(self.model(negative_input_ids, training=True))
            loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}


def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = EmbeddingModel(101, 10)
    triplet_model = TripletLossModel(model)
    triplet_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4), margin=1.0)
    dataset = TripletData(samples, labels, batch_size, num_negatives)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            loss = triplet_model.train_step(batch).get("loss")
            total_loss += loss.numpy()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")
    model.save_weights("model")


if __name__ == "__main__":
    main()