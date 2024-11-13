import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class TripletDataGenerator:
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = np.arange(len(samples))

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        np.random.shuffle(self.indices)
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        batch = self.indices[start_idx:end_idx]

        anchor_idx = batch
        positive_idx = []
        negative_indices = []
        for anchor in anchor_idx:
            idx = np.where(self.labels == self.labels[anchor])[0]
            positive_idx.append(np.random.choice(idx[idx != anchor]))
            idx = np.where(self.labels != self.labels[anchor])[0]
            negative_idx = np.setdiff1d(np.random.choice(idx, self.num_negatives, replace=False), [anchor])
            negative_indices.append(negative_idx)

        anchor_input_ids = tf.constant(self.samples[anchor_idx])
        positive_input_ids = tf.constant(self.samples[positive_idx])
        negative_input_ids = tf.stack([tf.constant(self.samples[negative_idx]) for negative_idx in negative_indices])

        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }


class TripletModel(tf.keras.Model):
    def __init__(self, num_embeddings, embedding_dim, num_negatives):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()
        self.num_negatives = num_negatives

    def call(self, inputs):
        anchor_input_ids = inputs['anchor_input_ids']
        positive_input_ids = inputs['positive_input_ids']
        negative_input_ids = inputs['negative_input_ids']

        anchor_embeddings = self.embedding(anchor_input_ids)
        anchor_embeddings = self.pooling(anchor_embeddings)
        anchor_embeddings = anchor_embeddings / tf.norm(anchor_embeddings, axis=1, keepdims=True)

        positive_embeddings = self.embedding(positive_input_ids)
        positive_embeddings = self.pooling(positive_embeddings)
        positive_embeddings = positive_embeddings / tf.norm(positive_embeddings, axis=1, keepdims=True)

        negative_embeddings = []
        for negative_input_ids in tf.unstack(negative_input_ids, axis=0):
            negative_embeddings.append(self.embedding(negative_input_ids))
            negative_embeddings[-1] = self.pooling(negative_embeddings[-1])
            negative_embeddings[-1] = negative_embeddings[-1] / tf.norm(negative_embeddings[-1], axis=1, keepdims=True)
        negative_embeddings = tf.stack(negative_embeddings)

        loss = tf.reduce_mean(tf.maximum(1.0 + 
                                         tf.reduce_sum((anchor_embeddings - positive_embeddings) ** 2, axis=1) - 
                                         tf.reduce_sum((anchor_embeddings - negative_embeddings[:, 0, :]) ** 2, axis=1), 
                                         0.0))

        self.add_loss(loss)
        return anchor_embeddings


def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletModel(101, 10, num_negatives)
    dataset = TripletDataGenerator(samples, labels, batch_size, num_negatives)
    dataset = tf.data.Dataset.from_generator(lambda: dataset, output_types=dataset[0].dtype, output_shapes={k: v.shape for k, v in dataset[0].items()})

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            with tf.GradientTape() as tape:
                _ = model(batch, training=True)
                loss = sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss.numpy()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")
    model.save_weights("model")


if __name__ == "__main__":
    main()