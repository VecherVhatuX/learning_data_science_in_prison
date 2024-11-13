import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics
import numpy as np

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.indices = np.arange(len(samples))

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        indices = np.random.permutation(self.indices)
        batch = indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        anchor_idx = batch
        positive_idx = []
        negative_indices = []
        for anchor in anchor_idx:
            idx = np.where(self.labels == self.labels[anchor])[0]
            positive_idx.append(np.random.choice(idx[idx != anchor]))
            idx = np.where(self.labels != self.labels[anchor])[0]
            negative_idx = np.setdiff1d(np.random.choice(idx, self.num_negatives, replace=False), [anchor])
            negative_indices.append(negative_idx)

        anchor_input_ids = self.samples[anchor_idx]
        positive_input_ids = self.samples[positive_idx]
        negative_input_ids = [self.samples[negative_idx] for negative_idx in negative_indices]

        return {
            'anchor_input_ids': anchor_input_ids,
            'positive_input_ids': positive_input_ids,
            'negative_input_ids': negative_input_ids
        }

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class TripletModel(tf.keras.Model):
    def __init__(self, num_embeddings, embedding_dim, num_negatives):
        super(TripletModel, self).__init__()
        self.embedding = layers.Embedding(num_embeddings, embedding_dim)
        self.pooling = layers.GlobalAveragePooling1D()
        self.num_negatives = num_negatives
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

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
        for negative_input_ids in negative_input_ids:
            negative_embeddings.append(self.embedding(negative_input_ids))
            negative_embeddings[-1] = self.pooling(negative_embeddings[-1])
            negative_embeddings[-1] = negative_embeddings[-1] / tf.norm(negative_embeddings[-1], axis=1, keepdims=True)
        negative_embeddings = tf.stack(negative_embeddings)

        return anchor_embeddings, positive_embeddings, negative_embeddings

    def triplet_loss(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        loss = tf.reduce_mean(tf.maximum(1.0 + 
                                         tf.reduce_sum((anchor_embeddings - positive_embeddings) ** 2, axis=1) - 
                                         tf.reduce_sum((anchor_embeddings - negative_embeddings[:, 0, :]) ** 2, axis=1), 
                                         0.0))
        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            anchor_embeddings, positive_embeddings, negative_embeddings = self(data, training=True)
            loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        anchor_embeddings, positive_embeddings, negative_embeddings = self(data, training=False)
        loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = TripletModel(101, 10, num_negatives)
    dataset = CustomDataset(samples, labels, batch_size, num_negatives)

    model.compile()
    history = model.fit(dataset, epochs=epochs, validation_data=dataset)
    model.save_weights("model")


if __name__ == "__main__":
    main()