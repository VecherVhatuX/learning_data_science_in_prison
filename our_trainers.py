import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.data import Dataset

def _generate_triplet_data(samples, labels, batch_size, num_negatives):
    idx = 0
    while True:
        anchor_idx = np.arange(idx * batch_size, min((idx + 1) * batch_size, len(samples)))
        anchor_labels = labels[anchor_idx]
        positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_labels])
        yield {
            'anchor_input_ids': samples[anchor_idx],
            'positive_input_ids': samples[positive_idx],
            'negative_input_ids': samples[negative_idx]
        }
        idx += 1
        if idx >= len(samples) // batch_size + (1 if len(samples) % batch_size != 0 else 0):
            idx = 0

def _triplet_loss_function(margin):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred
        anchor_positive_distance = tf.norm(anchor - positive, axis=-1)
        anchor_negative_distance = tf.norm(anchor[:, tf.newaxis] - negative, axis=-1)
        min_anchor_negative_distance = tf.reduce_min(anchor_negative_distance, axis=-1)
        triplet_loss = tf.maximum(anchor_positive_distance - min_anchor_negative_distance + margin, 0)
        return tf.reduce_mean(triplet_loss)
    return loss

class _TripletNetwork(models.Model):
    def __init__(self, num_embeddings, embedding_dim):
        super(_TripletNetwork, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Lambda(lambda x: x / tf.norm(x, axis=-1, keepdims=True))
        ])

    def call(self, inputs):
        return self.model(inputs)

class TripletModel:
    def __init__(self, num_embeddings, embedding_dim, margin, learning_rate):
        self.model = _TripletNetwork(num_embeddings, embedding_dim)
        self.loss_fn = _triplet_loss_function(margin)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for i, data in enumerate(dataset):
                with tf.GradientTape() as tape:
                    anchor_inputs = data['anchor_input_ids']
                    positive_inputs = data['positive_input_ids']
                    negative_inputs = data['negative_input_ids']
                    anchor_embeddings = self.model(anchor_inputs)
                    positive_embeddings = self.model(positive_inputs)
                    negative_embeddings = self.model(negative_inputs)
                    loss = self.loss_fn(None, (anchor_embeddings, positive_embeddings, negative_embeddings))
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                total_loss += loss
                if i >= len(dataset.dataset) // dataset.batch_size:
                    break
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate(self, dataset):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            anchor_inputs = data['anchor_input_ids']
            positive_inputs = data['positive_input_ids']
            negative_inputs = data['negative_input_ids']
            anchor_embeddings = self.model(anchor_inputs)
            positive_embeddings = self.model(positive_inputs)
            negative_embeddings = self.model(negative_inputs)
            loss = self.loss_fn(None, (anchor_embeddings, positive_embeddings, negative_embeddings))
            total_loss += loss
            if i >= len(dataset.dataset) // dataset.batch_size:
                break
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def predict(self, input_ids):
        return self.model(input_ids)

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    dataset = tf.data.Dataset.from_generator(lambda: _generate_triplet_data(samples, labels, batch_size, num_negatives), 
                                             output_types={'anchor_input_ids': tf.int32, 'positive_input_ids': tf.int32, 'negative_input_ids': tf.int32})
    dataset = dataset.batch(batch_size)
    validation_dataset = tf.data.Dataset.from_generator(lambda: _generate_triplet_data(samples, labels, batch_size, num_negatives), 
                                                        output_types={'anchor_input_ids': tf.int32, 'positive_input_ids': tf.int32, 'negative_input_ids': tf.int32})
    validation_dataset = validation_dataset.batch(batch_size)

    model = TripletModel(num_embeddings, embedding_dim, margin, lr)
    model.train(dataset, epochs)
    model.evaluate(validation_dataset)
    input_ids = tf.convert_to_tensor([1, 2, 3, 4, 5])
    output = model.predict(input_ids)
    print(output)

if __name__ == "__main__":
    main()