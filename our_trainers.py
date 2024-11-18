import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

def create_triplet_network(num_embeddings, embedding_dim):
    inputs = Input(shape=(10,))
    x = Embedding(num_embeddings, embedding_dim, input_length=10)(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(embedding_dim)(x)
    x = Lambda(lambda x: x / K.linalg.norm(x, axis=-1, keepdims=True))(x)
    return Model(inputs=inputs, outputs=x)

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    anchor_positive_distance = tf.norm(anchor_embeddings - positive_embeddings, axis=-1)
    anchor_negative_distance = tf.norm(anchor_embeddings[:, None] - negative_embeddings, axis=-1)

    min_anchor_negative_distance = tf.reduce_min(anchor_negative_distance, axis=-1)
    return tf.reduce_mean(tf.maximum(0.0, anchor_positive_distance - min_anchor_negative_distance + margin))

def train_network(network, dataset, epochs, lr, device):
    optimizer = Adam(learning_rate=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            with tf.device(device):
                anchor_inputs = data['anchor_input_ids']
                positive_inputs = data['positive_input_ids']
                negative_inputs = data['negative_input_ids']

                anchor_embeddings = network(anchor_inputs)
                positive_embeddings = network(positive_inputs)
                negative_embeddings = network(negative_inputs)

                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, network.margin)

                with tf.GradientTape() as tape:
                    tape.watch(network.trainable_variables)
                    loss_value = loss
                grads = tape.gradient(loss_value, network.trainable_variables)
                optimizer.apply_gradients(zip(grads, network.trainable_variables))
                total_loss += loss_value.numpy()
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

def evaluate_network(network, dataset):
    total_loss = 0.0
    for i, data in enumerate(dataset):
        anchor_inputs = data['anchor_input_ids']
        positive_inputs = data['positive_input_ids']
        negative_inputs = data['negative_input_ids']

        anchor_embeddings = network(anchor_inputs)
        positive_embeddings = network(positive_inputs)
        negative_embeddings = network(negative_inputs)

        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, network.margin)
        total_loss += loss.numpy()
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict(network, input_ids):
    return network(input_ids)

def save_model(network, path):
    network.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)

class TripletDataset:
    def __init__(self, samples, labels, batch_size, num_negatives):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def __len__(self):
        return -(-len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.samples))
        anchor_idx = np.arange(start_idx, end_idx)
        anchor_labels = self.labels[anchor_idx]

        positive_idx = np.array([np.random.choice(np.where(self.labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(self.labels != label)[0], size=self.num_negatives, replace=False) for label in anchor_labels])

        return {
            'anchor_input_ids': self.samples[anchor_idx],
            'positive_input_ids': self.samples[positive_idx],
            'negative_input_ids': self.samples[negative_idx]
        }

    def get_samples(self):
        return self.samples

    def get_labels(self):
        return self.labels

    def get_batch_size(self):
        return self.batch_size

    def get_num_negatives(self):
        return self.num_negatives

    def print_info(self):
        print("Dataset Information:")
        print(f"  Number of Samples: {self.samples.shape}")
        print(f"  Number of Labels: {self.labels.shape}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Number of Negatives: {self.num_negatives}")

class TripletModel:
    def __init__(self, num_embeddings, embedding_dim, margin, lr, device):
        self.network = create_triplet_network(num_embeddings, embedding_dim)
        self.margin = margin
        self.lr = lr
        self.device = device

    def train(self, dataset, epochs):
        train_network(self.network, dataset, epochs, self.lr, self.device)

    def evaluate(self, dataset):
        evaluate_network(self.network, dataset)

    def predict(self, input_ids):
        return predict(self.network, input_ids)

    def save_model(self, path):
        save_model(self.network, path)

    def load_model(self, path):
        self.network = load_model(path)

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, (100,))
    batch_size = 32
    num_negatives = 5
    epochs = 10
    num_embeddings = 101
    embedding_dim = 10
    margin = 1.0
    lr = 1e-4

    dataset = TripletDataset(samples, labels, batch_size, num_negatives)

    model = TripletModel(num_embeddings, embedding_dim, margin, lr, device)
    model.train(dataset, epochs)
    input_ids = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)[None, :]
    output = model.predict(input_ids)
    print(output)
    model.save_model("triplet_model.h5")
    model.load_model("triplet_model.h5")
    print("Model saved and loaded successfully.")

if __name__ == "__main__":
    main()