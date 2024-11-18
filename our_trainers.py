import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

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

    def fetch_samples(self):
        return self.samples

    def fetch_labels(self):
        return self.labels

    def fetch_batch_size(self):
        return self.batch_size

    def fetch_num_negatives(self):
        return self.num_negatives

    def print_dataset_info(self):
        print("Dataset Information:")
        print(f"  Number of Samples: {self.samples.shape}")
        print(f"  Number of Labels: {self.labels.shape}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Number of Negatives: {self.num_negatives}")


class TripletNetwork(Model):
    def __init__(self, num_embeddings, embedding_dim):
        super(TripletNetwork, self).__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim, input_length=10)
        self.dense = Dense(embedding_dim)
        self.lambda_layer = Lambda(lambda x: x / K.linalg.norm(x, axis=-1, keepdims=True))

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dense(tf.math.reduce_mean(x, axis=1))
        x = self.lambda_layer(x)
        return x


class TripletModel:
    def __init__(self, num_embeddings, embedding_dim, margin, learning_rate, device):
        self.device = device
        self.model = TripletNetwork(num_embeddings, embedding_dim)
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train_model(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(dataset):
                with tf.device(self.device):
                    anchor_inputs = data['anchor_input_ids']
                    positive_inputs = data['positive_input_ids']
                    negative_inputs = data['negative_input_ids']
                    
                    anchor_embeddings = self.model(anchor_inputs)
                    positive_embeddings = self.model(positive_inputs)
                    negative_embeddings = self.model(negative_inputs)
                    
                    anchor_positive_distance = tf.norm(anchor_embeddings - positive_embeddings, axis=-1)
                    anchor_negative_distance = tf.norm(anchor_embeddings[:, None] - negative_embeddings, axis=-1)
                    
                    min_anchor_negative_distance = tf.reduce_min(anchor_negative_distance, axis=-1)
                    loss = self.loss_fn(min_anchor_negative_distance, anchor_positive_distance)
                    loss += self.model.loss_fn.margin * tf.reduce_mean(tf.maximum(0.0, anchor_positive_distance - min_anchor_negative_distance))
                    
                    with tf.GradientTape() as tape:
                        tape.watch(self.model.trainable_variables)
                        loss_value = loss
                    grads = tape.gradient(loss_value, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    total_loss += loss_value.numpy()
            print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

    def evaluate_model(self, dataset):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            anchor_inputs = data['anchor_input_ids']
            positive_inputs = data['positive_input_ids']
            negative_inputs = data['negative_input_ids']
            
            anchor_embeddings = self.model(anchor_inputs)
            positive_embeddings = self.model(positive_inputs)
            negative_embeddings = self.model(negative_inputs)
            
            anchor_positive_distance = tf.norm(anchor_embeddings - positive_embeddings, axis=-1)
            anchor_negative_distance = tf.norm(anchor_embeddings[:, None] - negative_embeddings, axis=-1)
            
            min_anchor_negative_distance = tf.reduce_min(anchor_negative_distance, axis=-1)
            loss = self.loss_fn(min_anchor_negative_distance, anchor_positive_distance)
            loss += 1 * tf.reduce_mean(tf.maximum(0.0, anchor_positive_distance - min_anchor_negative_distance))
            
            total_loss += loss.numpy()
        print(f'Validation Loss: {total_loss / (i+1):.3f}')

    def make_prediction(self, input_ids):
        return self.model(input_ids)


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
    model.train_model(dataset, epochs)
    input_ids = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)[None, :]
    output = model.make_prediction(input_ids)
    print(output)


if __name__ == "__main__":
    main()