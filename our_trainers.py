import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    def __len__():
        return -(-len(samples) // batch_size)
    
    def __getitem__(idx):
        start_idx = idx * batch_size
        end_idx = min((idx + 1) * batch_size, len(samples))
        anchor_idx = np.arange(start_idx, end_idx)
        anchor_labels = labels[anchor_idx]
        
        positive_idx = np.array([np.random.choice(np.where(labels == label)[0], size=1)[0] for label in anchor_labels])
        negative_idx = np.array([np.random.choice(np.where(labels != label)[0], size=num_negatives, replace=False) for label in anchor_labels])
        
        return {
            'anchor_input_ids': samples[anchor_idx],
            'positive_input_ids': samples[positive_idx],
            'negative_input_ids': samples[negative_idx]
        }
    
    def get_samples():
        return samples
    
    def get_labels():
        return labels
    
    def get_batch_size():
        return batch_size
    
    def get_num_negatives():
        return num_negatives
    
    def print_info():
        print("Dataset Information:")
        print(f"  Number of Samples: {samples.shape}")
        print(f"  Number of Labels: {labels.shape}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Number of Negatives: {num_negatives}")
    
    return type('TripletDataset', (), {
        '__len__': __len__,
        '__getitem__': __getitem__,
        'get_samples': get_samples,
        'get_labels': get_labels,
        'get_batch_size': get_batch_size,
        'get_num_negatives': get_num_negatives,
        'print_info': print_info,
    })()

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

def train(model, dataset, epochs, device):
    optimizer = Adam(learning_rate=model.lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(dataset):
            with tf.device(device):
                anchor_inputs = data['anchor_input_ids']
                positive_inputs = data['positive_input_ids']
                negative_inputs = data['negative_input_ids']
                
                anchor_embeddings = model.network(anchor_inputs)
                positive_embeddings = model.network(positive_inputs)
                negative_embeddings = model.network(negative_inputs)
                
                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, model.margin)
                
                with tf.GradientTape() as tape:
                    tape.watch(model.network.trainable_variables)
                    loss_value = loss
                grads = tape.gradient(loss_value, model.network.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.network.trainable_variables))
                total_loss += loss_value.numpy()
        print(f'Epoch: {epoch+1}, Loss: {total_loss/(i+1):.3f}')

def evaluate(model, dataset):
    total_loss = 0.0
    for i, data in enumerate(dataset):
        anchor_inputs = data['anchor_input_ids']
        positive_inputs = data['positive_input_ids']
        negative_inputs = data['negative_input_ids']
        
        anchor_embeddings = model.network(anchor_inputs)
        positive_embeddings = model.network(positive_inputs)
        negative_embeddings = model.network(negative_inputs)
        
        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, model.margin)
        total_loss += loss.numpy()
    print(f'Validation Loss: {total_loss / (i+1):.3f}')

def predict(model, input_ids):
    return model.network(input_ids)

class TripletModel:
    def __init__(self, num_embeddings, embedding_dim, margin, lr, device):
        self.network = create_triplet_network(num_embeddings, embedding_dim)
        self.margin = margin
        self.lr = lr
        self.device = device

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

    dataset = create_triplet_dataset(samples, labels, batch_size, num_negatives)

    model = TripletModel(num_embeddings, embedding_dim, margin, lr, device)
    train(model, dataset, epochs, device)
    input_ids = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)[None, :]
    output = predict(model, input_ids)
    print(output)


if __name__ == "__main__":
    main()