import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
import numpy as np

def create_embedding_model(num_embeddings, embedding_dim):
    return models.Sequential([
        layers.Embedding(input_dim=num_embeddings, output_dim=embedding_dim),
        layers.GlobalAveragePooling1D()
    ])

def create_triplet_dataset(samples, labels, batch_size, num_negatives):
    def get_batch(idx, samples, labels, batch_size, num_negatives):
        batch = np.random.choice(len(samples), batch_size, replace=False)
        anchor_idx = batch
        positive_idx = np.array([np.random.choice(np.where(labels == labels[anchor])[0]) for anchor in anchor_idx])
        while np.any(positive_idx == anchor_idx):
            positive_idx = np.array([np.random.choice(np.where(labels == labels[anchor])[0]) for anchor in anchor_idx])

        negative_indices = [np.random.choice(np.where(labels != labels[anchor])[0], num_negatives, replace=False) for anchor in anchor_idx]
        negative_indices = [np.setdiff1d(negative_idx, [anchor]) for anchor, negative_idx in zip(anchor_idx, negative_indices)]

        return {
            'anchor_input_ids': samples[anchor_idx],
            'positive_input_ids': samples[positive_idx],
            'negative_input_ids': np.stack([samples[negative_idx] for negative_idx in negative_indices]),
        }

    def on_epoch_end(samples, labels):
        np.random.shuffle(samples)

    return tf.data.Dataset.from_generator(
        lambda: (get_batch(idx, samples, labels, batch_size, num_negatives) for idx in range(-(-len(samples) // batch_size))),
        output_types={'anchor_input_ids': tf.int32, 'positive_input_ids': tf.int32, 'negative_input_ids': tf.int32},
        output_shapes={'anchor_input_ids': (batch_size, samples.shape[1]), 'positive_input_ids': (batch_size, samples.shape[1]), 'negative_input_ids': (batch_size, num_negatives, samples.shape[1])}
    ).prefetch(tf.data.AUTOTUNE)

def normalize_embeddings(embeddings):
    return embeddings / tf.norm(embeddings, axis=1, keepdims=True)

def triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin):
    return tf.reduce_mean(tf.maximum(margin + 
                                     tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=1) - 
                                     tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings[:, 0, :]), axis=1), 
                                     0.0))

def train_step(model, optimizer, inputs, margin):
    with tf.GradientTape() as tape:
        anchor_input_ids = inputs["anchor_input_ids"]
        positive_input_ids = inputs["positive_input_ids"]
        negative_input_ids = inputs["negative_input_ids"]

        anchor_embeddings = normalize_embeddings(model(anchor_input_ids))
        positive_embeddings = normalize_embeddings(model(positive_input_ids))
        negative_embeddings = normalize_embeddings(model(negative_input_ids))

        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return {"loss": loss}

def main():
    samples = np.random.randint(0, 100, (100, 10))
    labels = np.random.randint(0, 2, 100)
    batch_size = 32
    num_negatives = 5
    epochs = 10

    model = create_embedding_model(100, 10)
    dataset = create_triplet_dataset(samples, labels, batch_size, num_negatives)
    optimizer = SGD(learning_rate=1e-4)
    margin = 1.0

    for epoch in range(epochs):
        total_loss = 0
        for inputs in dataset:
            loss = train_step(model, optimizer, inputs, margin)
            total_loss += loss["loss"]
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

    model.save_weights("model.h5")

if __name__ == "__main__":
    main()