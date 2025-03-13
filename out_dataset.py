import json
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, losses

gather_texts = lambda triplet_data: [text for item in triplet_data for text in (item['anchor'], item['positive'], item['negative'])]

tokenize_sequences = lambda tokenizer, data_item: {
    'anchor_seq': tf.convert_to_tensor(tokenizer.transform([data_item['anchor']])[0]),
    'positive_seq': tf.convert_to_tensor(tokenizer.transform([data_item['positive']])[0]),
    'negative_seq': tf.convert_to_tensor(tokenizer.transform([data_item['negative']])[0])
}

shuffle_data = lambda data_samples: random.shuffle(data_samples) or data_samples

generate_triplets = lambda instance_dict, bug_samples, non_bug_samples: [
    {
        'anchor': instance_dict[os.path.basename(folder)],
        'positive': bug_sample,
        'negative': random.choice(non_bug_samples)
    }
    for folder, _ in snippet_files
    for bug_sample in bug_samples
]

load_json_data = lambda file_path, root_dir: (
    lambda json_content: (
        {entry['instance_id']: entry['problem_statement'] for entry in json_content},
        [(folder, os.path.join(root_dir, 'snippet.json')) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    )
)(json.load(open(file_path, 'r')))

prepare_triplet_dataset = lambda instance_dict, snippet_files: generate_triplets(instance_dict, *zip(*[json.load(open(path)) for path in snippet_files]))

class TripletData:
    def __init__(self, triplet_data):
        self.triplet_data = triplet_data
        self.tokenizer = LabelEncoder().fit(gather_texts(triplet_data))

    get_samples = lambda self: self.triplet_data

class TripletDataset(tf.data.Dataset):
    def __init__(self, triplet_data):
        self.data = TripletData(triplet_data)
        self.samples = self.data.get_samples()

    __len__ = lambda self: len(self.samples)
    __getitem__ = lambda self, index: tokenize_sequences(self.data.tokenizer, self.samples[index])

class TripletModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(TripletModel, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim)
        self.dense_network = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])

    call = lambda self, anchor, positive, negative: (
        self.dense_network(self.embedding_layer(anchor)),
        self.dense_network(self.embedding_layer(positive)),
        self.dense_network(self.embedding_layer(negative))
    )

calculate_loss = lambda anchor_embeds, positive_embeds, negative_embeds: tf.reduce_mean(tf.maximum(0.2 + tf.norm(anchor_embeds - positive_embeds, axis=1) - tf.norm(anchor_embeds - negative_embeds, axis=1), 0))

train_model = lambda model, train_loader, valid_loader, epochs: (
    lambda optimizer, history: [
        [
            [
                (lambda anchor_embeds, positive_embeds, negative_embeds, loss: (
                    optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables)),
                    history.append((loss.numpy(), *evaluate_model(model, valid_loader)))
                )(model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq']), calculate_loss(anchor_embeds, positive_embeds, negative_embeds))
            ) for batch in train_loader
        ] for _ in range(epochs)
    ] or history
)(optimizers.Adam(learning_rate=1e-5), [])

evaluate_model = lambda model, valid_loader: (
    lambda loss, correct: (
        loss / len(valid_loader), correct / len(valid_loader.dataset))
    )(
        sum(calculate_loss(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])).numpy() for batch in valid_loader),
        sum(count_matches(*model(batch['anchor_seq'], batch['positive_seq'], batch['negative_seq'])) for batch in valid_loader)
    )

count_matches = lambda anchor_output, positive_output, negative_output: tf.reduce_sum(tf.cast(tf.reduce_sum(anchor_output * positive_output, axis=1) > tf.reduce_sum(anchor_output * negative_output, axis=1), tf.int32)).numpy()

plot_results = lambda history: (
    lambda train_losses, val_losses, train_accuracies: (
        plt.figure(figsize=(10, 5)),
        plt.subplot(1, 2, 1),
        plt.plot(train_losses, label='Training Loss'),
        plt.plot(val_losses, label='Validation Loss'),
        plt.title('Loss Over Epochs'),
        plt.xlabel('Epochs'),
        plt.ylabel('Loss'),
        plt.legend(),
        plt.subplot(1, 2, 2),
        plt.plot(train_accuracies, label='Training Accuracy'),
        plt.title('Accuracy Over Epochs'),
        plt.xlabel('Epochs'),
        plt.ylabel('Accuracy'),
        plt.legend(),
        plt.show())
    )(*zip(*history))

save_model = lambda model, filepath: (model.save_weights(filepath), print(f'Model saved at {filepath}'))

load_model = lambda model, filepath: (model.load_weights(filepath), print(f'Model loaded from {filepath}'), model)

main = lambda: (
    lambda dataset_path, snippets_directory, instance_dict, snippet_paths, triplet_data, train_data, valid_data, train_loader, valid_loader, model, history: (
        plot_results(history),
        save_model(model, 'triplet_model.h5'))
    )(
        'datasets/SWE-bench_oracle.npy',
        'datasets/10_10_after_fix_pytest',
        *load_json_data('datasets/SWE-bench_oracle.npy', 'datasets/10_10_after_fix_pytest')),
        prepare_triplet_dataset(instance_dict, snippet_paths),
        *np.array_split(np.array(triplet_data), 2),
        tf.data.Dataset.from_tensor_slices(train_data.tolist()).batch(32).shuffle(len(train_data)),
        tf.data.Dataset.from_tensor_slices(valid_data.tolist()).batch(32),
        TripletModel(vocab_size=len(train_loader.dataset.data.tokenizer.classes_) + 1, embedding_dim=128),
        train_model(model, train_loader, valid_loader, epochs=5))
    )

if __name__ == "__main__":
    main()