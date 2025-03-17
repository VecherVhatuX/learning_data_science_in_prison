import os
import json
import random
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from transformers import T5Tokenizer

# Configuration for the model, including hyperparameters, paths, and settings.
ModelConfigurations = lambda: {
    "model_name": "t5-base",  # Name of the pre-trained model to use.
    "alpha": 16,  # Alpha parameter for model configuration.
    "dropout_rate": 0.1,  # Dropout rate for regularization.
    "decomposition_rank": 64,  # Rank for decomposition techniques.
    "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],  # Layers to modify.
    "quantization_config": {  # Configuration for quantization techniques.
        "nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
        "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
        "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
        "unsloth_training": False,
    },
    "use_triplet_loss": True,  # Whether to use triplet loss for training.
    "data_source": "timdettmers/openassistant-guanaco",  # Source of the dataset.
    "token_flags": {"use_special_token": False, "include_special_tokens": False},  # Tokenization flags.
    "data_splits": ["train", "test"],  # Data splits for training and testing.
    "tokenized_file": None,  # Path to tokenized file, if any.
    "results_dir": "./results",  # Directory to save results.
    "epochs": 3,  # Number of training epochs.
    "batch_sizes": {"train": 16, "eval": 64},  # Batch sizes for training and evaluation.
    "warmup_steps": 500,  # Number of warmup steps for learning rate scheduling.
    "weight_decay": 0.01,  # Weight decay for regularization.
    "logging_dir": "./logs",  # Directory to save logs.
    "model_save_interval": 500,  # Interval to save model checkpoints.
    "max_checkpoints": 2,  # Maximum number of checkpoints to keep.
    "seed": 42,  # Random seed for reproducibility.
    "checkpoint_path": None,  # Path to load checkpoints, if any.
    "negative_samples_per_batch": 5  # Number of negative samples per batch for triplet loss.
}

# A sequential model for text embedding, consisting of an embedding layer, LSTM, and dense layers.
TextEmbeddingModel = lambda embedding_dim, vocab_size: tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim),  # Embedding layer to convert tokens to vectors.
    layers.LSTM(embedding_dim, return_sequences=True),  # LSTM layer to process sequences.
    lambda x: x[:, -1, :],  # Extract the last hidden state of the LSTM.
    layers.Dense(embedding_dim),  # Dense layer to further process the embeddings.
    layers.Dense(vocab_size)  # Final dense layer to predict the next token.
])

# Function to calculate triplet loss, which encourages the model to learn embeddings where the anchor is closer to the positive than to the negative.
calculate_triplet_loss = lambda anchor, positive, negative: tf.reduce_mean(
    tf.maximum(tf.reduce_mean(tf.square(anchor - positive), axis=-1) - 
    tf.reduce_mean(tf.square(anchor - negative), axis=-1) + 2.0, 0.0)
)

# Handler for managing the dataset, including shuffling, fetching samples, and creating batches.
DatasetHandler = lambda data, config, tokenizer: {
    "data": data,  # The dataset.
    "config": config,  # Configuration for the dataset handler.
    "tokenizer": tokenizer,  # Tokenizer to encode text data.
    "indices": list(range(len(data))),  # Indices of the dataset for shuffling.
    "shuffle_dataset": lambda self: random.shuffle(self["indices"]),  # Shuffle the dataset indices.
    "fetch_sample": lambda self, idx: (  # Fetch a sample from the dataset.
        self["tokenizer"].encode(self["data"][self["indices"][idx]]['input'], max_length=512, padding='max_length', truncation=True),
        self["tokenizer"].encode(self["data"][self["indices"][idx]]['output'], max_length=512, padding='max_length', truncation=True),
        self["fetch_negative_samples"](idx)
    ),
    "fetch_negative_samples": lambda self, idx: [  # Fetch negative samples for triplet loss.
        self["tokenizer"].encode(self["data"][random.choice([j for j in range(len(self["data"])) if j != self["indices"][idx]])]['input'],
        max_length=512, padding='max_length', truncation=True) 
        for _ in range(self["config"]["negative_samples_per_batch"])
    ],
    "create_batch": lambda self: [self["fetch_sample"](i) for i in range(len(self["data"]))]  # Create a batch of samples.
}

# Adapter for loading data in batches, compatible with TensorFlow's data pipeline.
DataLoaderAdapter = lambda dataset, config, tokenizer: {
    "dataset": DatasetHandler(dataset, config, tokenizer),  # Dataset handler.
    "batch_size": config["batch_sizes"]['train'],  # Batch size for training.
    "__len__": lambda self: len(self["dataset"]["data"]) // self["batch_size"],  # Number of batches.
    "__getitem__": lambda self, index: tuple(tf.convert_to_tensor(x) for x in zip(*self["dataset"]["create_batch"]()[index * self["batch_size"]:(index + 1) * self["batch_size"]]))  # Get a batch of data.
}

# Function to load data from a JSON file.
load_data = lambda file_path: json.load(open(file_path, 'r')) if os.path.exists(file_path) else print(f"File not found: {file_path}")

# Function to initialize the T5 tokenizer.
initialize_tokenizer = lambda: T5Tokenizer.from_pretrained("t5-base")

# Function to configure the optimizer (Adam) with a learning rate.
configure_optimizer = lambda model: optimizers.Adam(learning_rate=0.001)

# Function to set up the training process, including the model, optimizer, and loss function.
setup_training = lambda model, optimizer: (model, optimizer, calculate_triplet_loss)

# Function to execute the training loop, including forward pass, loss calculation, and backpropagation.
execute_training = lambda model, config, data_loader, optimizer, loss_function: (
    model.train(),
    add_early_stopping(patience=3),
    [(
        [(
            (input_ids, labels, neg_samples) := batch,
            (outputs := model(input_ids)),
            (loss := loss_function(outputs, labels, neg_samples)),
            (gradients := tf.GradientTape().gradient(loss, model.trainable_variables)),
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        ) for batch in data_loader],
        early_stopping(loss.numpy()) and print("Early stopping triggered.") and break
    ) for epoch in range(config["epochs"])]
)

# Function to assess the model's performance on the test dataset.
assess_model = lambda model, data_loader, loss_function: (
    model.eval(),
    print(f"Mean Evaluation Loss: {sum(loss_function(model(input_ids), labels, neg_samples).numpy() for input_ids, labels, neg_samples in data_loader) / len(data_loader):.4f}")
)

# Function to store the trained model's weights to a file.
store_model = lambda model, file_path: model.save_weights(file_path)

# Function to store training logs (e.g., loss history) to a JSON file.
store_training_logs = lambda history, file_path: json.dump(history, open(file_path, 'w'))

# Function to add a learning rate scheduler to the optimizer.
add_learning_rate_scheduler = lambda optimizer, config: optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)

# Function to add early stopping to the training process.
add_early_stopping = lambda patience=3: (
    lambda current_loss: (
        (current_loss < best_loss) and (best_loss := current_loss) and (counter := 0) or
        (counter := counter + 1) and (counter >= patience)
    ) if 'best_loss' in locals() else (best_loss := float('inf'), counter := 0, lambda current_loss: (
        (current_loss < best_loss) and (best_loss := current_loss) and (counter := 0) or
        (counter := counter + 1) and (counter >= patience)
    )
)

# Function to initiate the training process, including data loading, model initialization, and training execution.
initiate_training = lambda: (
    (config := ModelConfigurations()),
    (train_data := load_data("train.json")),
    (test_data := load_data("test.json")),
    (tokenizer := initialize_tokenizer()),
    (train_loader := DataLoaderAdapter(train_data, config, tokenizer)),
    (test_loader := DataLoaderAdapter(test_data, config, tokenizer)),
    (model := TextEmbeddingModel(128, 30522)),
    (optimizer := configure_optimizer(model)),
    (model, optimizer, loss_function := setup_training(model, optimizer)),
    (scheduler := add_learning_rate_scheduler(optimizer, config)),
    execute_training(model, config, train_loader, optimizer, loss_function),
    assess_model(model, test_loader, loss_function),
    store_model(model, os.path.join(config["results_dir"], "triplet_model.h5"))
)

if __name__ == "__main__":
    initiate_training()