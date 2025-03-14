import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import T5Tokenizer

ModelConfig = lambda: type('ModelConfig', (), {
    "model_name": "t5-base",
    "alpha": 16,
    "dropout_rate": 0.1,
    "decomposition_rank": 64,
    "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    "quantization_config": {
        "nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
        "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
        "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
        "unsloth_training": False,
    },
    "use_triplet_loss": True,
    "data_source": "timdettmers/openassistant-guanaco",
    "token_flags": {"use_special_token": False, "include_special_tokens": False},
    "data_splits": ["train", "test"],
    "tokenized_file": None,
    "results_dir": "./results",
    "epochs": 3,
    "batch_sizes": {"train": 16, "eval": 64},
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "model_save_interval": 500,
    "max_checkpoints": 2,
    "seed": 42,
    "checkpoint_path": None,
    "negative_samples_per_batch": 5
})()

create_triplet_network = lambda embedding_size, vocab_count: Model(
    inputs := layers.Input(shape=(None,)),
    layers.Dense(vocab_count)(layers.Dense(embedding_size)(
        layers.LSTM(embedding_size, return_sequences=True)(
            layers.Embedding(vocab_count, embedding_size)(inputs)
        )[:, -1, :]
    ))
)

calculate_triplet_loss = lambda anchor, positive, negative: tf.reduce_mean(
    tf.maximum(tf.reduce_mean(tf.square(anchor - positive), axis=-1) - 
               tf.reduce_mean(tf.square(anchor - negative), axis=-1) + 2.0, 0.0)
)

DataHandler = lambda data, config, tokenizer: type('DataHandler', (), {
    "data": data,
    "config": config,
    "tokenizer": tokenizer,
    "indices": list(range(len(data))),
    "randomize": lambda self: random.shuffle(self.indices),
    "fetch_sample": lambda self, idx: (
        self.tokenizer.encode(self.data[self.indices[idx]]['input'], max_length=512, padding='max_length', truncation=True),
        self.tokenizer.encode(self.data[self.indices[idx]]['output'], max_length=512, padding='max_length', truncation=True),
        self.fetch_negative_samples(idx)
    ),
    "fetch_negative_samples": lambda self, idx: [
        self.tokenizer.encode(self.data[random.choice([j for j in range(len(self.data)) if j != self.indices[idx]])]['input'],
                             max_length=512, padding='max_length', truncation=True) 
        for _ in range(self.config.negative_samples_per_batch)
    ],
    "generate_batch_samples": lambda self: [self.fetch_sample(i) for i in range(len(self.data))]
})(data, config, tokenizer)

BatchGenerator = lambda dataset, config, tokenizer: type('BatchGenerator', (tf.keras.utils.Sequence,), {
    "dataset": DataHandler(dataset, config, tokenizer),
    "batch_size": config.batch_sizes['train'],
    "__len__": lambda self: len(self.dataset.data) // self.batch_size,
    "__getitem__": lambda self, index: tuple(np.array(x) for x in zip(*self.dataset.generate_batch_samples()[index * self.batch_size:(index + 1) * self.batch_size]))
})(dataset, config, tokenizer)

load_data_file = lambda file_path: json.load(open(file_path, 'r')) if os.path.exists(file_path) else (print(f"File not found: {file_path}"), None)

fetch_tokenizer = lambda: T5Tokenizer.from_pretrained("t5-base")

create_optimizer = lambda: tf.keras.optimizers.Adam(learning_rate=0.001)

configure_training = lambda model, optimizer: model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: calculate_triplet_loss(y_true[0], y_true[1], y_pred))

train_model = lambda model, config, data_loader: model.fit(data_loader, epochs=config.epochs, callbacks=[add_early_stopping()])

assess_model = lambda model, data_loader: print(f"Mean Evaluation Loss: {sum(calculate_triplet_loss(model(input_ids), model(labels), model(neg_samples)).numpy() for input_ids, labels, neg_samples in data_loader) / len(data_loader):.4f}")

store_weights = lambda model, file_path: model.save_weights(file_path)

store_training_history = lambda history, file_path: json.dump(history.history, open(file_path, 'w'))

add_lr_scheduler = lambda optimizer, config: tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9))

add_early_stopping = lambda: tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

execute_training = lambda: (
    (config := ModelConfig()),
    (train_data := load_data_file("train.json")),
    (test_data := load_data_file("test.json")),
    (tokenizer := fetch_tokenizer()),
    (train_loader := BatchGenerator(train_data, config, tokenizer)) if train_data and test_data else None,
    (test_loader := BatchGenerator(test_data, config, tokenizer)) if train_data and test_data else None,
    (model := create_triplet_network(128, 30522)) if train_data and test_data else None,
    (optimizer := add_lr_scheduler(create_optimizer(), config)) if train_data and test_data else None,
    configure_training(model, optimizer) if train_data and test_data else None,
    train_model(model, config, train_loader) if train_data and test_data else None,
    assess_model(model, test_loader) if train_data and test_data else None,
    store_weights(model, os.path.join(config.results_dir, "triplet_model.h5")) if train_data and test_data else None
)

if __name__ == "__main__":
    execute_training()