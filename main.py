import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

ModelConfiguration = lambda: {
    "model_name": "t5-base", "alpha": 16, "dropout_rate": 0.1, "decomposition_rank": 64,
    "layers_to_modify": ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    "quantization_config": {"nested_quantization": False, "four_bit_dtype": "float16", "storage_dtype": "uint8",
                            "quantization_strategy": "nf4", "flash_attention": False, "low_rank_peft": False,
                            "eight_bit_quantization": False, "four_bit_quantization": False, "reentrant_training": False,
                            "unsloth_training": False},
    "use_triplet_loss": True, "data_source": "timdettmers/openassistant-guanaco",
    "token_flags": {"use_special_token": False, "include_special_tokens": False}, "data_splits": ["train", "test"],
    "tokenized_file": None, "results_dir": "./results", "epochs": 3, "batch_sizes": {"train": 16, "eval": 64},
    "warmup_steps": 500, "weight_decay": 0.01, "logging_dir": "./logs", "model_save_interval": 500,
    "max_checkpoints": 2, "seed": 42, "checkpoint_path": None, "negative_samples_per_batch": 5
}

SentenceEmbedder = lambda embedding_dim, vocab_size: nn.Sequential(
    nn.Embedding(vocab_size, embedding_dim),
    nn.LSTM(embedding_dim, embedding_dim, batch_first=True),
    lambda x: x[0][:, -1, :],
    nn.Linear(embedding_dim, embedding_dim),
    nn.Linear(embedding_dim, vocab_size)
)

calculate_triplet_loss = lambda anchor, positive, negative: torch.mean(
    torch.maximum(torch.mean(torch.square(anchor - positive), dim=-1) - 
    torch.mean(torch.square(anchor - negative), dim=-1) + 2.0, torch.tensor(0.0))
)

SentenceDataset = lambda data, config, tokenizer: (
    lambda indices=list(range(len(data))): (
        lambda __len__=lambda: len(data),
        __getitem__=lambda idx: (
            torch.tensor(tokenizer.encode(data[indices[idx]]['input'], max_length=512, padding='max_length', truncation=True)),
            torch.tensor(tokenizer.encode(data[indices[idx]]['output'], max_length=512, padding='max_length', truncation=True)),
            torch.tensor([tokenizer.encode(data[random.choice([j for j in range(len(data)) if j != indices[idx]])]['input'],
                          max_length=512, padding='max_length', truncation=True) for _ in range(config["negative_samples_per_batch"])])
        )
    )()
)

load_data = lambda file_path: json.load(open(file_path, 'r')) if os.path.exists(file_path) else print(f"File not found: {file_path}") or None

initialize_t5_tokenizer = lambda: T5Tokenizer.from_pretrained("t5-base")

configure_optimizer = lambda model: optim.Adam(model.parameters(), lr=0.001)

setup_training = lambda model, optimizer: (model, optimizer, calculate_triplet_loss)

run_training = lambda model, config, data_loader, optimizer, loss_function: (
    model.train(),
    [(
        [(
            optimizer.zero_grad(),
            loss_function(model(input_ids), labels, neg_samples).backward(),
            optimizer.step()
        ) for input_ids, labels, neg_samples in data_loader],
        early_stopping(loss.item()) and print("Early stopping triggered.") or None
    ) for epoch in range(config["epochs"])]
)

run_evaluation = lambda model, data_loader, loss_function: (
    model.eval(),
    print(f"Mean Evaluation Loss: {sum(loss_function(model(input_ids), labels, neg_samples).item() for input_ids, labels, neg_samples in data_loader) / len(data_loader):.4f}")
)

save_trained_model = lambda model, file_path: torch.save(model.state_dict(), file_path)

save_training_history = lambda history, file_path: json.dump(history, open(file_path, 'w'))

add_learning_rate_scheduler = lambda optimizer, config: optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

configure_early_stopping = lambda patience=3: (
    lambda best_loss=float('inf'), counter=0: lambda current_loss: (
        (current_loss < best_loss and (best_loss := current_loss) or counter := 0) or (counter := counter + 1),
        counter >= patience
    )[-1]
)()

execute_training = lambda: (
    lambda config=ModelConfiguration(),
           train_data=load_data("train.json"),
           test_data=load_data("test.json"),
           tokenizer=initialize_t5_tokenizer(),
           train_dataset=SentenceDataset(train_data, config, tokenizer),
           test_dataset=SentenceDataset(test_data, config, tokenizer),
           train_loader=DataLoader(train_dataset, batch_size=config["batch_sizes"]['train'], shuffle=True),
           test_loader=DataLoader(test_dataset, batch_size=config["batch_sizes"]['eval'], shuffle=False),
           model=SentenceEmbedder(128, 30522),
           optimizer=configure_optimizer(model),
           model, optimizer, loss_function=setup_training(model, optimizer),
           scheduler=add_learning_rate_scheduler(optimizer, config): (
        run_training(model, config, train_loader, optimizer, loss_function),
        run_evaluation(model, test_loader, loss_function),
        save_trained_model(model, os.path.join(config["results_dir"], "triplet_model.pth"))
    )
)()

if __name__ == "__main__":
    execute_training()