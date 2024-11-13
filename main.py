import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@dataclass
class ModelConfig:
    model_identifier: str = field(default="t5-base")
    chat_template: Optional[str] = field(default="none")
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_rank: Optional[int] = field(default=64)
    lora_target_layers: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj")
    nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_storage_dtype: Optional[str] = field(default="uint8")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    use_flash_attention: Optional[bool] = field(default=False)
    use_peft_lora: Optional[bool] = field(default=False)
    use_8bit_quantization: Optional[bool] = field(default=False)
    use_4bit_quantization: Optional[bool] = field(default=False)
    use_reentrant: Optional[bool] = field(default=False)
    use_unsloth: Optional[bool] = field(default=False)
    use_triplet_loss_trainer: Optional[bool] = field(default=False)

@dataclass
class TrainingDataConfig:
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco")
    append_concat_token: Optional[bool] = field(default=False)
    add_special_tokens: Optional[bool] = field(default=False)
    splits: Optional[str] = field(default="train,test")
    tokenized_dataset_path: Optional[str] = field(default=None)

@dataclass
class TrainingConfig:
    output_dir: str = field(default="./results")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=64)
    warmup_steps: int = field(default=500)
    weight_decay: float = field(default=0.01)
    logging_dir: str = field(default="./logs")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=2)
    seed: int = field(default=42)
    resume_from_checkpoint: Optional[str] = field(default=None)

class Dataset(Dataset):
    def __init__(self, data, batch_size, num_negative_samples):
        self.data = data
        self.batch_size = batch_size
        self.num_negative_samples = num_negative_samples
        self.indices = list(range(len(data)))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.data[i] for i in batch_indices]
        positive_samples = [sample for sample in batch if sample["label"] == 1]
        negative_samples = [sample for sample in batch if sample["label"] == 0]
        return positive_samples, negative_samples

    def epoch_shuffle(self):
        import random
        random.shuffle(self.indices)

class Base(torch.nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

class BaseModel(Base):
    def __init__(self):
        super().__init__()

class Model(BaseModel):
    pass

class ModelHandler:
    def __init__(self, model_args, training_args):
        self.model_args = model_args
        self.training_args = training_args

    def create_model(self):
        return Model()

    def create_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=0.001)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, path):
        model = Model()
        model.load_state_dict(torch.load(path))
        return model

class DataHandler:
    def __init__(self, model_args, data_args):
        self.model_args = model_args
        self.data_args = data_args

    def load_data(self, file_name):
        return json.load(open(file_name))

    def process_data(self, examples):
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        apply_chat_template = self.model_args.chat_template != "none"
        if apply_chat_template:
            inputs = []
            labels = []
            for example in examples:
                inputs.append(f"{example['input']} {tokenizer.sep_token}")
                labels.append(f"{example['output']} {tokenizer.sep_token}")
            examples["input_ids"] = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids
            examples["labels"] = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).input_ids
            examples["attention_mask"] = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).attention_mask
        else:
            examples["input_ids"] = tokenizer([example["input"] for example in examples], return_tensors="pt", padding=True, truncation=True).input_ids
            examples["labels"] = tokenizer([example["output"] for example in examples], return_tensors="pt", padding=True, truncation=True).input_ids
            examples["attention_mask"] = tokenizer([example["input"] for example in examples], return_tensors="pt", padding=True, truncation=True).attention_mask
        return examples

    def prepare_datasets(self):
        train_data = self.load_data("train.json")
        test_data = self.load_data("test.json")
        train_data = self.process_data(train_data)
        test_data = self.process_data(test_data)
        return train_data, test_data

    def create_data_loaders(self):
        train_data, test_data = self.prepare_datasets()
        train_dataset = Dataset(train_data, self.data_args.per_device_train_batch_size, 5)
        test_dataset = Dataset(test_data, self.data_args.per_device_eval_batch_size, 5)
        return DataLoader(train_dataset, batch_size=None), DataLoader(test_dataset, batch_size=None)

class Trainer:
    def __init__(self, model, train_dataset, optimizer):
        self.model = model
        self.train_dataset = train_dataset
        self.optimizer = optimizer

    def train_step(self, batch):
        positive_samples, negative_samples = batch
        input_ids = torch.cat([sample["input_ids"] for sample in positive_samples + negative_samples])
        attention_mask = torch.cat([sample["attention_mask"] for sample in positive_samples + negative_samples])
        labels = torch.cat([sample["labels"] for sample in positive_samples + negative_samples])
        self.optimizer.zero_grad()
        self.model.zero_grad()
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs, save_path):
        for epoch in range(epochs):
            self.train_dataset.dataset.epoch_shuffle()
            total_loss = 0
            for batch in self.train_dataset:
                loss = self.train_step(batch)
                total_loss += loss
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.train_dataset)}")
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))

def run_pipeline(model_args, data_args, training_args):
    model_handler = ModelHandler(model_args, training_args)
    data_handler = DataHandler(model_args, data_args)
    model = model_handler.create_model()
    optimizer = model_handler.create_optimizer(model)
    train_dataset, _ = data_handler.create_data_loaders()
    trainer = Trainer(model, train_dataset, optimizer)
    trainer.train(training_args.num_train_epochs, training_args.output_dir)

def resume_pipeline(model_args, data_args, training_args, checkpoint_path):
    model_handler = ModelHandler(model_args, training_args)
    data_handler = DataHandler(model_args, data_args)
    model = model_handler.load_model(checkpoint_path)
    optimizer = model_handler.create_optimizer(model)
    train_dataset, _ = data_handler.create_data_loaders()
    trainer = Trainer(model, train_dataset, optimizer)
    trainer.train(training_args.num_train_epochs, training_args.output_dir)

if __name__ == "__main__":
    model_args = ModelConfig(model_identifier="t5-base", chat_template="none")
    data_args = TrainingDataConfig(dataset_name="timdettmers/openassistant-guanaco")
    training_args = TrainingConfig(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)

    run_pipeline(model_args, data_args, training_args)
    #resume_pipeline(model_args, data_args, training_args, "./results/model.pth")