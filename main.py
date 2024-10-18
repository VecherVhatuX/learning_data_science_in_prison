"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailments are positive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v2.py

OR
python training_nli_v2.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
from datetime import datetime
import os

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType

import torch
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning


from datasets import Dataset, load_dataset
from pathlib import Path
import os, json
from tqdm import tqdm 

import pickle

from random import sample
from datasets import Dataset
from sentence_transformers import InputExample

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, PrefixTuningConfig

def disable_ssl_warnings():
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    original_request = requests.Session.request
    def patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_request(self, *args, **kwargs)
    requests.Session.request = patched_request
    
disable_ssl_warnings()

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

data_path = os.environ.get('DATA_PATH', '/home/ma-user/data/data.pkl')
output_path = os.environ.get('OUTPUT_PATH', '/tmp/output')
model_path = os.environ.get('MODEL_PATH', '/home/ma-user/bert-base-uncased')

n = torch.cuda.device_count()
print(f"There are {n} GPUs available for torch.")
for i in range(n):
  name = torch.cuda.get_device_name(i)
  print(f"GPU {i}: {name}")

if not model_path:
    model_path =  "bert-base-uncased"#"google-t5/t5-small"

train_batch_size = 64  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 75
num_epochs = 1

# with open('data.pkl', 'wb') as f:
#     pickle.dump(all_dataset, f)
    
with open(data_path, 'rb') as f:
    all_dataset = pickle.load(f)

train_data = all_dataset[0:61]
eval_data = all_dataset[61:]


def prepare_dataset(data, negative_sample_size=3):
    dataset_list = []
    for item in data:
        query = item['query']
        relevant_doc = item['relevant_doc']
        # Берем случайное количество нерелевантных документов
        non_relevant_docs = sample(item['irrelevant_docs'], min(len(item['irrelevant_docs']), negative_sample_size))
        # if query and relevant_doc:
        #     dataset_list.append({
        #             "anchor": query,
        #             "positive": relevant_doc
        #         })
        for item in non_relevant_docs:
            dataset_list.append({
                "anchor": query,
                "positive": relevant_doc,
                "negative": item
            })

    return dataset_list

# Готовим данные
train_data = prepare_dataset(train_data, negative_sample_size=10)
eval_data = prepare_dataset(eval_data, negative_sample_size=10)

# Преобразуем в Dataset
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

print(len(train_dataset), len(eval_dataset))
model_name = Path(model_path).stem

# Save path of the model
output_dir = (
    output_path + "/output/training_nli_v2_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


prefix_tuning_config = PrefixTuningConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, num_virtual_tokens=10)

sentence_transformer = SentenceTransformer(model_path)
print(sentence_transformer)
print(sentence_transformer.max_seq_length, sentence_transformer[0].auto_model.config.max_position_embeddings )

from peft import LoraConfig, TaskType, get_peft_model
peft_config = LoraConfig(
    target_modules=["dense"],
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType

# peft_config = PromptTuningConfig(
#         task_type=TaskType.FEATURE_EXTRACTION,
#         prompt_tuning_init=PromptTuningInit.RANDOM,
#         num_virtual_tokens=1
#     )

sentence_transformer._modules["0"].auto_model = get_peft_model(
    sentence_transformer._modules["0"].auto_model, peft_config
)
print(sentence_transformer.max_seq_length, sentence_transformer[0].auto_model.config.max_position_embeddings )
model = sentence_transformer

model.train()

train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(10000))
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev").select(range(1000))

train_loss = losses.MultipleNegativesRankingLoss(model)

    

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=20,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    logging_steps=100,
    run_name="nli-v2",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss
)

trainer.train()


with open(output_path, 'wb') as w:
    w.write('asddddddddddddddddd')
