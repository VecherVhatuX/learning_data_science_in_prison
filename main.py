import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, set_seed, TrainingArguments
from trl import SFTConfig, SFTTrainer
from utils import create_and_prepare_model, create_datasets
from our_trainers import TripletLossTrainer
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import requests
from functools import partial


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    use_triplet_loss_trainer: Optional[bool] = field(
        default=False,
        metadata={"help": "Use our TripletLossTrainer(Trainer)"},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    tokenized_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenized dataset on disk."},
    )


def disable_ssl_warnings():
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    original_request = requests.Session.request

    def patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_request(self, *args, **kwargs)

    requests.Session.request = patched_request


def set_seed_func(seed):
    return partial(set_seed, seed)


def prepare_model_func(model_args, data_args, training_args):
    return partial(create_and_prepare_model, model_args, data_args, training_args)


def create_datasets_func(tokenizer, data_args, training_args):
    return partial(create_datasets, tokenizer, data_args, training_args, apply_chat_template=data_args.chat_template_format != "none")


def get_trainer_func(model, peft_config, train_dataset, eval_dataset, model_args, training_args):
    if model_args.use_triplet_loss_trainer:
        model = get_peft_model(model, peft_config)
        return TripletLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            layer_index=-1,
        )
    else:
        return SFTTrainer(
            model=model,
            tokenizer=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )


def train_func(trainer, checkpoint):
    trainer.train(resume_from_checkpoint=checkpoint)


def save_model_func(trainer):
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        config_file = os.path.abspath(sys.argv[1])
        if config_file.endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=config_file)
        else:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            model_args = ModelArguments(**config.get("ModelArguments", {}))
            data_args = DataTrainingArguments(**config.get("DataTrainingArguments", {}))
            training_args = TrainingArguments(**config.get("TrainingArguments", {}))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    disable_ssl_warnings()

    pipeline = [
        set_seed_func(training_args.seed),
        prepare_model_func(model_args, data_args, training_args),
        lambda x: x[0] if data_args.tokenized_dataset_path else create_datasets_func(x[2], data_args, training_args)(),
        lambda x: get_trainer_func(x[0], x[1], x[2], x[3], model_args, training_args),
        lambda x: train_func(x, training_args.resume_from_checkpoint),
        lambda x: save_model_func(x),
    ]

    result = None
    for func in pipeline:
        try:
            result = func(result)
        except TypeError:
            result = func(*result)

    return result


if __name__ == "__main__":
    main()