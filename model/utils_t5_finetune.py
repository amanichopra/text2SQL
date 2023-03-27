import pandas as pd
import seaborn as sns
from pynvml import *

sns.set_style("darkgrid")


def print_gpu_utilization():
    """
    print gpu utilization
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    """
    print summary
    """
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def map_to_length(x):
    """
    map article and summary len to dict as well as if sample is longer than 512 tokens
    """
    from transformers import AutoTokenizer

    tokenizer_eda = AutoTokenizer.from_pretrained("t5-base")
    x["input_len"] = len(tokenizer_eda(x["input"]).input_ids)
    x["input_longer_256"] = int(x["input_len"] > 256)
    x["input_longer_128"] = int(x["input_len"] > 128)
    x["input_longer_64"] = int(x["input_len"] > 64)
    x["out_len"] = len(tokenizer_eda(x["target"]).input_ids)
    x["out_longer_256"] = int(x["out_len"] > 256)
    x["out_longer_128"] = int(x["out_len"] > 128)
    x["out_longer_64"] = int(x["out_len"] > 64)
    return x


def format_eda_dataset_wikisql(example):
    """
    format dataset for eda purpose
    """
    return {"input": example["question"], "target": example["sql"]["human_readable"]}


def format_eda_dataset_cosql(example):
    return {"input": example["input"], "target": example["target"]}


def format_dataset_wikisql(example):
    """
    format dataset for training purpose
    """
    return {
        "input": "translate to SQL: " + example["question"],
        "target": example["sql"]["human_readable"],
    }


def format_dataset_cosql(example):
    return {
        "input": "translate to SQL: " + example["input"],
        "target": example["target"],
    }


def format_dataset_spider(example):
    return {
        "input": "translate to SQL: " + example["question"],
        "target": example["query"],
    }


def preprocess_cosql(dataset, scope: str) -> pd.DataFrame:
    scope_valid = ["all", "final", "interaction"]
    if scope not in scope_valid:
        raise ValueError(f"scope must be one of {scope_valid}")

    processed_input = []
    processed_target = []
    for dialog in dataset:
        if scope in ["all", "final"]:
            processed_input.append(dialog["final"]["utterance"])
            processed_target.append(dialog["final"]["query"])
        if scope in ["all", "interaction"]:
            for turn in dialog["interaction"]:
                processed_input.append(turn["utterance"])
                processed_target.append(turn["query"])

    processed_dataset = pd.DataFrame(
        {"input": processed_input, "target": processed_target}
    )

    return processed_dataset
