import json

from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)
from utils_t5_finetune import (
    format_dataset_cosql,
    preprocess_cosql,
    print_gpu_utilization,
)

# global variables
MODEL_VERSIONS = ["base", "small"]
FINETUNED_ON = "wikisql"
FINETUNE_PATH = "wikisql_cosql"

OVERWRITE_OUTPUT_DIR = True
DO_TRAIN = True
DO_EVAL = True
EVALUATION_STRATEGY = "epoch"
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 10
LOG_LEVEL = "info"
LOGGING_STRATEGY = "epoch"
# LOGGING_STEPS=100
SAVE_STRATEGY = "epoch"
# SAVE_STEPS=1000
SAVE_TOTAL_LIMIT = 3
# EVAL_STEPS=1000
LOAD_BEST_MODEL_AT_END = True
REPORT_TO = "all"
SKIP_MEMORY_METRICS = True
PUSH_TO_HUB = False
PREDICT_WITH_GENERATE = True


# helper functions
def convert_to_features(example_batch):
    """
    tokenizer the dataset
    """
    input_encodings = tokenizer.batch_encode_plus(
        example_batch["input"], pad_to_max_length=True, max_length=64
    )
    target_encodings = tokenizer.batch_encode_plus(
        example_batch["target"], pad_to_max_length=True, max_length=64
    )

    encodings = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
        "decoder_attention_mask": target_encodings["attention_mask"],
    }

    return encodings


def compute_metrics_rogue2(pred):
    """
    function to compute metrics
    """
    rouge = load_metric("rouge")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# load data
with open("data\cosql_dataset\sql_state_tracking\cosql_train.json", "r") as f:
    train_data = json.load(f)
with open("data\cosql_dataset\sql_state_tracking\cosql_dev.json", "r") as f:
    test_data = json.load(f)

train_df_final = preprocess_cosql(train_data, scope="final")
test_df_final = preprocess_cosql(test_data, scope="final")

train_data = Dataset.from_pandas(train_df_final)
test_data = Dataset.from_pandas(test_df_final)

# model
for MODEL_VERSION in MODEL_VERSIONS:
    CKPT = f"output/t5_{MODEL_VERSION}_{FINETUNED_ON}"
    OUTPUT_DIR = f"output/t5_{MODEL_VERSION}_{FINETUNE_PATH}"
    LOGGING_DIR = f"output/t5_{MODEL_VERSION}_{FINETUNE_PATH}/log"

    model = T5ForConditionalGeneration.from_pretrained(CKPT).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    print_gpu_utilization()

    train_dataset = train_data.map(
        format_dataset_cosql, remove_columns=train_data.column_names
    )
    test_dataset = test_data.map(
        format_dataset_cosql, remove_columns=test_data.column_names
    )

    train_dataset = train_dataset.map(
        convert_to_features, batched=True, remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        convert_to_features, batched=True, remove_columns=test_dataset.column_names
    )

    columns = ["input_ids", "attention_mask", "labels", "decoder_attention_mask"]

    train_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    # set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=OVERWRITE_OUTPUT_DIR,
        do_train=DO_TRAIN,
        do_eval=DO_EVAL,
        evaluation_strategy=EVALUATION_STRATEGY,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        log_level=LOG_LEVEL,
        logging_dir=LOGGING_DIR,
        logging_strategy="epoch",
        # logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        # save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        # eval_steps=EVAL_STEPS,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        # report_to=REPORT_TO,
        skip_memory_metrics=SKIP_MEMORY_METRICS,
        push_to_hub=PUSH_TO_HUB,
        predict_with_generate=PREDICT_WITH_GENERATE,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_rogue2,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # training
    train_result = trainer.train()

    # compute train results
    metrics = train_result.metrics
    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    # save train results
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # compute evaluation results
    metrics = trainer.evaluate()
    max_val_samples = len(test_dataset)
    metrics["eval_samples"] = min(max_val_samples, len(test_dataset))

    # save evaluation results
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # save model checkpoint and tokenizer
    trainer.save_model(output_dir=OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
