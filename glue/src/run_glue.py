import logging
import os
import random
import sys

import datasets
import evaluate
import numpy as np
import torch
from arguments import DataTrainingArguments, ModelArguments
from checkpoint_utils import cleanup_checkpoints
from data_utils import (configure_tokenization, determine_task_type_and_labels,
                        identify_text_fields, load_data,
                        prepare_label_id_mapping, preprocess_function)
from model_utils import DEFAULT_PAD_TOKEN, smart_tokenizer_and_embedding_resize
from train_utils import train_model

import transformers
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding,
                          EvalPrediction, HfArgumentParser, LlamaTokenizer,
                          default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

current_dir = os.path.dirname(__file__)
loraplus_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Append the 'loraplus' directory to sys.path
sys.path.append(loraplus_dir)
from loraplus import LoraPlusTrainer, LoraPlusTrainingArguments

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s,%(msecs)03d >> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set logging verbosity for transformers and datasets libraries to INFO
    transformers.utils.logging.set_verbosity_info()
    datasets.utils.logging.set_verbosity_info()

    # Explicitly enable the default logging handler and format
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_data(data_args, model_args, training_args)
    is_regression, num_labels, label_list = determine_task_type_and_labels(
        raw_datasets, data_args
    )

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
    )

    torch_dtype = (
        torch.float32
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        torch_dtype=torch_dtype,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if "gpt" in config.name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if "llama" in config.name_or_path or isinstance(tokenizer, LlamaTokenizer):
        if tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is None, `<unk>` token is id 0.
        logger.info("Adding special tokens.")
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": "<unk>",
            }
        )

    if training_args.use_lora:
        target_modules = training_args.target_modules
        assert target_modules is not None
        target_modules = target_modules.split(",")
        target_modules = [target_module.strip() for target_module in target_modules]
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            fan_in_fan_out=True,
            target_modules=target_modules,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            use_original_init=training_args.lora_use_original_init,
            modules_to_save=["classifier", "score"],
        )

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model = get_peft_model(model, peft_config)

        logger.info("Training using LoRA!")
        model.print_trainable_parameters()

    # Manually freeze dense layer in RoBERTa
    if "roberta" in config.name_or_path:
        for name, param in model.named_parameters():
            if "classifier" in name and "dense" in name:
                param.requires_grad = False

    max_seq_length, padding = configure_tokenization(tokenizer, data_args)
    label_to_id = prepare_label_id_mapping(
        model, data_args, is_regression, label_list, num_labels
    )
    sentence_key1, sentence_key2 = identify_text_fields(raw_datasets, data_args)

    preprocess_wrapper = lambda examples: preprocess_function(
        examples,
        tokenizer,
        max_seq_length,
        padding,
        label_to_id,
        sentence_key1,
        sentence_key2,
    )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_wrapper,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[
            "validation_matched" if data_args.task_name == "mnli" else "validation"
        ]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = LoraPlusTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_model(trainer, training_args, data_args, train_dataset, last_checkpoint=last_checkpoint)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(valid_mm_dataset), data_args.max_eval_samples
                )
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics(
                "eval", combined if task is not None and "mnli" in task else metrics
            )
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # cleanup checkpoints
    cleanup_checkpoints(trainer, training_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
