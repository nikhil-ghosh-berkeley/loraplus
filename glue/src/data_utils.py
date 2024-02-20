import logging
import os

from datasets import load_dataset, load_from_disk
from transformers.utils import logging

from transformers import PretrainedConfig

logger = logging.get_logger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_data(data_args, model_args, training_args):
    """
    Loads the dataset based on the provided arguments. Supports loading from the Hugging Face Hub,
    local disk, or custom CSV/JSON files.

    Args:
        data_args: DataTrainingArguments object with data configuration.
        model_args: ModelArguments object with model configuration.
        training_args: TrainingArguments object with training configuration.

    Returns:
        raw_datasets: Loaded datasets.
    """
    # Conditionally load datasets based on the specified task or custom files
    if data_args.task_name is not None:
        # Option for loading datasets from local disk
        if data_args.use_local:
            dataset_path = os.path.join(data_args.data_dir, "glue", data_args.task_name)
            raw_datasets = load_from_disk(dataset_path)
            logger.info(f"Loaded dataset from local disk: {dataset_path}")
        else:
            # Loading from the Hugging Face datasets hub
            raw_datasets = load_dataset(
                "glue", data_args.task_name, cache_dir=model_args.cache_dir
            )
            logger.info(f"Loaded GLUE task: {data_args.task_name}")
    elif data_args.dataset_name is not None:
        # Loading a specific dataset from the hub
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        logger.info(f"Loaded dataset from the hub: {data_args.dataset_name}")
    else:
        # Loading datasets from custom local files (CSV/JSON)
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }
        if training_args.do_predict:
            assert (
                data_args.test_file is not None
            ), "Test file must be specified for prediction."
            data_files["test"] = data_args.test_file
            logger.info(
                "Loaded custom data files for training, validation, and testing."
            )
        raw_datasets = load_dataset(
            data_args.train_file.split(".")[-1],
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    return raw_datasets


def determine_task_type_and_labels(raw_datasets, data_args):
    """
    Determines if the task is regression or classification and configures labels.

    Args:
        raw_datasets: The loaded datasets.
        data_args: DataTrainingArguments object.

    Returns:
        A tuple containing is_regression (bool), num_labels (int), and label_list (sorted list or None).
    """
    # Check if task is regression based on task name or label data type
    is_regression = (
        data_args.task_name == "stsb"
        if data_args.task_name is not None
        else raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    )
    if not is_regression:
        # Classification task: Extract and sort labels
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Sorting for determinism
        num_labels = len(label_list)
    else:
        # Regression task: No label list needed
        label_list = None
        num_labels = 1

    logger.info(
        f"Task type determined: {'regression' if is_regression else 'classification'} with {num_labels} labels."
    )
    return is_regression, num_labels, label_list


def configure_tokenization(tokenizer, data_args):
    """
    Configures tokenization settings based on tokenizer limits and data arguments.

    Args:
        tokenizer: The tokenizer object.
        data_args: DataTrainingArguments object.

    Returns:
        A tuple containing max_seq_length (int) and padding strategy (bool or 'max_length').
    """
    # Determine the maximum sequence length
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Determine padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    logger.info(
        f"Tokenization configured with max_seq_length={max_seq_length} and padding={'max_length' if padding else 'dynamic'}."
    )
    return max_seq_length, padding


def identify_text_fields(raw_datasets, data_args):
    """
    Identifies the keys for text fields to be used in tokenization, based on the dataset or task.

    Args:
        raw_datasets: The loaded datasets.
        data_args: DataTrainingArguments object containing configuration for the data processing.

    Returns:
        sentence1_key: The key in the dataset for the first piece of text to be tokenized.
        sentence2_key: The key for the second piece of text, if applicable.
    """
    if data_args.task_name is not None:
        # Task-specific key identification, predefined for known tasks
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Custom dataset key identification
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key = non_label_column_names[0]
            sentence2_key = None

    return sentence1_key, sentence2_key


def prepare_label_id_mapping(model, data_args, is_regression, label_list, num_labels):
    """
    Prepares a label-to-ID mapping and updates the model's configuration accordingly.

    Args:
        model: The model object for which to prepare the label mapping.
        data_args: DataTrainingArguments, containing the task name and other data-related settings.
        is_regression: Boolean indicating if the current task is regression.
        label_list: A list of labels derived from the dataset.
        num_labels: The number of unique labels.

    Returns:
        A label-to-ID mapping dictionary.
    """
    label_to_id = None

    if not is_regression:
        # Check if model's label2id matches the dataset's labels
        if (
            data_args.task_name
            and model.config.label2id
            and model.config.label2id
            != PretrainedConfig(num_labels=num_labels).label2id
        ):
            # Convert model's label2id keys to lowercase for comparison
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if sorted(label_name_to_id.keys()) == sorted(
                [label.lower() for label in label_list]
            ):
                label_to_id = {label_list[i]: i for i in range(num_labels)}
            else:
                logger.warning(
                    "Model's labels do not match the dataset. Ignoring the model labels."
                )
        else:
            # Generate label_to_id from dataset labels if no task name is provided or if regression
            label_to_id = {label: i for i, label in enumerate(label_list)}

        # Update the model's label2id and id2label configurations
        if label_to_id:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in label_to_id.items()}
    else:
        # For regression tasks, ensure label_to_id remains None
        label_to_id = None

    return label_to_id


def preprocess_function(
    examples,
    tokenizer,
    max_seq_length,
    padding,
    label_to_id=None,
    sentence1_key="sentence1",
    sentence2_key=None,
):
    """
    Tokenizes text inputs and maps labels to IDs if necessary.

    Args:
        examples: Input examples to process.
        tokenizer: The tokenizer to use for encoding texts.
        max_seq_length: Maximum sequence length for tokenization.
        padding: Padding strategy for tokenization.
        label_to_id: Optional mapping from label names to label IDs.
        sentence1_key: Column name in the dataset for the first sentence.
        sentence2_key: Optional column name for the second sentence, for pair tasks.

    Returns:
        Dictionary of processed features.
    """
    # Tokenize the text
    tokenize_args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(
        *tokenize_args, padding=padding, max_length=max_seq_length, truncation=True
    )

    # Map labels to IDs if provided
    if label_to_id is not None and "label" in examples:
        result["label"] = [
            (label_to_id[l] if l != -1 else -1) for l in examples["label"]
        ]

    return result
