import logging
import os

from datasets import load_dataset, load_from_disk
from logging_utils import setup_logging

from transformers import PretrainedConfig

setup_logging()
logger = logging.getLogger(__name__)

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


"""
# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
# or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
# sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
# label if at least two columns are provided.
#
# If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
# single column. You can easily tweak this behavior (see below)
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if data_args.task_name is not None:
    # Downloading and loading a dataset from the hub.
    if data_args.use_local:
        raw_datasets = load_from_disk(
            os.path.join(data_args.data_dir, "glue", data_args.task_name)
        )
        datasets.set_caching_enabled(False)
    else:
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=True if model_args.token else None,
        )
elif data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=True if model_args.token else None,
    )
else:
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError(
                "Need either a GLUE task or a test file for `do_predict`."
            )

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.token else None,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.token else None,
        )
# See more about loading any type of standard or custom dataset at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Labels
if data_args.task_name is not None:
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
else:
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = raw_datasets["train"].features["label"].dtype in [
        "float32",
        "float64",
    ]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

# Preprocessing the raw_datasets
if data_args.task_name is not None:
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
else:
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [
        name for name in raw_datasets["train"].column_names if name != "label"
    ]
    if (
        "sentence1" in non_label_column_names
        and "sentence2" in non_label_column_names
    ):
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

# Padding strategy
if data_args.pad_to_max_length:
    padding = "max_length"
else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False

# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and data_args.task_name is not None
    and not is_regression
):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        label_to_id = {
            i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
        }
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
            "\nIgnoring the model labels as a result.",
        )
elif data_args.task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
elif data_args.task_name is not None and not is_regression:
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(
        *args, padding=padding, max_length=max_seq_length, truncation=True
    )

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [
            (label_to_id[l] if l != -1 else -1) for l in examples["label"]
        ]
    return result
"""
