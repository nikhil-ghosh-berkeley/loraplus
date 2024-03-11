import os
import fire
from datasets import load_dataset
from typing import Iterable, Union

glue_tasks = {"cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"}


# example usage: python download_glue.py --task_names mnli,qqp --data_dir /scratch/users/nikhil_ghosh
def download_glue_tasks(task_names: Union[str, Iterable[str]], data_dir: str = "data"):
    print(f"downloading tasks: {str(task_names)}")

    if isinstance(task_names, str):
        task_names = [task_names]

    assert set(task_names).issubset(glue_tasks)

    for task_name in task_names:
        save_dir = os.path.join(data_dir, "glue", task_name)
        os.makedirs(save_dir, exist_ok=True)
        raw_datasets = load_dataset(
            "glue",
            task_name,
        )
        raw_datasets.save_to_disk(save_dir)


if __name__ == "__main__":
    fire.Fire(download_glue_tasks)
