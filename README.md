# LoRA+

This repository contains the code for LoRA+, introduced in "LoRA+: Efficient Low Rank Adaptation of Large Models".

## Usage
LoRA+ introduces one new required hyperparameter to your optimizer (and another optional hyperparameter).
### LoRA+ arguments
* `loraplus_lr_ratio`: the ratio of learning rates $\eta_B / \eta_A$ where $\eta_A$ is the optimizer learning rate. Set this to a value greater than 1 for best performance. See the paper for more information.
* `loraplus_lr_embedding`: (optional) if LoRA modules are added to embedding layers your can specify an different learning rate for them. Default value `1e-6`.

### With Huggingface Trainer
To integrate LoRA+ into a finetuning project using huggingface `Trainer` is straightforward. Just replace the `Trainer` in your project with `LoraPlusTrainer` in `loraplus.py` and pass in the LoRA+ arguments above to its `TrainingArguments`.

### No Trainer
For training with LoRA+ without `Trainer` you can just use an optimizer created with the `create_loraplus_optimizer` function in `loraplus.py`. This function wraps an optimizer class and sets the learning rates of your model parameters appropriately for the optimizer. 

```python
import torch

# LoRA model
model = ...

optimizer_cls = torch.optim.AdamW
optimizer_kwargs = {'lr': 5e-5, 'eps': 1e-6, 'betas': (0.9, 0.999), 'weight_decay': 0.0}
loraplus_lr_ratio = 20.0
optimizer = _create_optimizer(model, optimizer_cls, optimizer_kwargs, loraplus_ratio):
```

## Examples
In the `glue/` folder we have code for finetuning models on GLUE using LoRA+. 

### Requirements
To run the code, first install the requirements using:
```
pip install -r requirements.txt
```

### Running GLUE Experiments

Download the glue tasks using e.g. :
```
python download_glue.py --task_names mnli,qqp --datadir data/
```

Check `scripts/` folder for finetuning examples with gpt2, roberta-base, and llama-7b.

## Citation

This code is part of the LoRA+ project. For citations, please use 
```
@article{hayou2024lora,
      title={LoRA+: Efficient Low Rank Adaptation of Large Models}, 
      author={Soufiane Hayou and Nikhil Ghosh and Bin Yu},
      year={2024},
      journal={arXiv 2402.12354}
}
```
