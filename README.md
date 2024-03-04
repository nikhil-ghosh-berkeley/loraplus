# LoRA+

This repository contains the code for LoRA+, introduced in [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354).

## Usage
LoRA+ introduces one new required hyperparameter to your optimizer (and another optional hyperparameter). Setting this hyperparameter appropriately can improve finetuning performance, especially on more challenging downstream tasks.
### LoRA+ arguments
* `loraplus_lr_ratio`: the ratio of learning rates $\eta_B / \eta_A$ where $\eta_A$ is passed in as the optimizer learning rate (e.g., `learning_rate` or `lr`). See the note below for some advice on how to set this.
* `loraplus_lr_embedding`: (optional) if LoRA modules are added to embedding layers your can specify a different learning rate for them. Default value `1e-6`.

**NOTE**: 
`loraplus_lr_ratio` should be $\geq 1$, but the optimal choice of `loraplus_lr_ratio` is 
1. model and task dependent.
2. needs to be set in tandem with the optimizer learning rate (i.e., $\eta_A$).
   
As a rule of thumb, `loraplus_lr_ratio` should be larger when the task is more difficult and the model needs to update its features to learn well. In this case, it helps to make the learning rate $\eta_A$ slightly smaller (e.g., by a factor of 2) than typical vanilla LoRA learning rates. Please see the [paper](https://arxiv.org/abs/2402.12354) for examples.

### Code
The code for using LoRA+ can be found in `lora_plus.py`.

**With Huggingface Trainer**

To integrate LoRA+ into a finetuning project using huggingface `Trainer` is straightforward. Just replace the `Trainer` in your project with `LoraPlusTrainer` and pass in the training arguments (including LoRA+ arguments) using `LoraPlusTrainingArguments`. See the `image_classification.ipynb` notebook for an example.

**No Trainer**

For training with LoRA+ without `Trainer` you can just use an optimizer created with the `create_loraplus_optimizer` function. This function wraps an optimizer class and sets the learning rates of your model parameters appropriately for the optimizer. 

```python
import torch

# LoRA model
model = ...

optimizer_cls = torch.optim.AdamW
optimizer_kwargs = {'lr': 5e-5, 'eps': 1e-6, 'betas': (0.9, 0.999), 'weight_decay': 0.0}
loraplus_lr_ratio = 20.0
optimizer = create_loraplus_optimizer(model, optimizer_cls, optimizer_kwargs, loraplus_ratio):
```

## Examples
In the `glue/` folder we have code for finetuning models on GLUE using LoRA+ which can be used to reproduce results in the paper. We also include a notebook `image_classification.ipynb` for demonstrating code usage.

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
@article{hayou2024loraplus,
      title={LoRA+: Efficient Low Rank Adaptation of Large Models}, 
      author={Soufiane Hayou and Nikhil Ghosh and Bin Yu},
      year={2024},
      journal={arXiv 2402.12354}
}
```

## Acknowledgements
We thank Amazon Web Services (AWS) for providing the compute for this project through cloud credits under an Amazon Research Award.
