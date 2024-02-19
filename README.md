# loraplus

This repository contains the code for LoRA+, introduced in "LoRA+: Efficient Low Rank Adaptation of Large Models".

## Requirements

First, install the requirements using:
```
pip install -r requirements
```

## Running GLUE Experiments

Download the glue tasks using e.g. :
```
python download_glue.py --task_names mnli,qqp --datadir
```

Check `scripts' folder for examples with gpt2/roberta-base.


## Running LLama Experiments
TODO

## Using LoRA+ to finetune others models

Integrating LoRA+ in your finetuning project is straightforward. Just add src/loraplustrainer.py in your project and pass the necessary args.


## Citation

This code is part of the LoRA+ project. For citations, please use 
TODO
