# LLama3 Finetuning

The purpose of this repository is to understand the finetuning process of Llama-3 Large Language Model

## How to read this repository?

TODO

## Install PyPi Libraries

`pip install -r requirements.txt`

## Dataset

### About Dataset

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

### Download Dataset

`from datasets import load_dataset`

`dataset = load_dataset("yahma/alpaca-cleaned")`

Refer `dataset.py` for converting the training dataset into the Alpaca format for fine-tuning.

### Example

```
"instruction": "Calculate the area of the following shape in square centimeters.",
"input": "rectangle of size 4 cm x 5 cm",
"output": ""
```

```
"instruction": "Create a graphic or logo that visually represents the word \"courage\".",
"input": "",
"output": "<No Output>"
```

## Architecture

![Llama3_Finetuning_Architecture](https://github.com/bala1802/LLama3_Finetuning/assets/22103095/9303a4b5-3e13-431d-a7ec-ac2acebd3652)

