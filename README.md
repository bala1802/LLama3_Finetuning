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

### Alpaca Example

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

## Model Finetuning

![Llama3_Finetuning_Architecture](https://github.com/bala1802/LLama3_Finetuning/assets/22103095/9303a4b5-3e13-431d-a7ec-ac2acebd3652)

🔍 Quantize the Llama3 8B large Language Model to 16-bit model. This technique reduces the memory and computation requirements of the Neural Network layer by representing the weights and activations in only 4 bits. Refer quantization_utils.py

🧠 Identify the Layers that require weight updates and freeze the rest during fine-tuning. Managing the layers this way will allow the crucial layers to adapt to the new domain-specific data, while preserving the rest of the parameters of the pre-trained model.

The layer names can be identified by printing the Architecture of the model

## Inferencing

![Llama3_Inferencing](https://github.com/bala1802/LLama3_Finetuning/assets/22103095/e19a2b2d-40fa-4a48-bad3-b06ac983e5db)

🔍 Extract the Adapter from the Fine-tuned Quantized model. This Adapter encapsulates the refined parameters tailored to the domain-sepcific data.

🧩 Integrate the Adapter with the original Pre-Trained Llama-3 Large Language Model. This fusion enables the Language Model with the domain knowledge acquired during the fine-tuning process.

💬 User provides the prompt to the Langauage Model for interaction

🚀 The Language Model generates the response for the provided Prompt.


