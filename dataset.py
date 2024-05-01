from datasets import load_dataset

import config
from prompts import ALPACA_PROMPT

def format_prompt(examples, eos_token):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, input, output) + eos_token # without this token generation goes on forever!
        texts.append(text)
    return { "text" : texts, }

def extract_dataset():
    dataset = load_dataset(config.DATASET, split = "train")
    dataset = dataset.map(format_prompt, batched = True,)
    return dataset
