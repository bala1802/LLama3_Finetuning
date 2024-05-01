from unsloth import FastLanguageModel
from transformers import TextStreamer

import prompts

def generate(model, tokenizer, instruction, input):
    inputs = tokenizer([prompts.ALPACA_PROMPT.format(instruction, input,"")], 
                        return_tensors = "pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

if __name__ == "__main__":
    print("Hello from inference")
    