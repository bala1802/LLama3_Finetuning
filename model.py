from unsloth import FastLanguageModel
import config

def initialize_model_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
                                    model_name = config.LLAMA_3_8B
                                    max_seq_length = config.MAX_SEQ_LENGTH,
                                    dtype = config.d_TYPE,
                                    load_in_4bit = config.LOAD_IN_4BIT)
    return model, tokenizer

if __name__ == "__main__":
    print("Hello from model")