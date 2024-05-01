from unsloth import FastLanguageModel

import config

def get_peft_model(model):
    model = FastLanguageModel.get_peft_model(model, 
                                             r = config.R,
                                             target_modules = config.TARGET_MODULES,
                                             lora_alpha = config.LORA_ALPHA,
                                             lora_dropout = config.LORA_DROPOUT,
                                             bias = config.BIAS,
                                             use_gradient_checkpointing = config.USE_GRADIENT_CHECKPOINTING,
                                             random_state = config.RANDOM_STATE,
                                             use_rslora = config.USE_RSLORA,
                                             loftq_config = config.LOFTQ_CONFIG)
    return model

if __name__ == "__main__":
    print("Hello from unsloth")