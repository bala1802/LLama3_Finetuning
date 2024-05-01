import torch
from trl import SFTTrainer
from transformers import TrainingArguments

import config

def train(model, tokenizer, dataset):
    trainer = SFTTrainer(model = model,
                         tokenizer = tokenizer,
                         train_dataset = dataset,
                         dataset_text_field = config.DATASET_TEXT_FIELD,
                         max_seq_length = config.MAX_SEQ_LENGTH,
                         dataset_num_proc = config.DATASET_NUM_PROC,
                         packing = config.PACKING,
                         args = get_training_arguments()
                         )
    return trainer

def get_training_arguments():
    training_arguments = TrainingArguments(per_device_train_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE,
                             gradient_accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS, 
                             warmup_steps = config.WARM_STEPS,
                             max_steps = config.MAX_STEPS,
                             num_train_epochs=config.NUM_TRAINING_EPOCHS, 
                             learning_rate = config.LEARNING_RATE, 
                             fp16 = not torch.cuda.is_bf16_supported(),
                             bf16 = torch.cuda.is_bf16_supported(),
                             logging_steps = config.LOGGING_STEPS,
                             optim = config.OPTIM,
                             weight_decay = config.WEIGHT_DECAY,
                             lr_scheduler_type = config.LR_SCEHDULER_TYPE,
                             seed = config.SEED,
                             output_dir = config.OUTPUT_DIR)
    return training_arguments

if __name__ == "__main__":
    print("Hello from train")    