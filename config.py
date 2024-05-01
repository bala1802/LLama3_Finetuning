#Parameter Efficient Finetuning
R = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj",]
LORA_ALPHA = 16,
LORA_DROPOUT = 0,
BIAS = "none",
USE_GRADIENT_CHECKPOINTING = "unsloth",
RANDOM_STATE = 3407,
USE_RSLORA = False,
LOFTQ_CONFIG = None

#Alpaca Dataset
DATASET = "yahma/alpaca-cleaned"

#Model
MAX_SEQ_LENGTH = 2048 # Choose any! Llama 3 is up to 8k
d_TYPE = None
LOAD_IN_4BIT = True # Use 4bit quantization to reduce memory usage. Can be False.

#Model Training
DATASET_TEXT_FIELD = "text"
DATASET_NUM_PROC = 2
PACKING = False
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
WARM_STEPS = 5
MAX_STEPS = None
NUM_TRAINING_EPOCHS=4
LEARNING_RATE = 2e-4
LOGGING_STEPS = 1,
OPTIM = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCEHDULER_TYPE = "linear"
SEED = 3407
OUTPUT_DIR = "outputs"

#Model Saving
FINETUNED_MODEL_PATH = "lora_model"
MODEL_SAVE_METHOD = "merged_16bit"

#Quantization
F16_QUANTIZATION = "f16"
Q4_K_M = "q4_k_m"