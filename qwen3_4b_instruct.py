"""
Script version of Qwen-3 fine-tuning model for Proposal description generator 
Written 22 October 2025
"""

from unsloth import FastModel
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only


#Intialize the model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

#Data Preparation
raw_data = pd.read_csv('./data/camden_output_data.csv', index_col=None, header=1, encoding='latin1')
raw_data.dropna(inplace = True)

# we drop first column and change column names to prepare the data in the prompt-completion style
df = raw_data[['ON-LINE APPLICATION', 'DECIDED-FINAL']].rename(
    columns={'ON-LINE APPLICATION': 'prompt', 'DECIDED-FINAL': 'completion'}
)
# immediately load the raw data into a Dataset object in the prompt-completion style
ds = Dataset.from_pandas(df)

tokenizer = get_chat_template(tokenizer, chat_template = "qwen3-instruct",)

def format_with_chat_template(example):
    # Create the message structure that the chat template expects
    dct = [
          {"role": "system", "content": "You are a trained planning officer at a local council in the UK. Change the following piece of text, which is a description of a planning application, to give it the best chance of success."},
          {"role": "user", "content": example["prompt"]},
          {"role": "assistant", "content": example["completion"]},
    ]
    # Apply the chat template, returning a single formatted string
    # `tokenize=False` means we get a string, not token IDs. SFTTrainer will tokenize it.
    templated = tokenizer.apply_chat_template(
        conversation=dct,
        tokenize = False,
        add_generation_prompt = False,
    )
    return {"text": templated}

# Use .map() to apply the function to the entire dataset
formatted_dataset = ds.map(format_with_chat_template)

result = formatted_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_ds, test_ds = result["train"], result["test"]

#Train the model 
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = test_ds, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 10, # Set this for 1 full training run.
       # max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        eval_strategy="epoch",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n", #<|im_start|>system\n
    response_part = "<|im_start|>assistant\n",
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")