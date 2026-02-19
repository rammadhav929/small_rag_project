# pip install torch transformers datasets accelerate


import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)
from datasets import load_dataset

# ---------------------------
# 1. Load Dataset
# ---------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# ---------------------------
# 2. Create Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ---------------------------
# 3. Create GPT Model Config
# ---------------------------
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_ctx=128,
    n_embd=256,      # Embedding size
    n_layer=4,       # Number of transformer layers
    n_head=4         # Attention heads
)

model = GPT2LMHeadModel(config)

# ---------------------------
# 4. Data Collator
# ---------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM
)

# ---------------------------
# 5. Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./llm_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=1,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# ---------------------------
# 6. Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator
)

trainer.train()

# ---------------------------
# 7. Save Model
# ---------------------------
trainer.save_model("./custom_llm")
tokenizer.save_pretrained("./custom_llm")
