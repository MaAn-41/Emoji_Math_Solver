# Emoji Math Fine-Tuning with LoRA

## Overview
This project fine-tunes the `deepseek-ai/deepseek-math-7b-instruct` model to solve emoji-based math problems using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. The dataset consists of emoji-based math equations and their step-by-step solutions, formatted for instruction tuning.

## Features
- **Hugging Face Datasets Integration:** Uses `datasets` to structure and preprocess the dataset.
- **LoRA Fine-Tuning:** Utilizes `peft` to fine-tune the model efficiently.
- **Tokenization & Data Collation:** Prepares data with instruction-response formatting.
- **Training & Validation Splits:** Splits dataset into training (90%) and validation (10%) sets.
- **Optimized Model Loading:** Uses 8-bit quantization for efficient fine-tuning.

## Installation
Ensure you have the necessary dependencies installed:
```bash
!pip install datasets
!pip install -U bitsandbytes
!pip install peft
```

## Dataset
The dataset consists of emoji-based math problems structured as:
```json
{
  "problem": "🍎 + 🍎 + 🍎 = 12",
  "solution": "Since 🍎 + 🍎 + 🍎 = 12, each 🍎 must equal 4. So 🍎 = 4."
}
```
After preprocessing, it is formatted as:
```json
{
  "input": "Solve this emoji math problem: 🍎 + 🍎 + 🍎 = 12",
  "output": "Since 🍎 + 🍎 + 🍎 = 12, each 🍎 must equal 4. So 🍎 = 4."
}
```

## Model Fine-Tuning Process
1. **Load and tokenize data:**
   - Applies an instruction-response format for improved fine-tuning.
2. **Prepare dataset for training:**
   - Converts processed data into PyTorch datasets.
3. **Apply LoRA:**
   - Uses LoRA configuration to reduce memory usage while fine-tuning.
4. **Train model using Hugging Face Trainer.**

## LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

## Running the Training
Once everything is set up, start the training process:
```python
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    args=TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
    ),
)
trainer.train()
```

## Usage
After training, you can use the fine-tuned model for inference:
```python
input_text = "Solve this emoji math problem: 🍎 + 🍎 + 🍎 = 12"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Acknowledgments
- **Hugging Face Transformers** for providing pre-trained models and training utilities.
- **DeepSeek-AI** for the base model used in this fine-tuning process.
- **LoRA (Low-Rank Adaptation)** for enabling efficient fine-tuning on large models.

## License
This project is released under the MIT License.

