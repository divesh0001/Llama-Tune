# 🚀 Fine-Tuning LLaMA-2 with QLoRA

Fine-tune **LLaMA-2-7B** efficiently using **QLoRA** (Quantized Low-Rank Adaptation) for instruction tuning. This method enables fine-tuning large models on consumer GPUs using **4-bit quantization**.

## 📌 Features
- Uses **NousResearch/Llama-2-7b-chat-hf** as the base model.
- Implements **QLoRA** for memory-efficient fine-tuning.
- Trains on **guanaco-llama2-1k**, a small instruction dataset.
- Saves and loads fine-tuned models.

## ⚙️ Installation
Run the following commands to set up the environment:
```bash
pip install accelerate peft bitsandbytes transformers trl datasets torch
```

## 📂 Dataset
We use **mlabonne/guanaco-llama2-1k**, an instruction-tuning dataset. It helps the model improve chatbot-like responses.

## 🏗 Model Setup
Load the **LLaMA-2-7B** model with **4-bit quantization**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "NousResearch/Llama-2-7b-chat-hf"
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## 🏋️ Fine-Tuning with QLoRA
### **LoRA Configuration**
```python
from peft import LoraConfig
peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")
```

### **Training Setup**
```python
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args={
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 0.3,
        "optim": "paged_adamw_32bit"
    },
    packing=True,
)
trainer.train()
```

## 📦 Saving & Loading the Model
Save the fine-tuned model:
```python
trainer.model.save_pretrained("Llama-2-7b-chat-finetune")
```
Load it for inference:
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="Llama-2-7b-chat-finetune", tokenizer=tokenizer)
pipe("Tell me a joke about AI.")
```

## 🚀 Results & Performance
- **Fine-tuned model performs better on instruction-based tasks.**
- **QLoRA reduces memory consumption, allowing fine-tuning on consumer GPUs.**

## 📜 License
Apache-2.0 License

---
### 🌟 Contribute
Pull requests are welcome! If you find issues, report them.

Happy fine-tuning! 🚀
