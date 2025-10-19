---
title: 使用unsloth加速训练deepseek-r1-distill-qwen-1.5b
pubDatetime: 2025-03-20T00:00:00Z
slug: unsloth-fine-tune-deepseekqwen1-5b
featured: false
draft: false
tags:
  - unsloth
  - deepseek
description:
  通过使用unsloth 库来实现在线微调deepseek模型
---

 - **wandb** : 可视化训练过程

```python
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel
import wandb
import torch
```

配置wandb密钥。transformers会自动使用wandb进行实验跟踪。


```python
secret_value_0 = "f94dda6c6bae501b9220c25ecc5f530e90ac475a"

wandb.login(key=secret_value_0)
run = wandb.init(project="deepseek-fine-tune", job_type="training")
```

## 1. 加载数据集

通过datasets库中的load_dataset函数加载数据集，指定名称之后，会自动从huggingface数据集中下载对应的数据集。

你也可以使用自己的数据集：
```
raw_datasets = load_dataset("json", data_files="./datasets/ppt_dataset_1.json")

```
加载后的数据集对象是一个DatasetDict对象。


```python
# 加载数据集
# raw_datasets = load_dataset("json", data_files="./datasets/ppt_dataset_1.json")

raw_datasets = load_dataset("LooksJuicy/ruozhiba")

# 选取小型数据集
small_datasets = raw_datasets["train"].select(range(300))
```

## 2. 加载训练模型

通过指定给定的模型名称会自动加载预训练模型。
这里由于使用的Unsloth库，加载的是量化后的预训练模型。

- max_seq_length = 2048– 控制上下文长度。虽然 Llama-3 支持 8192，但我们建议使用 2048 进行测试。Unsloth 可实现 4 倍更长的上下文微调。

- dtype = None– 默认为无；使用torch.float16或torch.bfloat16适用于较新的 GPU。

`load_in_4bit = True` – 启用 4 位量化，在 16GB GPU 上进行微调时，内存使用量减少 4 倍。在较大的 GPU（例如 H100）上禁用此功能可略微提高准确度（1-2%）。对于`完全微调`- 设置`full_finetuning = True` 和 8 位微调- 设置`load_in_8bit = True` 。


```python
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
max_seq_length = 512

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.float16,
)
```

lora微调的配置。
- r（分解等级）：控制微调过程。
  - 建议： 8， 16， 32， 64， 128
  - 更高：在困难任务上准确度更高，但会增加记忆力和过度拟合的风险。
  - 较低：速度更快，节省内存，但可能会降低准确性。

- lora_alpha（缩放因子）：决定学习强度
  - 建议：等于或加倍等级（r）。
  - 更高：学习更多，但可能过度拟合。
  - 较低：学习速度较慢，更具普遍性。

- lora_dropout（默认值：0）：正则化的 Dropout 概率。
  - 更高：更多正规化，训练速度更慢。
  - 较低（0）：训练速度更快，对过度拟合的影响最小。

  
- target_modules：需要微调的模块（默认包括"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"）。
 - 建议对所有模块进行微调以获得最佳效果。

- use_gradient_checkpointing：减少长上下文的内存使用量。

### 目标模块说明 
这些组件将输入转换为注意力机制：
**q_proj、k_proj、v_proj**：处理查询、键和值。
**o_proj**：将注意力结果整合到模型中。
**gate_proj**：管理门控层中的流程。
**up_proj，down_proj**：调整维度以提高效率。


```python
# 定义 LoRA 配置, 将适配器附加到量化模型
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # LoRA 秩
    lora_alpha=16,  # LoRA alpha
    target_modules=["q_proj", "v_proj", "gate_proj", "o_proj"],
    lora_dropout=0.1,  # LoRA dropout
    bias="none",  # Bias type
    use_gradient_checkpointing=True,  # 梯度检查点（Unsloth优化版）
    max_seq_length=max_seq_length,
)
```

## 3. 分词处理

1. 构建对话模板
2. 预处理数据
3. 对齐

```
### Instruction
{}

### Input
{}

### Response
{}

```


```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 pad_token


def tokenize_function(examples):
    # 初始化容器
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for ins, o in zip(examples["instruction"], examples["output"]):
        # 构建完整对话结构
        context = f"[指令/问题]{ins} \n[AI] "
        response = f"{o}{tokenizer.pad_token}"
        full_text = context + response

        # 统一分词（禁用自动添加特殊标记）
        encoding = tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            padding="max_length" if tokenizer.pad_token else False,
            # add_special_tokens=False,  # 关键！防止与模板中的特殊标记冲突
        )

        # 确定上下文长度（需要单独分词）
        context_enc = tokenizer(context, add_special_tokens=False)
        context_len = len(context_enc["input_ids"])

        # 创建标签（掩码输入部分）
        labels = [
            -100 if i < context_len else token_id
            for i, token_id in enumerate(encoding["input_ids"])
        ]

        # 处理填充符的损失计算
        labels = [
            (l if l != tokenizer.pad_token_id else -100)
            for l in labels
        ]

        # 存储结果
        model_inputs["input_ids"].append(encoding["input_ids"])
        model_inputs["attention_mask"].append(encoding["attention_mask"])
        model_inputs["labels"].append(labels)

    return model_inputs
```

## 4. 应用分词并划分训练集和数据集


```python
# 应用分词函数
tokenized_datasets = small_datasets.map(tokenize_function, batched=True)

# 划分训练集与验证集
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
```

## 5. 配置训练超参数


```python
# 训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 模型保存路径
    eval_strategy="steps",  # 每轮验证
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=2,  # 训练批量大小
    per_device_eval_batch_size=8,  # 验证批量大小
    gradient_accumulation_steps=2,
    num_train_epochs=5,  # 训练轮数
    weight_decay=0.01,  # 权重衰减
    fp16=True,  # 混合精度训练（如果 GPU 支持）
    logging_steps=20,
    eval_steps=50,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    lr_scheduler_type="cosine",  # 学习率调度器
    # optim="adamw_torch_fused",
)

```

## 6. 定义训练器


```python
trainer = Trainer(
    model=model,  # 模型
    
wandb.finish()
```

## 8. 保存训练模型


```python
# # 保存模型
from peft import PeftModel
from transformers import  AutoModelForCausalLM, AutoTokenizer

model.save_pretrained("./lora_model/cprogram/12/")

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
base_tokenizer = AutoTokenizer.from_pretrained(model_name)

tuned_model = PeftModel.from_pretrained(base_model, "./lora_model/cprogram/12/")

merge_model = tuned_model.merge_and_unload()
merge_model.save_pretrained("./tuned_merge_model/cprogram/12/")

merge_model = AutoModelForCausalLM.from_pretrained(
    "./tuned_merge_model/cprogram/12/", device_map="cuda"
)
```
