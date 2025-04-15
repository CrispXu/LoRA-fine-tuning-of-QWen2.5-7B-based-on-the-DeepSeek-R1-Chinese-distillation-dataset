# Qwen2.5-7B-train.py
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab

# 设备名称
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 训练集处理函数
def trainset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取原JSONL文件
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行原始数据（每一行均是一个JSON格式）
            data = json.loads(line)
            input = data["input"]
            output = data["content"]
            message = {
                "input": f"文本:{input}",
                "output": output,
            }
            messages.append(message)

    # 保存处理后的JSONL文件，每行也是一个JSON格式
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 测试集处理函数
def testset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取原JSONL文件
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行原始数据（每一行均是一个JSON格式）
            data = json.loads(line)
            input = data["instruction_zh"]
            output = data["output_zh"]
            message = {
                "input": f"文本:{input}",
                "output": output,
            }
            messages.append(message)

    # 保存处理后的JSONL文件，每行也是一个JSON格式
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 定义处理单个样本的函数，输入是数据集的一个样本字典
def process_func(example):
    # 设置最大序列长度，超过此长度将截断（问答任务通常需要更长上下文）
    MAX_LENGTH = 512  
    
    # 初始化存储容器（PyTorch模型需要的三种关键输入）
    input_ids, attention_mask, labels = [], [], []  
    
    # 构建问答对话模板（核心结构）
    instruction = tokenizer(
        # 使用ChatML格式的对话标记
        f"<|im_start|>system\n你是一个知识渊博的助手，请根据问题给出详细准确的回答<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"  # 插入用户问题
        f"<|im_start|>assistant\n",  # 注意这里没有关闭标签，留给模型生成回答
        add_special_tokens=False  # 重要！避免自动添加[CLS]/[SEP]等标记
    )
    
    # 处理目标回答（即希望模型生成的部分）
    response = tokenizer(
        f"{example['output']}<|im_end|>",  # 显式添加对话结束标记
        add_special_tokens=False  # 保持标记一致性
    )
    
    # 拼接完整输入序列：指令部分 + 回答部分
    input_ids = instruction["input_ids"] + response["input_ids"]
    
    # 创建注意力掩码：有效内容为1（后续padding部分会补0）
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    
    # 创建标签：-100表示需要屏蔽的位置（PyTorch的CrossEntropyLoss会自动忽略）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    
    # 动态长度处理（保证所有样本统一长度）
    if len(input_ids) < MAX_LENGTH:
        # 计算需要填充的长度
        pad_len = MAX_LENGTH - len(input_ids)
        
        # 填充input_ids（使用pad_token_id）
        input_ids += [tokenizer.pad_token_id] * pad_len
        
        # 填充attention_mask（0表示padding部分无需关注）
        attention_mask += [0] * pad_len
        
        # 填充labels（保持屏蔽状态）
        labels += [-100] * pad_len
    else:
        # 截断处理：保留前MAX_LENGTH个token
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # 返回格式化后的字典（与HuggingFace Trainer兼容）
    return {
        "input_ids": input_ids,          # 模型输入的token ID序列
        "attention_mask": attention_mask,# 指示哪些位置是有效内容
        "labels": labels                 # 训练时计算loss的参考目标
    }

model_dir = 'Qwen2.5-7B'
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, torch_dtype=torch.bfloat16)
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法


train_dataset_path = 'data/distill_r1_110k.jsonl'
test_dataset_path = 'test/Alpaca_data_gpt4_zh.jsonl'
train_jsonl_new_path = 'data/train.jsonl'
test_jsonl_new_path = 'test/test.jsonl'
if not os.path.exists(train_jsonl_new_path):
    trainset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    testset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到微调数据集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 创建LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

# 将LoRA应用于模型
model = get_peft_model(model, config)

# 创建微调参数
args = TrainingArguments(
    output_dir='logs',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
# SwanLab微调过程回调数据
swanlab_callback = SwanLabCallback(project="Qwen2.5-FineTuning", experiment_name="Qwen2.5-7B")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# 开始微调
trainer.train()

merged_model = model.merge_and_unload()
merged_model.save_pretrained("output")
tokenizer.save_pretrained("output")

# 模型结果结果评估
def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 模型评估：获取测试集的前10条测试数据
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]

test_text_list = []
for index, row in test_df.iterrows():
    instruction = '你是一个知识渊博的助手，请根据问题给出详细准确的回答'
    input_value = row['input']

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})

    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()

