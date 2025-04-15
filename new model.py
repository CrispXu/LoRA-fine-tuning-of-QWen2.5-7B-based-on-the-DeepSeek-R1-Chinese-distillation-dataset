import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载微调后的模型和tokenizer
model_path = "output"  # 训练代码保存的路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).eval()

def qwen_chat(prompt, max_new_tokens=512, temperature=0.9):
    """
    与微调后的模型进行对话
    :param prompt: 用户输入的问题/指令
    :param max_new_tokens: 生成的最大token数
    :param temperature: 温度参数(控制生成随机性)
    :return: 模型生成的回答
    """
    # 构建对话模板
    messages = [
        {"role": "system", "content": "你是一个知识渊博的助手，请根据问题给出详细准确的回答"},
        {"role": "user", "content": prompt}
    ]
    
    # 应用ChatML模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# 示例使用
if __name__ == "__main__":
    test_questions = [
        "如何区分感冒和流感？",
        "请解释量子纠缠的基本原理",
        "用Python写一个快速排序算法"
    ]
    
    for question in test_questions:
        print(f"用户问题：{question}")
        answer = qwen_chat(question)
        print(f"模型回答：{answer}\n{'-'*50}")