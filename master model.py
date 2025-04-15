import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# 设备配置
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# 模型与分词器加载
# ---------------------------
model_path = "Qwen2.5-7B"  # 🤗 Hub官方路径
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    pad_token="<|endoftext|>"  # 显式设置填充token
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配设备
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).eval()  # 设置为评估模式

# ---------------------------
# 核心对话函数
# ---------------------------
def qwen_base_chat(prompt, max_new_tokens=512, temperature=0.7):
    """
    与基础模型对话
    :param prompt: 用户输入的问题/指令
    :param max_new_tokens: 生成的最大token数（默认512）
    :param temperature: 温度参数，控制生成随机性（0.0-1.0）
    :return: 模型生成的回答
    """
    # 构建对话模板
    messages = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": prompt}
    ]
    
    # 应用官方对话模板
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
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码输出
    response = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# ---------------------------
# 示例调用
# ---------------------------
if __name__ == "__main__":
    test_questions = [
        "请解释相对论的基本原理",
        "用Python实现快速排序算法",
        "如何预防感冒？"
    ]
    
    for question in test_questions:
        print(f"用户问题：{question}")
        print(f"模型回答：{qwen_base_chat(question)}\n{'-'*50}")