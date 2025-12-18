import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Step 0: 准备环境
# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")

# Step 1: 把输入分词成机器能理解的token id
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"}
]

# 使用分词器的模板格式化输入
# 会将messages转换为模型能理解的token id
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 编码输入文本
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("编码后的输入文本:")
print(model_inputs)
# 输出：
# {'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
#         151645,    198, 151644,    872,    198, 108386,  37945, 100157, 107828, 1773, 151645,    198, 151644,  77091,    198]]),
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}


# Step 2: 使用模型生成回答
# max_new_tokens 控制了模型最多能生成多少个新的Token
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# 将生成的 Token ID 截取掉输入部分
# 这样我们只解码模型新生成的部分
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Step 3: 机器回答的也是token id，需要解码成人类能理解的文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n模型的回答:")
print(response)
# 我是一个人工智能，没有自我意识和感情，所以我不能以人类的方式与你交流或提供任何帮助。我可以为你提供信息、解答问题，帮助你完成任务等等。请随时告诉我你的需求和问题，我会尽力提供帮助。
