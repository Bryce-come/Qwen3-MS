import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from modelscope import snapshot_download
import os
import swanlab

os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    })


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件（Windows 下强制使用 UTF-8 防止解码错误）
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think> \n {answer}"
            message = { 
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """ 
    # 构造prompt和回答
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 不手动添加pad，由collator负责padding
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]

    # labels 仅对assistant部分计算损失，prompt部分置为 -100
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def select_device_and_dtype():
    # 优先尝试 CUDA，但对不被当前PyTorch支持的架构（如 sm_120）回退CPU
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 12:
                # 当前PyTorch不支持sm_120，回退CPU
                raise RuntimeError("Unsupported CUDA capability for current PyTorch")
            _ = torch.zeros(1, device="cuda")  # 实测一次分配
            print("INFO: Using CUDA (GPU) for training.")
            return "cuda", torch.float16
        except Exception as e:
            print(f"WARN: CUDA available but failed to use, falling back to CPU. Reason: {e}")
            pass
    print("INFO: No compatible CUDA device found, using CPU for training.")
    return "cpu", torch.float32


def predict(messages, model, tokenizer, device):
    # 以模型的实际设备为准，避免不一致
    model_device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    # 显式提供 attention_mask，并搬到模型设备
    input_ids = model_inputs.input_ids.to(model_device)
    attention_mask = model_inputs.attention_mask.to(model_device) if hasattr(model_inputs, "attention_mask") else None

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids[0]):]
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 定义模型名称
model_name = "Qwen/Qwen3-0.6B"

# 获取脚本所在目录，并创建模型缓存路径
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "models")

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_name, cache_dir=cache_path, revision="master")

# Transformers加载模型权重（本地）
device, load_dtype = select_device_and_dtype()

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
# 确保存在pad token，便于padding
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=load_dtype)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
model.to(device)

# 加载、处理数据集和测试集
train_dataset_path = os.path.join(script_path, "train.jsonl")
val_dataset_path = os.path.join(script_path, "val.jsonl")

train_jsonl_new_path = os.path.join(script_path, "train_format.jsonl")
val_jsonl_new_path = os.path.join(script_path, "val_format.jsonl")

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(val_jsonl_new_path):
    dataset_jsonl_transfer(val_dataset_path, val_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 得到验证集
eval_df = pd.read_json(val_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

# 使用能够为labels补 -100 的 collator
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

args = TrainingArguments(
    output_dir=os.path.join(script_path, "output/Qwen3-0.6B"),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-0.6B",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)

trainer.train()

# 用验证集的前3条，主观看模型
test_df = pd.read_json(val_jsonl_new_path, lines=True)[:3]

test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer, device)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()