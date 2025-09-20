import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import re

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_NEW_TOKENS = 512


def select_device_and_dtype():
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 12:
                raise RuntimeError("Unsupported CUDA capability for current PyTorch")
            _ = torch.zeros(1, device="cuda")
            return "cuda", torch.float16
        except Exception:
            pass
    return "cpu", torch.float32


def predict(messages, model, tokenizer):
    model_device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids.to(model_device)
    attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # 只解码新生成部分
    new_tokens = generated[:, input_ids.shape[1]:]
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    # 自动查找最新的 checkpoint
    # 构造相对于脚本所在目录的路径，使其不受运行位置的影响
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output/Qwen3-0.6B")
    latest_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [
            d for d in os.listdir(output_dir) 
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            # 通过 checkpoint 编号排序找到最新的
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            print(f"INFO: 自动找到最新的 checkpoint: {latest_checkpoint}")

    parser = argparse.ArgumentParser(description="Qwen3 推理脚本（命令行读取）")
    parser.add_argument("--input", "-i", type=str, help="用户输入的问题文本。如果不提供，将在命令行交互读取。")
    parser.add_argument("--instruction", "-s", type=str, default=PROMPT, help="system 提示词")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=latest_checkpoint, 
        help=f"checkpoint 路径。默认为自动查找的最新 checkpoint"
    )
    parser.add_argument("--max_new_tokens", "-m", type=int, default=MAX_NEW_TOKENS, help="生成的最大新token数")
    args = parser.parse_args()

    # 检查 checkpoint 路径是否有效
    if not args.checkpoint or not os.path.isdir(args.checkpoint):
        print(f"\n错误：必须提供一个有效的 checkpoint 路径。")
        print(f"路径 '{args.checkpoint}' 无效或不存在。")
        print(f"请确保您已经在 '{output_dir}' 目录下完成了训练并生成了 checkpoint，或通过 -c 参数手动指定。")
        exit(1)

    # 覆盖默认最大生成长度（如用户提供）
    MAX_NEW_TOKENS = args.max_new_tokens

    # 获取输入
    user_input = args.input
    if not user_input:
        try:
            user_input = input("请输入用户问题：").strip()
        except EOFError:
            user_input = ""

    device, dtype = select_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model.to(device)
    model.eval()

    messages = [
        {"role": "system", "content": args.instruction},
        {"role": "user", "content": user_input},
    ]

    output = predict(messages, model, tokenizer)
    print(output)


