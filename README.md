# Qwen3-Medical-SFT 项目说明文档

本项目基于 Qwen3-1.7B，提供医疗领域监督式微调（SFT）与推理的完整流程：数据准备、全量训练、（可选）LoRA 推理与仅推理脚本。

---

## 1. 环境与依赖
- Python 3.12（建议）
- 主要依赖（见 `requirements.txt`）：
  - swanlab
  - modelscope==1.22.0
  - transformers>=4.50.0
  - datasets==3.2.0
  - accelerate
  - pandas
  - addict
  - peft（LoRA）
- PyTorch：请安装与你显卡架构兼容的版本。若为 RTX 5090（sm_120），需安装支持该架构的新版本/夜版，否则将回退到 CPU。

安装：
```bash
pip install -r requirements.txt
```

---

## 2. 目录与文件作用
- `data.py`
  - 从 ModelScope 加载数据集，生成 `train.jsonl`、`val.jsonl`；含镜像端点与重试逻辑。
- `train.py`
  - 全量 SFT 训练：本地加载基础模型 `./Qwen/Qwen3-1___7B`；
  - 设备自适应（CUDA 优先，不可用/不兼容回退 CPU），CUDA 用 fp16，CPU 用 fp32；
  - 仅对 assistant 段计算损失，padding 由 `DataCollatorForSeq2Seq` 统一处理；
  - 输出到 `./output/Qwen3-1.7B/`（如 `checkpoint-1084`），末尾对验证集前 3 条做演示预测。
- `train_lora.py`
  - LoRA/PEFT 训练（可选），输出 LoRA 适配器（`adapter_config.json`、`adapter_model.safetensors`）。
- `inference.py`
  - 基础模型推理示例（单轮）。
- `inference_lora.py`
  - 在基础模型上加载 LoRA 适配器推理；请将 `model_id` 指向你的 LoRA 输出目录。
- `download_model.py`
  - 通过 ModelScope 下载 `Qwen/Qwen3-1.7B` 至本地；如网络受限建议使用已下载目录。
- `predict.py`
  - 仅推理脚本，默认 `./output/Qwen3-1.7B/checkpoint-1084`；支持命令行参数。
- `train.ipynb`
  - Notebook 版训练流程。
- `README.md` / `README_EN.md`
  - 项目说明。
- `train.jsonl`、`val.jsonl`
  - 原始训练/验证数据。
- `train_format.jsonl`、`val_format.jsonl`
  - 训练前重组后的数据（instruction/input/output）。

---

## 3. 模型说明（Qwen3-1.7B）
- 使用 `tokenizer.apply_chat_template(messages, ...)` 构造聊天输入；checkpoint 目录下保存 `chat_template.jinja`。
- 精度：CUDA 下 `float16`，CPU 下 `float32`（脚本自动选择）。

---

## 4. 数据格式与 I/O
1) 原始 JSONL（`train.jsonl` / `val.jsonl`）每行示例：
```json
{"question": "高血压如何饮食？", "think": "分析风险与饮食结构...", "answer": "建议控制盐分，增加蔬果..."}
```
2) 训练前重组（`train_format.jsonl` / `val_format.jsonl`）每行示例：
```json
{"instruction": "你是一个医学专家...", "input": "高血压如何饮食？", "output": "<think>分析...</think>\n建议控制盐分..."}
```
3) 推理输入（脚本内部构造）：
```python
messages = [
  {"role": "system", "content": instruction},
  {"role": "user", "content": user_input}
]
```
4) 推理输出：去除提示词后模型新生成的文本。

---

## 5. 使用方法
### 5.1 准备基础模型
- 本地基础模型目录：`./Qwen/Qwen3-1___7B`（注意目录名中的下划线）。
- 若未下载，可运行 `download_model.py`（需可访问 ModelScope）。

### 5.2 生成数据
```bash
python data.py
```

### 5.3 全量训练
```bash
python train.py
```
- 输出目录：`./output/Qwen3-1.7B/`（如 `checkpoint-1084`）。
- 若已训练完成，不要再次运行 `train.py` 以免重复训练。

### 5.4 仅推理（不训练）
- 使用 checkpoint：
```bash
python predict.py -i "我血糖偏高，三餐如何安排？" -c ./output/Qwen3-1.7B/checkpoint-1084 -m 256
```
- 基础模型示例：
```bash
python inference.py
```

### 5.5 LoRA（可选）
- 训练：
```bash
python train_lora.py
```
- 推理：修改 `inference_lora.py` 的 `model_id` 为 LoRA 输出目录后运行：
```bash
python inference_lora.py
```

---

## 6. 关键实现细节
- 设备/精度自适应：优先 CUDA；如架构不被当前 PyTorch 支持（如 sm_120）或不可用，则回退 CPU。CUDA 用 fp16，CPU 用 fp32。
- Label 策略（避免 loss=0）：仅对 assistant 段计算损失；prompt 段 `labels=-100`；`DataCollatorForSeq2Seq(..., label_pad_token_id=-100)` 负责 padding。
- 生成阶段：显式传入 `attention_mask` 给 `generate`；只解码新增 token，避免回显提示词。
- 日志：使用 `swanlab` 记录 loss、eval_loss、学习率等。

---

## 7. 命令速查
- 生成数据：`python data.py`
- 训练：`python train.py`
- 仅推理（checkpoint）：`python predict.py -i "你的问题" -c ./output/Qwen3-1.7B/checkpoint-1084 -m 256`
- 基础模型推理：`python inference.py`
- LoRA 训练：`python train_lora.py`
- LoRA 推理：修改路径后 `python inference_lora.py`

---

## 8. 常见问题（FAQ）
- RTX 5090 不被当前 PyTorch 支持导致回退 CPU？安装支持 sm_120 的 PyTorch（参考官网 Get Started）。
- Windows `UnicodeDecodeError: 'gbk'...`？已在读取 JSONL 时强制 `encoding='utf-8'`，请确认源数据为 UTF-8。
- 训练 loss 为 0？多因 label/pad 构造不当；本项目已仅对 assistant 段计算损失并由 collator 统一 padding。
- 推理很慢？多因 CPU 推理。可升级 PyTorch 以启用 GPU；或调小 `-m`、常驻加载模型、或采用量化。

---

## 9. 免责声明
本项目仅用于研究与学习，内容不构成专业医疗建议。实际医疗问题请咨询专业医生。
