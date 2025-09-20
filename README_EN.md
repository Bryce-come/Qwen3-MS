# Qwen3 Fine-tuning in Practice: Medical R1 Inference-Style Chat

[中文](README.md)

[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)

- **Base Model:** [Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B/summary)
- **Fine-tuned Model:** [Qwen3-1.7b-Medical-R1-sft](https://modelscope.cn/models/testUser/Qwen3-1.7b-Medical-R1-sft/summary)
- **Dataset:** [delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
- **SwanLab**：[qwen3-sft-medical](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/runs/agps0dkifth5l1xytcdyk/chart)
- **Fine-tuning Methods:** Full-parameter Fine-tuning, LoRA Fine-tuning
- **Inference Style:** R1 Reasoning Style
- **Hardware Requirements:**
  - **Full-parameter Fine-tuning:** 32GB VRAM
  - **LoRA Fine-tuning:** 28GB VRAM
- **Tutorial:** [Hands-on Guide for Qwen3 LLM Fine-tuning (Complete Code)](https://zhuanlan.zhihu.com/p/1903848838214705484)

> To further reduce VRAM requirements, consider using Qwen3-0.6B or decreasing `MAX_LENGTH`.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Automatically handles dataset downloading, preprocessing, and validation set splitting. Generates `train.jsonl` and `val.jsonl`.

```bash
python data.py
```

## Training

### Full-parameter Fine-tuning

```bash
python train.py
```

### LoRA Fine-tuning

```bash
python train_lora.py
```

SwanLab Training Logs: [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)

Comparison shows full-parameter fine-tuning outperforms LoRA:

![](./readme_images/charts.png)

## Inference

**Full-parameter Fine-tuning**

```bash
python inference.py
```

**LoRA Fine-tuning**

```bash
python inference_lora.py
```

## Related Tools

- [SwanLab](https://github.com/SwanHubX/SwanLab): Open-source, modern deep learning experiment tracking and visualization platform
- [Transformers](https://github.com/huggingface/transformers): HuggingFace's library for state-of-the-art pretrained models
- [PEFT](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning library for large language models