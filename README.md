# Qwen3 医疗助手项目完整说明文档

## 项目概述

本项目是基于Qwen3大语言模型的医疗领域微调项目，实现了医疗智能助手功能。项目采用R1推理风格，能够提供专业的医疗咨询、诊断建议、治疗方案等多种医疗场景服务。

### 核心特性

- 🏥 **多场景医疗咨询**：支持10种医疗场景的专业咨询
- 🧠 **R1推理风格**：具备思考过程的医疗推理能力
- 🔧 **多种微调方式**：支持全参数微调和LoRA微调
- 📊 **完整训练流程**：从数据准备到模型部署的完整pipeline
- 🎯 **专业医疗提示词**：针对不同医疗场景优化的提示词模板

## 项目结构

```
week9/项目实战/
├── 📁 核心脚本
│   ├── medical_assistant.py      # 医疗助手主程序（350行）
│   ├── train.py                  # 全参数微调训练脚本（222行）
│   ├── train_lora.py             # LoRA微调训练脚本（188行）
│   ├── predict.py                # 模型推理脚本（111行）
│   ├── inference.py              # 基础推理脚本（57行）
│   ├── inference_lora.py         # LoRA推理脚本（56行）
│   ├── data.py                   # 数据处理脚本（57行）
│   └── download_model.py         # 模型下载脚本（15行）
├── 📁 文档说明
│   ├── README.md                 # 项目主说明文档
│   ├── PROJECT_DOC_CN.md         # 中文项目文档
│   ├── data.md                   # 数据处理说明
│   ├── train.md                  # 训练过程说明
│   ├── inference.md              # 推理说明
│   ├── inference_lora.md         # LoRA推理说明
│   ├── predict.md                # 预测脚本说明
│   └── medical_assistant_usage.md # 医疗助手使用说明
├── 📁 配置文件
│   ├── requirements.txt          # 依赖包列表
│   ├── sample_questions.json     # 示例问题集
│   └── train.ipynb              # Jupyter训练笔记本
├── 📁 数据文件
│   ├── train.jsonl              # 训练数据集
│   ├── val.jsonl                # 验证数据集
│   ├── train_format.jsonl       # 格式化训练数据
│   └── val_format.jsonl         # 格式化验证数据
├── 📁 模型文件
│   ├── models/                  # 基础模型存储
│   └── output/                  # 训练输出模型
└── 📁 日志文件
    └── swanlog/                 # SwanLab训练日志
```

## 技术架构

### 基础模型
- **Qwen3-0.6B**: 项目主要使用的轻量级模型，适合资源受限环境
- **Qwen3-1.7B**: 可选模型，用于对比实验

### 微调技术
1. **全参数微调**: 更新模型所有权重参数
2. **LoRA微调**: 低秩适应，高效微调技术

### 推理风格
- **R1推理风格**: 包含思考过程的推理模式
- **医疗专业提示词**: 针对不同医疗场景优化

## 环境要求

### 硬件要求（基于Qwen3-0.6B）
- **全参数微调**: 16GB显存
- **LoRA微调**: 12GB显存
- **推理**: 4GB显存（推荐）

### 软件依赖
```bash
swanlab                    # 训练监控
modelscope==1.22.0        # 模型下载
transformers              # 模型加载
datasets==3.2.0           # 数据处理
peft                      # LoRA微调
accelerate                # 训练加速
pandas                    # 数据处理
addict                    # 配置管理
```

## 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 数据准备
```bash
python data.py
```
自动完成：
- 数据集下载（delicate_medical_r1_data）
- 数据预处理和格式化
- 训练/验证集划分（9:1比例）

### 3. 模型训练

#### 全参数微调
```bash
python train.py
```

#### LoRA微调
```bash
python train_lora.py
```

### 4. 模型推理

#### 基础推理
```bash
python inference.py
```

#### 医疗助手交互
```bash
python medical_assistant.py
```

## 医疗场景支持

项目支持10种专业医疗场景：

| 场景ID | 场景名称 | 专业领域 | 示例问题 |
|--------|----------|----------|----------|
| 1 | 症状诊断 | 临床诊断 | "我最近经常头痛，伴有恶心，这是什么原因？" |
| 2 | 治疗方案 | 治疗指导 | "高血压患者应该如何控制血压？" |
| 3 | 疾病预防 | 预防医学 | "如何预防心血管疾病？" |
| 4 | 医学教育 | 医学知识 | "什么是高血压？" |
| 5 | 紧急评估 | 急诊医学 | "胸痛持续了3天，需要立即就医吗？" |
| 6 | 营养指导 | 营养学 | "糖尿病患者应该如何选择食物？" |
| 7 | 心理健康 | 心理学 | "如何缓解焦虑情绪？" |
| 8 | 儿科咨询 | 儿科学 | "儿童发热应该如何处理？" |
| 9 | 老年健康 | 老年医学 | "老年人如何预防跌倒？" |
| 10 | 女性健康 | 妇科学 | "更年期症状如何缓解？" |

## 使用方式

### 交互式使用
```bash
python medical_assistant.py
```

### 命令行使用
```bash
# 单次问答
python medical_assistant.py -q "我最近头痛，可能是什么原因？" -s diagnosis

# 批量处理
python medical_assistant.py -b sample_questions.json

# 指定模型路径
python medical_assistant.py -c ./output/Qwen3-0.6B/checkpoint-900
```

### 编程接口
```python
from medical_assistant import MedicalAssistant

# 创建助手实例
assistant = MedicalAssistant("./output/Qwen3-0.6B/checkpoint-900")
assistant.load_model()

# 询问问题
response = assistant.ask_question(
    "我最近胃部不适，可能是什么原因？", 
    scenario_type="diagnosis"
)
print(response)
```

## 训练监控

项目集成SwanLab进行训练监控：
- 实时损失曲线
- 学习率变化
- 模型性能指标
- 训练日志记录

访问地址：[SwanLab训练监控](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)

## 模型性能

### 训练效果对比
- **全参数微调**: 效果更好，但资源消耗大
- **LoRA微调**: 资源消耗小，效果略逊于全参数微调

### 推理示例
```
Question: 医生，我最近胃部不适，听说有几种抗溃疡药物可以治疗，您能详细介绍一下这些药物的分类、作用机制以及它们是如何影响胃黏膜的保护与损伤平衡的吗？

LLM:
<think>
用户询问抗溃疡药物的分类、作用机制，以及它们如何影响胃黏膜的保护和损伤平衡。
首先，我需要回忆抗溃疡药物的主要分类：
1. 抑酸药：H2受体拮抗剂、质子泵抑制剂
2. 胃黏膜保护剂：硫糖铝、铋剂
3. 促胃动力药：多潘立酮、西沙必利
4. 抗幽门螺杆菌药物：抗生素组合

每个类别的作用机制：
- 抑酸药：减少胃酸分泌，减轻对黏膜的损伤
- 保护剂：形成保护层，增强黏膜屏障
- 促胃动力药：增强胃蠕动，减少胃酸反流
- 抗幽门螺杆菌药物：杀灭病原体，减少损伤因素
</think>

当然可以。抗溃疡药物主要分为四类：抑酸药、胃黏膜保护剂、促胃动力药和抗幽门螺杆菌药物。

抑酸药如H₂受体拮抗剂和质子泵抑制剂，它们的作用是减少胃酸的分泌，从而减轻对胃黏膜的损伤。胃黏膜保护剂如硫糖铝和胶体果胶铋，它们能帮助形成一层保护膜，覆盖在胃黏膜表面，减少胃酸和胃蛋白酶对黏膜的侵蚀。

促胃动力药如多潘立酮和西沙必利，通过增强胃部的蠕动，减少胃内食物的滞留时间，从而降低溃疡的风险。抗幽门螺杆菌药物则通过抑制幽门螺杆菌的生长，减少其对胃黏膜的伤害。

这些药物的共同目标是通过不同的机制来平衡胃黏膜的保护因素和损伤因素，帮助溃疡愈合。如果您有具体的症状或担心某种药物，可以告诉我，我会为您推荐合适的治疗方案。
```

## 注意事项

### 医疗免责声明
- 本助手仅提供参考建议，不能替代专业医疗诊断
- 紧急情况请立即就医
- 所有建议仅供参考，具体治疗请咨询专业医生

### 技术限制
- 模型基于训练数据，可能存在知识局限性
- 建议结合最新医学指南使用
- 定期更新模型以保持准确性

## 扩展开发

### 添加新的医疗场景
1. 在`MEDICAL_PROMPTS`中添加新的提示词
2. 在`MEDICAL_SCENARIOS`中添加场景描述
3. 在`SAMPLE_QUESTIONS`中添加示例问题

### 自定义模型路径
```python
assistant = MedicalAssistant("./your/custom/model/path")
```

### 批量处理自定义问题
```json
[
  {
    "question": "您的问题",
    "scenario": "diagnosis",
    "max_tokens": 512
  }
]
```

## 相关资源

- **基础模型**: [Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B/summary)
- **数据集**: [delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
- **训练监控**: [SwanLab](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)
- **技术文档**: [Qwen3大模型微调入门实战](https://zhuanlan.zhihu.com/p/1903848838214705484)

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：
1. Fork项目仓库
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目遵循相应的开源许可证，请查看LICENSE文件了解详情。

---

**项目维护者**: Bryce-come  
**最后更新**: 2024年12月  
**版本**: v1.0.0
