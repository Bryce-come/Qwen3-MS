#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗助手集成脚本
基于 Qwen3-1.7B 医疗微调模型，提供多种医疗场景的智能助手功能
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
from datetime import datetime
import os

# 医疗专业提示词模板
MEDICAL_PROMPTS = {
    "diagnosis": "你是一名经验丰富的临床医生，请根据患者描述的症状，给出专业的初步诊断建议和进一步检查建议。",
    "treatment": "你是一名医生，请根据病情描述，提供专业的治疗方案建议，包括用药指导和注意事项。",
    "prevention": "你是一名预防医学专家，请提供专业的疾病预防建议和健康生活方式指导。",
    "education": "你是一名医学教育专家，请用通俗易懂的方式解释医学概念，帮助患者理解疾病相关知识。",
    "emergency": "你是一名急诊科医生，请评估症状的紧急程度，给出是否需要立即就医的建议。",
    "nutrition": "你是一名营养师，请根据患者的健康状况，提供专业的营养建议和饮食指导。",
    "mental_health": "你是一名心理医生，请关注患者的心理健康状况，提供专业的心理支持和建议。",
    "pediatric": "你是一名儿科医生，请根据儿童的特殊情况，提供适合的医疗建议和护理指导。",
    "geriatric": "你是一名老年医学专家，请考虑老年人的特殊需求，提供适合的医疗建议。",
    "women_health": "你是一名妇科医生，请为女性患者提供专业的健康建议和医疗指导。"
}

# 常见医疗场景
MEDICAL_SCENARIOS = {
    "1": "症状诊断",
    "2": "治疗方案",
    "3": "疾病预防",
    "4": "医学教育",
    "5": "紧急评估",
    "6": "营养指导",
    "7": "心理健康",
    "8": "儿科咨询",
    "9": "老年健康",
    "10": "女性健康"
}

# 预设问题示例
SAMPLE_QUESTIONS = {
    "diagnosis": [
        "我最近经常头痛，伴有恶心，这是什么原因？",
        "胸痛持续了3天，呼吸时加重，可能是什么问题？",
        "持续发热一周，体温在38-39度之间，需要做什么检查？"
    ],
    "treatment": [
        "高血压患者应该如何控制血压？",
        "糖尿病患者除了控制血糖，还需要注意什么？",
        "感冒期间应该怎么用药？"
    ],
    "prevention": [
        "如何预防心血管疾病？",
        "冬季如何预防感冒？",
        "如何预防骨质疏松？"
    ],
    "education": [
        "什么是高血压？",
        "糖尿病的发病机制是什么？",
        "心肌梗死是如何发生的？"
    ]
}

class MedicalAssistant:
    def __init__(self, checkpoint_path="./output/Qwen3-0.6B/checkpoint-900"):
        """初始化医疗助手"""
        self.checkpoint_path = checkpoint_path
        self.device, self.dtype = self._select_device_and_dtype()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def _select_device_and_dtype(self):
        """选择设备和数据类型"""
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
    
    def load_model(self):
        """加载模型和分词器"""
        print("正在加载医疗助手模型...")
        
        # 检查路径是否存在
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, 
            use_fast=False, 
            trust_remote_code=True,
            local_files_only=True  # 只使用本地文件
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, 
            torch_dtype=self.dtype,
            local_files_only=True  # 只使用本地文件
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成！使用设备: {self.device}")
    
    def predict(self, messages, max_new_tokens=512):
        """执行预测"""
        model_device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids.to(model_device)
        attention_mask = inputs.attention_mask.to(model_device) if hasattr(inputs, "attention_mask") else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        # 只解码新生成部分
        new_tokens = generated[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return response
    
    def ask_question(self, question, scenario_type="diagnosis", max_tokens=512):
        """询问医疗问题"""
        if scenario_type not in MEDICAL_PROMPTS:
            scenario_type = "diagnosis"
        
        messages = [
            {"role": "system", "content": MEDICAL_PROMPTS[scenario_type]},
            {"role": "user", "content": question}
        ]
        
        # 记录对话历史
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_type,
            "question": question,
            "response": None
        })
        
        response = self.predict(messages, max_new_tokens=max_tokens)
        
        # 更新对话历史
        self.conversation_history[-1]["response"] = response
        
        return response
    
    def show_scenarios(self):
        """显示可用的医疗场景"""
        print("\n🏥 医疗助手 - 可用场景:")
        print("=" * 50)
        for key, value in MEDICAL_SCENARIOS.items():
            print(f"{key:2}. {value}")
        print("=" * 50)
    
    def show_sample_questions(self, scenario_type):
        """显示示例问题"""
        if scenario_type in SAMPLE_QUESTIONS:
            print(f"\n📋 {MEDICAL_SCENARIOS.get(scenario_type, '医疗咨询')} - 示例问题:")
            print("-" * 40)
            for i, question in enumerate(SAMPLE_QUESTIONS[scenario_type], 1):
                print(f"{i}. {question}")
            print("-" * 40)
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🤖 医疗助手已启动！")
        print("输入 'help' 查看帮助，输入 'quit' 退出")
        
        while True:
            try:
                # 显示场景选择
                self.show_scenarios()
                
                # 选择场景
                scenario_choice = input("\n请选择医疗场景 (1-10): ").strip()
                if scenario_choice == 'quit':
                    break
                elif scenario_choice == 'help':
                    self.show_help()
                    continue
                elif scenario_choice not in MEDICAL_SCENARIOS:
                    print("❌ 无效选择，请重新输入")
                    continue
                
                # 获取场景类型
                scenario_type = list(MEDICAL_PROMPTS.keys())[int(scenario_choice) - 1]
                
                # 显示示例问题
                self.show_sample_questions(scenario_type)
                
                # 获取用户问题
                question = input(f"\n请输入您的{MEDICAL_SCENARIOS[scenario_choice]}问题: ").strip()
                if not question:
                    print("❌ 问题不能为空")
                    continue
                
                # 生成回答
                print("\n🔄 正在分析您的问题...")
                start_time = time.time()
                
                response = self.ask_question(question, scenario_type)
                
                end_time = time.time()
                
                # 显示回答
                elapsed_time = end_time - start_time
                print(f"\n💡 医疗助手回答 (耗时: {elapsed_time:.2f}秒):")
                print("=" * 60)
                print(response)
                print("=" * 60)
                
                # 询问是否继续
                continue_choice = input("\n是否继续咨询？(y/n): ").strip().lower()
                if continue_choice in ['n', 'no', '否']:
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用医疗助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                continue
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 医疗助手使用帮助:")
        print("=" * 50)
        print("1. 选择医疗场景 (1-10)")
        print("2. 输入您的医疗问题")
        print("3. 获得专业的医疗建议")
        print("\n💡 提示:")
        print("- 本助手仅提供参考建议，不能替代专业医疗诊断")
        print("- 紧急情况请立即就医")
        print("- 输入 'quit' 退出程序")
        print("=" * 50)
    
    def save_conversation(self, filename=None):
        """保存对话历史"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_conversation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"💾 对话历史已保存到: {filename}")
    
    def batch_questions(self, questions_file):
        """批量处理问题"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            print(f"📝 开始批量处理 {len(questions)} 个问题...")
            
            results = []
            for i, q in enumerate(questions, 1):
                print(f"\n处理第 {i}/{len(questions)} 个问题...")
                response = self.ask_question(
                    q.get('question', ''), 
                    q.get('scenario', 'diagnosis'),
                    q.get('max_tokens', 512)
                )
                
                results.append({
                    "question": q.get('question', ''),
                    "scenario": q.get('scenario', 'diagnosis'),
                    "response": response
                })
            
            # 保存结果
            output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 批量处理完成！结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"❌ 批量处理失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="医疗助手 - 基于Qwen3-0.6B的智能医疗咨询系统")
    parser.add_argument("--checkpoint", "-c", type=str, 
                       default="./output/Qwen3-0.6B/checkpoint-900", 
                       help="模型检查点路径")
    parser.add_argument("--question", "-q", type=str, 
                       help="直接询问问题（需要配合 --scenario 使用）")
    parser.add_argument("--scenario", "-s", type=str, 
                       default="diagnosis", 
                       choices=list(MEDICAL_PROMPTS.keys()),
                       help="医疗场景类型")
    parser.add_argument("--max-tokens", "-m", type=int, 
                       default=512, 
                       help="最大生成token数")
    parser.add_argument("--batch", "-b", type=str, 
                       help="批量处理问题文件（JSON格式）")
    parser.add_argument("--save-history", action="store_true", 
                       help="保存对话历史")
    
    args = parser.parse_args()
    
    # 创建医疗助手实例
    assistant = MedicalAssistant(args.checkpoint)
    
    # 加载模型
    assistant.load_model()
    
    if args.batch:
        # 批量处理模式
        assistant.batch_questions(args.batch)
    elif args.question:
        # 单次问答模式
        print(f"🤖 医疗助手回答:")
        print("=" * 50)
        response = assistant.ask_question(args.question, args.scenario, args.max_tokens)
        print(response)
        print("=" * 50)
    else:
        # 交互模式
        assistant.interactive_mode()
    
    # 保存对话历史
    if args.save_history and assistant.conversation_history:
        assistant.save_conversation()


if __name__ == "__main__":
    main()
