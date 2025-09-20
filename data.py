from modelscope.msdatasets import MsDataset
import json
import random
import os
import time

# 设置ModelScope镜像源
os.environ['MODELSCOPE_ENDPOINT'] = 'https://modelscope.oss-cn-beijing.aliyuncs.com'

# 设置随机种子以确保可重复性
random.seed(42)

# 加载数据集，添加重试机制
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"尝试加载数据集 (第 {attempt + 1} 次)...")
        ds = MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')
        print("数据集加载成功！")
        break
    except Exception as e:
        print(f"第 {attempt + 1} 次尝试失败: {str(e)}")
        if attempt < max_retries - 1:
            print("等待5秒后重试...")
            time.sleep(5)
        else:
            print("所有重试都失败了，请检查网络连接或数据集是否存在")
            raise e

# 将数据集转换为列表
data_list = list(ds)

# 随机打乱数据
random.shuffle(data_list)

# 计算分割点
split_idx = int(len(data_list) * 0.9)

# 分割数据
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

# 保存训练集
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# 保存验证集
with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"数据集已分割完成：")
print(f"训练集大小：{len(train_data)}")
print(f"验证集大小：{len(val_data)}")