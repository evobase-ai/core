# RUN_ME.py
# EvoBase Framework 概念验证 Demo
# 作者：Wang Zhongren
# 版本：v0.1 (实验性最小可行实现)

import os
import sys
import time

# 添加 src 目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.evobase import EvoBase
except ImportError as e:
    print("Error: 无法导入 EvoBase。请确保 src/evobase.py 存在！")
    print("错误信息：", e)
    sys.exit(1)

# ==============================
# 实验性声明（防尴尬神器）
# ==============================
print("=" * 60)
print("EvoBase Framework 概念验证 Demo")
print("=" * 60)
print("Warning: 这是 最小可行概念验证 (Proof-of-Concept)")
print("Warning: 仅演示架构闭环，不包含真实 LoRA 微调")
print("Warning: 性能提升需 30 天真实用户数据")
print("目标：验证 交互 → 记忆 → 睡眠 → 自检 流程可行")
print("欢迎 fork、提 issue、贡献真实训练！")
print("-" * 60)

# ==============================
# 初始化 EvoBase 实例
# ==============================
print("\nStep 1: 正在加载模型 (Qwen2-0.5B, CPU 模式)...")
agent = EvoBase(
    model_name="Qwen/Qwen2-0.5B",  # 小模型，CPU 可跑
    device="cpu"                   # 换成 "cuda" 如果有 GPU
)

# ==============================
# 模拟 3 天用户交互 + 夜间进化
# ==============================
print("\nStep 2: 开始 3 天模拟进化...\n")
time.sleep(1)

for day in range(1, 4):
    print(f"{'='*20} 第 {day} 天 {'='*20}")
    time.sleep(0.8)

    # 高价值输入（应该被记忆）
    agent.interact(f"第 {day} 天：我爱学习 AI 自进化！")
    
    # 低价值输入（应该被丢弃）
    agent.interact(f"第 {day} 天：今天天气真好啊哈哈哈")
    
    # 中等价值输入（观察）
    agent.interact(f"第 {day} 天：EvoBase 怎么实现 SQIA？")

    # 夜间睡眠：记忆压缩
    agent.sleep_distill()

    # 身份自检：防止灾难性遗忘
    agent.sqia_check()

    print(f"第 {day} 天结束，模型已“睡”过一次\n")
    time.sleep(1)

# ==============================
# 保存最终状态
# ==============================
final_path = "final_evolution_state.json"
agent.save(final_path)

print("=" * 60)
print("概念验证完成！")
print(f"最终记忆状态已保存至：{final_path}")
print("查看文件内容：")
try:
    import json
    with open(final_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data["memory"], ensure_ascii=False, indent=2)[:300] + "...")
except:
    print("（文件存在，但预览失败）")

print("\n下一步建议：")
print("1. 替换为你的真实对话数据")
print("2. 启用真实 LoRA 微调")
print("3. 运行 30 天，观测性能曲线")
print("4. 提交 PR 帮我进化！")

print("\nGitHub: https://github.com/evobase-ai/core")
print("DOI v2: https://doi.org/10.5281/zenodo.17555914")
print("=" * 60)
