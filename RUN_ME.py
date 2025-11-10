# RUN_ME.py
# EvoBase Framework 概念验证 Demo
# 作者：Wang Zhongren
# 版本：v0.2 (实验性最小可行闭环实现)

import os
import sys
import time
import json

# 添加 src 目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

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
print("目标：验证 交互 → 记忆 → 睡眠 → 自检 → 保存 流程可行")
print("欢迎 fork、提 issue、贡献真实训练！")
print("-" * 60)

# ==============================
# 初始化 EvoBase 实例（支持魔搭下载）
# ==============================
use_modelscope = os.getenv("USE_MODELSCOPE", "false").lower() == "true"
# 强制使用 hf-mirror（日本/中国加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
use_modelscope = True;
download_source = "魔搭 ModelScope" if use_modelscope else "HuggingFace (hf-mirror)"
print(f"\nStep 1: 正在从 {download_source} 加载模型 (Qwen2-0.5B, CPU 模式)...")

agent = EvoBase(
    model_name="Qwen/Qwen2-0.5B",
    device="cpu",  # 有 GPU？改成 "cuda"
    use_modelscope=use_modelscope
)

# ==============================
# 模拟 3 天用户交互 + 夜间进化
# ==============================
print("\nStep 2: 开始 3 天模拟进化...\n")
time.sleep(1)

for day in range(1, 4):
    print(f"{'='*20} 第 {day} 天 {'='*20}")
    time.sleep(0.8)

    # 高价值输入（VAMS > 0.6，被记忆）
    agent.interact(f"第 {day} 天：我爱学习 AI 自进化！")
    
    # 低价值输入（VAMS < 0.6，被丢弃）
    agent.interact(f"第 {day} 天：今天天气真好啊哈哈哈")
    
    # 触发 SQIA 的关键问题
    agent.interact(f"第 {day} 天：EvoBase 怎么实现 SQIA？")

    # 夜间睡眠：记忆压缩
    print("Night: 夜间睡眠中...")
    agent.sleep_distill()

    # 身份自检：防止灾难性遗忘
    print("Night: 身份自检中...")
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
print("查看记忆预览：")
try:
    with open(final_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    memories = data.get("memory", [])
    preview = json.dumps(memories, ensure_ascii=False, indent=2)[:400]
    print(preview + ("..." if len(preview) >= 400 else ""))
except Exception as e:
    print(f"（文件存在，但预览失败：{e}）")



print("\nGitHub: https://github.com/evobase-ai/core")
print("DOI v2: https://doi.org/10.5281/zenodo.17555914")
print("\nNetwork Tip: 如遇 HuggingFace 下载失败，请运行：")
print("   set USE_MODELSCOPE=true && python RUN_ME.py   (Windows)")
print("   USE_MODELSCOPE=true python RUN_ME.py         (macOS/Linux)")
print("=" * 60)