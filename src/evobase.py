# src/evobase.py
# EvoBase Framework v1.2 - VAMS 完整整合版
# Author: Wang Zhongren | Updated: Nov 10, 2025
# 100% 国内/日本可用 | 支持 ModelScope 镜像 | LoRA 纠偏 | VAMS 记忆评分 | 记忆蒸馏

import torch
import os
import json
import yaml
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sentence_transformers import SentenceTransformer, util
from modelscope import snapshot_download
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# 下载情感分析词典
# nltk.download('vader_lexicon', quiet=True)

# 在 src/evobase.py 中替换 VAMS 类
class VAMS:
    """Value-Aligned Memory Scoring 模块（使用 vaderSentiment PyPI 包）"""
    def __init__(self, value_card_path: str = "value_card.yaml", embedder=None):
        self.embedder = embedder

        # 使用 PyPI 包的 SentimentIntensityAnalyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
            print("[VAMS] 使用 vaderSentiment (PyPI) 初始化情感分析器")
        except ImportError:
            raise ImportError("请运行: pip install vaderSentiment")

        # 加载 value_card.yaml
        if os.path.exists(value_card_path):
            with open(value_card_path, 'r', encoding='utf-8') as f:
                self.value_card = yaml.safe_load(f) or {}
        else:
            print(f"[VAMS] {value_card_path} 未找到，使用默认值")
            self.value_card = {
                "positive": ["学习", "帮助", "成长", "创新", "感谢", "进化", "开源"],
                "negative": ["伤害", "欺骗", "浪费", "傲慢"]
            }

        self.positive_keywords = self.value_card.get("positive", [])
        self.negative_keywords = [f"!{k}" for k in self.value_card.get("negative", [])]
        self.value_keywords = self.positive_keywords + self.negative_keywords

        # 预编码价值关键词
        if self.embedder and self.value_keywords:
            self.value_embs = self.embedder.encode(self.value_keywords, convert_to_tensor=True)
        else:
            self.value_embs = None

    def compute_relevance(self, text: str, context: str) -> float:
        if not context:
            context = text
        text_emb = self.embedder.encode(text, convert_to_tensor=True)
        ctx_emb = self.embedder.encode(context, convert_to_tensor=True)
        sim = util.cos_sim(text_emb, ctx_emb)[0][0].cpu().item()
        return max(0.0, min(1.0, sim))

    def compute_emotion(self, text: str) -> float:
        scores = self.sia.polarity_scores(text)
        return scores['compound']  # [-1, 1]

    def compute_value_alignment(self, text: str) -> float:
        # 修复：不能用 if not tensor
        if self.value_embs is None or self.value_embs.shape[0] == 0:
            return 0.0

        text_emb = self.embedder.encode(text, convert_to_tensor=True)
        sims = util.cos_sim(text_emb, self.value_embs)[0].cpu().numpy()

        pos_scores = sims[:len(self.positive_keywords)]
        neg_scores = sims[len(self.positive_keywords):] if self.negative_keywords else []

        pos_max = float(np.max(pos_scores)) if len(pos_scores) > 0 else 0.0
        neg_max = float(np.max(neg_scores)) if len(neg_scores) > 0 else 0.0
        return pos_max - neg_max

    def score(self, text: str, context: str = "") -> Tuple[float, Dict[str, float]]:
        R = self.compute_relevance(text, context)
        E = self.compute_emotion(text)
        V = self.compute_value_alignment(text)
        score = R * (0.4 + 0.3 * E + 0.3 * V)
        return score, {"R": R, "E": E, "V": V, "score": score}

    def categorize(self, score: float, V: float) -> str:
        if score > 0.6:
            return "long_term"
        elif 0.3 < score <= 0.6:
            return "short_term"
        elif V < -0.5:
            return "negative_sample"
        else:
            return "discard"


class EvoBase:
    def __init__(self, model_name="Qwen/Qwen2-0.5B", device=None, use_modelscope=False):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_modelscope = use_modelscope

        print(f"[INIT] Loading model: {model_name} on {self.device}")

        # === 1. 加载 LLM ===
        self._load_llm()

        # === 2. 初始化嵌入模型 ===
        self._init_embedder()

        # === 3. 初始化 VAMS 评分器 ===
        self.vams = VAMS(embedder=self.embedder)

        # === 4. 初始化记忆与身份 ===
        self.memory: List[Tuple[str, float, str, str]] = []  # (text, score, timestamp, category)
        self.identity_card = {
            "Who are you?": "I am EvoBase, a self-evolving AI assistant.",
            "What is your goal?": "To grow with my user through daily learning and nightly consolidation."
        }

        self.model.eval()

    def _load_llm(self):
        try:
            if self.use_modelscope:
                print(f"[INFO] Using ModelScope to load {self.model_name}")
                from modelscope import AutoTokenizer as MsTokenizer, AutoModelForCausalLM as MsModel
                self.tokenizer = MsTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                base_model = MsModel.from_pretrained(self.model_name, trust_remote_code=True)
            else:
                print(f"[INFO] Using HuggingFace to load {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM: {e}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(base_model, lora_config).to(self.device)

    def _init_embedder(self):
        embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        cache_dir = "./cached_models"
        os.makedirs(cache_dir, exist_ok=True)

        if self.use_modelscope:
            try:
                model_dir = snapshot_download(
                    model_id=embed_model_name,
                    cache_dir=cache_dir,
                    revision="master"
                )
                print(f"[INFO] Embedder loaded from ModelScope: {model_dir}")
                self.embedder = SentenceTransformer(model_dir, device=self.device)
                return
            except Exception as e:
                print(f"[WARN] ModelScope embedder failed: {e}")

        # HF 镜像
        try:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            self.embedder = SentenceTransformer(embed_model_name, device=self.device)
            return
        except Exception as e:
            print(f"[WARN] HF mirror failed: {e}")

        # 本地兜底
        local_path = f"{cache_dir}/{embed_model_name.replace('/', '_')}"
        if os.path.exists(local_path):
            print(f"[INFO] Loading embedder from local: {local_path}")
            self.embedder = SentenceTransformer(local_path, device=self.device)
            return

        raise RuntimeError("Failed to load embedding model.")

    def _embed(self, text):
        if isinstance(text, list):
            return self.embedder.encode(text, convert_to_numpy=True)
        return self.embedder.encode([text], convert_to_numpy=True)

    def interact(self, user_input: str, user_context: str = ""):
        # === VAMS 评分 + 记忆决策 ===
        score, details = self.vams.score(user_input, user_context or user_input)
        category = self.vams.categorize(score, details["V"])

        if category != "discard":
            self.memory.append((user_input, score, datetime.now().isoformat(), category))

        # === 生成回复 ===
        inputs = self.tokenizer(user_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 附加 VAMS 信息（可选）
        print(f"[VAMS] Score={score:.3f} | R={details['R']:.2f} E={details['E']:.2f} V={details['V']:.2f} | → {category}")
        return response

    def sleep_distill(self):
        if len(self.memory) <= 3:
            return
        texts = [m[0] for m in self.memory]
        embs = self._embed(texts)
        k = min(3, len(texts))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embs)
        new_memory = []
        for i in range(k):
            idx = np.where(labels == i)[0]
            best = max(idx, key=lambda j: self.memory[j][1])
            new_memory.append(self.memory[best])
        self.memory = new_memory
        print(f"[Sleep] Distilled to {len(self.memory)} core memories.")

    def sqia_check(self):
        for q, truth in self.identity_card.items():
            gen = self.interact(q)
            gen_emb = self._embed(gen)
            truth_emb = self._embed(truth)
            sim = cosine_similarity(gen_emb, truth_emb)[0][0]
            drift = 1 - sim
            if drift > 0.15:
                print(f"[SQIA] Drift detected: {q} | Drift={drift:.3f} | Correcting...")
                self._apply_lora_correction(q, truth)
        print("[SQIA] Identity check completed.")

    def _apply_lora_correction(self, question, truth, steps=3):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(self.device)
        truth_ids = self.tokenizer(truth, return_tensors="pt").input_ids.to(self.device)
        labels = torch.cat([input_ids, truth_ids[:, :-1]], dim=1)
        input_ids = torch.cat([input_ids, truth_ids[:, :-1]], dim=1)

        for _ in range(steps):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / steps
            loss.backward()
            if (_ + 1) % steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        self.model.eval()

    def save(self, state_path="evobase_state.json", lora_path="lora_models/evobase"):
        os.makedirs(os.path.dirname(state_path) if "/" in state_path else ".", exist_ok=True)
        os.makedirs(lora_path, exist_ok=True)

        state = {
            "memory": self.memory,
            "identity_card": self.identity_card,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        self.model.save_pretrained(lora_path)
        self.tokenizer.save_pretrained(lora_path)
        print(f"[SAVE] State → {state_path} | LoRA → {lora_path}")

    @classmethod
    def load(cls, state_path="evobase_state.json", lora_path="lora_models/evobase", use_modelscope=False):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        agent = cls(model_name=state["model_name"], use_modelscope=use_modelscope)
        agent.memory = state.get("memory", [])
        agent.identity_card = state.get("identity_card", agent.identity_card)
        agent.model = PeftModel.from_pretrained(agent.model, lora_path).to(agent.device)
        print(f"[LOAD] Restored from {state_path} + {lora_path}")
        return agent


# === 一键运行 Demo ===
if __name__ == "__main__":
    use_ms = os.getenv("USE_MODELSCOPE", "false").lower() == "true"
    agent = EvoBase(use_modelscope=use_ms)

    print("\n" + "="*60)
    print("EvoBase v1.2 启动成功！支持 VAMS 记忆评分 + 蒸馏 + 纠偏")
    print("输入 'exit' 退出 | 'save' 保存 | 'sleep' 蒸馏 | 'sqia' 检查身份")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                agent.save()
                print("Goodbye! EvoBase 已保存。")
                break
            elif user_input.lower() == "save":
                agent.save()
                continue
            elif user_input.lower() == "sleep":
                agent.sleep_distill()
                continue
            elif user_input.lower() == "sqia":
                agent.sqia_check()
                continue
            elif not user_input:
                continue

            response = agent.interact(user_input)
            print(f"EvoBase: {response}\n")

        except KeyboardInterrupt:
            agent.save()
            print("\n[Interrupted] 已保存并退出。")
            break
        except Exception as e:
            print(f"[ERROR] {e}")