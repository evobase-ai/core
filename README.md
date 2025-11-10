# EvoBase Framework  
*Also known as EvolvingAgent-Zero in academic literature*

**A closed-loop self-evolution framework for on-device small language models (SLMs)**   
**Assisted by**: ChatGPT and Grok  
**License**: MIT

[![DOI v1: EvolvingAgent-Zero](https://zenodo.org/badge/DOI/10.5281/zenodo.17549670.svg)](https://doi.org/10.5281/zenodo.17549670)  
[![DOI v2: EvoBase Framework](https://zenodo.org/badge/DOI/10.5281/zenodo.17555914.svg)](https://doi.org/10.5281/zenodo.17555914)

## One-Click Demo

```bash
pip install -r requirements.txt
python RUN_ME.py
```

> **Output**: Daily interaction → VAMS filtering → Sleep consolidation → SQIA check → LoRA correction → State saved

### 🇨🇳 中国境内用户专属方案
**所有模型均通过魔搭（ModelScope）镜像下载，无需科学上网**：
```bash
USE_MODELSCOPE=true python RUN_ME.py  # Linux/Mac
# 或
set USE_MODELSCOPE=true && python RUN_ME.py  # Windows
```
> ✅ 已验证可在无外网环境下运行  
> ✅ 覆盖 Qwen 模型 + Sentence-BERT 嵌入模型

---

## Core Closed Loop

```
User Interaction 
    → VAMS Scoring (Value-Aligned Memory Scoring) 
    → Vector Memory Store 
    → Sleep Distiller (Nightly Consolidation) 
    → SQIA Self-Check (Identity Anchoring) 
    → LoRA Replacement 
    → New Model Version
```

---

## Key Mechanisms

| Component | Formula / Logic |
|---------|-----------------|
| **VAMS** | `Score = R × (0.4 + 0.3E + 0.3V)` <br> `R`: keyword-based relevance, `E`: TextBlob sentiment polarity, `V`: value-aligned keyword matching |
| **SQIA** | `Drift = 1 - cos(Gen(q), Truth(q))` <br> Uses Sentence-BERT embeddings; `if Drift > 0.15 → LoRA correction` |
| **Sleep** | Keep top-k memories by VAMS score |
| **LoRA** | PEFT-based adapter on `q_proj/v_proj` layers; fine-tuned during SQIA correction |

---

## Citation

### For the **academic paper** (EvolvingAgent-Zero):
```bibtex
@misc{wang2025evolvingagent,
  author       = {Wang, Zhongren},
  title        = {EvolvingAgent-Zero: A Self-Evolving Framework for Continuous Personalization in Small-Scale Language Models},
  year         = 2025,
  month        = nov,
  doi          = {10.5281/zenodo.17549670},
  url          = {https://doi.org/10.5281/zenodo.17549670},
  note         = {First academic proposal on November 8, 2025}
}
```

### For the **open-source framework** (EvoBase):
```bibtex
@software{wang2025evobase,
  author       = {Wang, Zhongren},
  title        = {EvoBase Framework: Continuous Self-Evolution via VAMS, SQIA, and LoRA Replacement},
  year         = 2025,
  month        = nov,
  doi          = {10.5281/zenodo.17555914},
  url          = {https://github.com/evobase-ai/core},
  note         = {Open-source implementation (MIT License), first released November 10, 2025}
}
```

---

## Files

- [`EvoBase Framework.pdf`](EvoBase%20Framework.pdf) – Full specification with pseudocode and diagrams  
- [`src/evobase.py`](src/evobase.py) – Core engine (now with real VAMS/SQIA/LoRA)  
- [`RUN_ME.py`](RUN_ME.py) – One-click demo script (supports ModelScope fallback)  
- [`requirements.txt`](requirements.txt) – Dependencies (includes sentence-transformers, textblob, peft, modelscope)  

---

## Run Locally (China Optimized)

```bash
git clone https://github.com/evobase-ai/core.git
cd core
pip install -r requirements.txt
python -m textblob.download_corpora  # Required for VAMS sentiment analysis
```

### 中国境内运行命令：
```bash
# 启用全链路魔搭镜像（无需外网）
USE_MODELSCOPE=true python RUN_ME.py  # Linux/Mac
set USE_MODELSCOPE=true && python RUN_ME.py  # Windows
```

> **Note**: First run will download from **魔搭 ModelScope**:
> - Qwen2-0.5B model (~1.2GB) 
> - Sentence-BERT (`all-MiniLM-L6-v2`, ~80MB)
> **全程使用国内 CDN 加速，无需任何代理**

---

**© 2025 EvoBase Project · Open Source · MIT License**  
**All future works must cite both DOIs to respect priority.**