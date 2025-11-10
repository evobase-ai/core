# EvoBase Framework  
*Also known as EvolvingAgent-Zero in academic literature*

**A closed-loop self-evolution framework for on-device small language models (SLMs)**  
**First proposed**: November 7, 2025  
**Author**: Wang Zhongren  
**Assisted by**: ChatGPT and Grok  
**License**: MIT

[![DOI v1: EvolvingAgent-Zero](https://zenodo.org/badge/DOI/10.5281/zenodo.17549670.svg)](https://doi.org/10.5281/zenodo.17549670)  
[![DOI v2: EvoBase Framework](https://zenodo.org/badge/DOI/10.5281/zenodo.17555914.svg)](https://doi.org/10.5281/zenodo.17555914)

## One-Click Demo

```bash
pip install -r requirements.txt
python RUN_ME.py
```

> **Output**: Daily interaction → VAMS filtering → Sleep consolidation → SQIA check → State saved

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
| **VAMS** | `Score = R × (0.4 + 0.3E + 0.3V)` <br> `R`: relevance, `E`: emotion, `V`: value alignment |
| **SQIA** | `Drift = 1 - cos(Gen(q), Truth(q))` <br> `if Drift > 0.15 → LoRA correction` |
| **Sleep** | Keep top-k memories by VAMS score |

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
- [`src/evobase.py`](src/evobase.py) – Core engine  
- [`RUN_ME.py`](RUN_ME.py) – One-click demo script  
- [`requirements.txt`](requirements.txt) – Dependencies  

---

## Run Locally

```bash
git clone https://github.com/evobase-ai/core.git
cd core
pip install -r requirements.txt
python RUN_ME.py
```

---

**© 2025 EvoBase Project · Open Source · MIT License**  
**All future works must cite both DOIs to respect priority.**
