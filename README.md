# EvoBase Framework

**Continuous Self-Evolution of Small Language Models via VAMS, SQIA, and LoRA Replacement**

**First proposed: November 7, 2025**  
**Author**: Wang Zhongren  
**Assisted by**: ChatGPT and Grok  
**License**: MIT

---

## Overview

EvoBase enables **on-device small language models (SLMs)** to **self-evolve continuously** through a closed loop:

```
User Interaction → VAMS Scoring → Vector Memory → Sleep Distillation → SQIA Check → LoRA Replacement → New Model
```

- **VAMS**: Value-Aligned Memory Scoring (relevance + emotion + values)  
- **SQIA**: Self-Query Identity Anchoring (prevents catastrophic forgetting)  
- **LoRA Replacement**: Daily incremental fine-tuning with identity preservation

---

## Key Components

| Component | Description |
|-----------|-------------|
| **VAMS**  | `Score = R × (0.4 + 0.3E + 0.3V)` → filters memories |
| **SQIA**  | Daily self-questioning: `if Drift > 0.15 → LoRA correction` |
| **Sleep Distiller** | Nightly memory consolidation (vector store compression) |

---

## Citation

```bibtex
@misc{wang2025evobase,
  title = {{EvoBase Framework}: Continuous Self-Evolution of Small Language Models via {VAMS}, {SQIA}, and {LoRA} Replacement},
  author = {Wang, Zhongren},
  year = {2025},
  month = {nov},
  howpublished = {\url{https://github.com/evobase-ai/core}},
  note = {First proposed on November 7, 2025. Assisted by ChatGPT and Grok. MIT License.}
}
```

---

## Files

- [EvoBase Framework.pdf](EvoBase%20Framework.pdf) _(Full specification with pseudocode)_
- [Daily Identity Check Pseudocode](src/daily_check.py)

---

© 2025 EvoBase Project · Open Source · MIT License