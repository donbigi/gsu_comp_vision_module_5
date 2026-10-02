# Validation — Feasibility Prototype

**What works**
- Local `Qwen2.5-VL-3B` · open weights · no external API
- Clip → structured verdict: `SUCCESS` / `FAILURE` + corrective sub-goal

**Real robot clip (ALOHA coffee)**
| Sub-goal | Verdict |
| --- | --- |
| "Grasp cup from machine" | ✅ `SUCCESS` |
| "Place tomato slice" (mismatch) | ❌ `FAILURE` — flagged |

**Caveats**
- Latency: minutes on CPU — target interactive ⇒ GPU/MPS
- Controller recovery loop: spec only, not yet executed

**Run it yourself** — https://github.com/<your-org>/<repo>
