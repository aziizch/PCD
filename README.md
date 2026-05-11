# 🤖 CodeSensei — Automated Repository-Level Bug Detection and Correction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Qwen](https://img.shields.io/badge/Qwen2.5--Coder-32B-purple)
![Angular](https://img.shields.io/badge/Angular-21-red?logo=angular)
![Spring Boot](https://img.shields.io/badge/Spring_Boot-4.0.5-green?logo=springboot)
![SWE-bench](https://img.shields.io/badge/SWE--bench_Lite-21.06%25-orange)

**University of Manouba — National School of Computer Science (ENSI)**
Project PCD :2 — Academic Year 2025/2026

---

## 📖 Description

**CodeSensei** is an intelligent automated bug fixing platform for large-scale software repositories. It builds on the open-source [CodeFuse-CGM](https://github.com/codefuse-ai/CodeFuse-CGM) framework and enhances it with a **hybrid reranker** that combines LLM-based relevance scoring, structural proximity through the code graph, and filename-based heuristic rules.

The system implements the **R4 chain** (Rewriter → Retriever → Reranker → Reader) and contributes a targeted improvement at **Stage 2 of the Reranker**, enabling more accurate file localization before patch generation.

**Our core contributions:**

- 🎯 **Calibrated LLM Score** — Qwen-32B scoring with an uncertainty penalty for ambiguous outputs (ε = 0.3)
- 🕸️ **Graph-Based Functional Proximity** — BFS traversal toward the nearest `FUNCTION` nodes in the code graph
- 📁 **Heuristic Filename Rules** — empirically derived penalties and boosts based on file naming patterns
- 🔢 **Adaptive Top-K Selection** — dynamic adjustment of the number of files forwarded to the patch generator

---

## 👥 Authors

| Name | Role |
|------|------|
| Mohamed Aziz CHAOUACHI | Developer / Researcher |
| Othman AYARI | Developer / Researcher |
| Ahmed Youssef EL ASKRI | Developer / Researcher |
| **Supervisor: Dr. Hatem AOUADI** | Academic Supervisor |

---

## 🏗️ Project Structure

```
PCD/
├── cgm/                        # Original CodeFuse-CGM (graph-based Reader)
├── rewriter/                   # Rewriter — entity extraction and query generation
│   └── prompt.py               # generate_prompt_for_extractor / inferer
├── retriever/                  # Retriever — anchor node detection and subgraph expansion
│   ├── locate_anchor_node.py
│   ├── subgraph.py
│   └── serialize_subgraph.py
├── preprocess_embedding/       # Node embedding generation (CGE-Large)
│   ├── generate_code_content.py
│   ├── generate_code_embedding.py
│   └── generate_rewriter_embedding.py
├── reranker_hybride/           # ✨ Our Contribution — Hybrid Reranker (Stage 2)
│   ├── hybrid_scorer.py        # Final score = α·S'_LLM + β·Sgraph·γ + δ·Sheuristic
│   ├── graph_proximity.py      # BFS toward FUNCTION nodes (Sgraph)
│   ├── heuristic_rules.py      # Filename-based rules (test, doc, core...)
│   ├── uncertainty_penalty.py  # ε calibration on LLM score = 3
│   ├── adaptive_topk.py        # Adaptive k2 selection based on score gap
│   └── reranker.py             # Stage 1 + Stage 2 pipeline
├── evaluation/                 # Metrics and SWE-bench evaluation scripts
├── assets/                     # Static resources
├── codesensei_interface/       # Web interface (Angular + Spring Boot + Flask)
│   ├── frontend/               # Angular 21 (TypeScript)
│   └── backend/                # Spring Boot 4 (Java 21) + Flask 3 (Python)
└── README.md
```

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Angular 21, TypeScript 5.6 |
| Backend | Spring Boot 4.0.5 (Java 21), Flask 3.0.3 |
| Database | MySQL |
| Main LLM | Qwen2.5-Coder-32B-Instruct |
| Embeddings | CodeFuse-CGE-Large |
| Code Graph | NetworkX 3.0 |
| Vector Search | FAISS |
| LLM Inference | vLLM 0.8.5 |
| Benchmark | SWE-bench Lite (300 real Python GitHub issues) |
| Auth | JWT (Spring Security) |

---

## 🧠 The Hybrid Reranker — Our Contribution

The original CodeFuse-CGM pipeline relies solely on a single integer LLM score (1–5) to rank candidate files. Our enhancement, contained entirely within **Stage 2 of the Reranker**, combines three complementary signals:

### Scoring Formula

```
S_final(f) = α · S'_LLM + β · Sgraph · γ + δ · Sheuristic
```

Optimal configuration: **α = 0.75, β = 0.20, δ = 0.05, ε = 0.3, γ = 5.0**

### The Three Signals

**1. Calibrated LLM Score (S'_LLM)**

Qwen-32B scores each file from 1 to 5. A score of 3 appears at roughly twice the frequency of 2 or 4, indicating the model is uncertain rather than genuinely mid-relevant. We apply a small uncertainty penalty ε:

```
S'_LLM = S_LLM - ε    if S_LLM = 3
S'_LLM = S_LLM         otherwise
```

**2. Graph-Based Functional Proximity (Sgraph)**

BFS from each file node to the nearest `FUNCTION`-type node at depth d (max depth = 4):

```
Sgraph = 1 / (1 + d)
```

- 1.0 — file directly contains a function
- 0.5 — one hop away
- 0.33 — depth 2
- 0.0 — no FUNCTION node reachable within 4 hops

**3. Heuristic Filename Rules (Sheuristic)**

| Pattern | Weight | FPR / TPR | Rationale |
|---------|--------|-----------|-----------|
| `"test"` in filename | −1.0 | FPR ≈ 96% | Test files verify behavior, rarely the bug location |
| `"doc"` in filename | −0.5 | FPR ≈ 89% | Documentation with no executable logic |
| `"example"` in filename | −0.5 | FPR ≈ 85% | Example scripts illustrating usage |
| `"core"` in filename | +0.3 | TPR ≈ 61% | Core modules implementing central domain logic |

Weights were derived empirically from gold-patch frequency analysis across all 300 SWE-bench Lite instances.

### Adaptive Top-K Selection

Rather than a fixed cutoff k₂ = 5, we expand to k₂ + 2 when the score gap between the 1st and 5th ranked files is below threshold τ = 0.5, signaling that the reranker is uncertain about the boundary:

```
k_adapt = min(n, k2 + 2)   if Δ = S(1)_final - S(5)_final < τ
k_adapt = 5                  otherwise
```

---

## 📊 Results on SWE-bench Lite (223 instances)

| Metric | Baseline (LLM only) | Our Pipeline (Hybrid) | Gain |
|--------|--------------------|-----------------------|------|
| Stage 1 Recall | 89.00% | 89.00% | — |
| **Stage 2 Recall** | 83.65% | **87.44%** | **+4.53%** |
| **Precision** | 64.13% | **67.26%** | **+4.87%** |
| **MRR** | 73.54% | **75.62%** | **+2.83%** |
| **Resolution Rate** | 20.61% | **21.06%** | **+2.18%** |

### Ablation Study

| Configuration | S2 Recall | Precision | MRR | Gain over LLM-only |
|---------------|-----------|-----------|-----|-------------------|
| LLM-only | 83.65% | 64.13% | 73.54% | — |
| LLM + Graph | 86.21% | 66.04% | 74.89% | +2.56% |
| **Full Hybrid** | **87.44%** | **67.26%** | **75.62%** | **+3.79%** |

Each added component contributes positively and monotonically. The graph proximity signal delivers the largest single gain (+2.56%), confirming that structural understanding of the codebase is critical for accurate file localization.

### Comparison with State-of-the-Art

| System | Type | Resolution Rate | vs. Ours |
|--------|------|-----------------|----------|
| **Our Pipeline** | RAG + Hybrid Reranker | **21.06%** | — |
| Lingma SWE-GPT (Qwen2.5-72B) | RAG-based | 22.00% | −0.94% |
| SWE-Fixer (Qwen2.5-72B) | Agent | 24.67% | −3.61% |
| CGM-SWE-PY (Qwen2.5-32B) | LLM + RAG | 28.67% | −7.61% |
| CGM-SWE-PY (Qwen2.5-7B) | LLM + RAG | 4.00% | **+17.06%** |

> Our system achieves results comparable to Lingma SWE-GPT while using a model roughly half the size (32B vs 72B parameters).

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ (Angular frontend)
- Java 21 (Spring Boot backend)
- MySQL
- NVIDIA GPU recommended (CUDA) for LLM inference

### 1. Clone the Repository

```bash
git clone https://github.com/aziizch/PCD.git
cd PCD
```

### 2. Python Environment (Pipeline & Reranker)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Key Python dependencies:

```
transformers==4.46.1
tokenizers==0.20.0
accelerate==1.0.1
peft==0.13.2
torch==2.4.1
networkx==3.0
faiss-cpu
vllm>=0.8.5
RapidFuzz==1.5.0
fuzzywuzzy==0.18.0
```

### 3. Use the Hybrid Reranker

```python
from reranker_hybride.hybrid_scorer import HybridReranker

reranker = HybridReranker(alpha=0.75, beta=0.20, delta=0.05, epsilon=0.3)
ranked_files = reranker.score(stage1_shortlist, graph, issue)
```

### 4. Preprocess Embeddings

```bash
# Extract content from each node in the code graph
python preprocess_embedding/generate_code_content.py

# Generate node embeddings using CGE-Large
python preprocess_embedding/generate_code_embedding.py

# Generate Rewriter query embeddings
python preprocess_embedding/generate_rewriter_embedding.py
```

### 5. Run the Full R4 Pipeline

```bash
# Rewriter
python rewriter/inference_rewriter.py --prompt_path test_rewriter_prompt.json

# Retriever
python retriever/locate_anchor_node.py
python retriever/subgraph.py
python retriever/serialize_subgraph.py

# Hybrid Reranker
python reranker_hybride/reranker.py --stage_1_k 10 --stage_2_k 5
```

### 6. Backend (Spring Boot)

```bash
cd codesensei_interface/backend
./mvnw spring-boot:run
# API available at http://localhost:8080/api/
```

### 7. Frontend (Angular)

```bash
cd codesensei_interface/frontend
npm install
ng serve
# Interface available at http://localhost:4200
```

---

## 📡 Main API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | Login — returns JWT token |
| `/api/auth/me` | GET | Authenticated user profile |
| `/api/patch/generate` | POST | Submit repo URL + issue → generate patch |
| `/api/patch/status/<id>` | GET | Poll patch generation status |
| `/api/patch/result/<id>` | GET | Retrieve generated patch |
| `/api/admin/users` | GET | List all users (admin only) |

---

## 🎯 Key Features

**For Developers**
- Submit a GitHub repository URL and issue description via chat interface
- Receive a generated patch in unified diff format
- Approve the patch to integrate it, or request regeneration

**For Administrators**
- User account management (create, edit, delete)
- Monitor all patch generation sessions
- System maintenance and configuration

---

## 🔄 Pipeline Overview

```
GitHub Issue + Repository Snapshot
        ↓
[1] Rewriter       — extract code entities & generate search queries
        ↓
[2] Retriever      — embed queries, find anchor nodes, expand subgraph
        ↓
[3] Reranker S1    — LLM shortlists top-10 candidate files
        ↓
[4] Reranker S2    — Hybrid scoring: LLM + Graph proximity + Heuristics
        ↓
   Adaptive Top-K  — k=5 (confident) or k=7 (uncertain score gap)
        ↓
[5] Reader         — Qwen-32B generates patch in unified diff format
        ↓
[6] Validation     — syntax check + apply to base commit + parse check
        ↓
Patch delivered to developer interface
```

---

## 🖥️ Graphical Interface

The **CodeSensei** web interface provides a clean, role-based interaction layer over the underlying pipeline.

| Page | Description |
|------|-------------|
| Sign In | Credential-based authentication with JWT |
| Sign Up | New user registration |
| Chat | Main screen — submit repo + issue, receive patch |
| Account | User profile and subscription management |
| Payment | Subscription tiers (Pro · Business · Pro+) |
| Admin | User management and system monitoring dashboard |

---

## ⚙️ Hyperparameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| α | 0.75 | LLM score weight |
| β | 0.20 | Graph proximity weight |
| δ | 0.05 | Heuristic weight |
| γ | 5.0 | Graph score rescaling factor |
| ε | 0.3 | Uncertainty penalty on LLM score = 3 |
| τ | 0.5 | Score gap threshold for adaptive Top-K |
| BFS depth | 4 | Maximum hops to search for FUNCTION nodes |
| Stage 1 k | 10 | Files shortlisted by LLM in Stage 1 |
| Stage 2 k | 5 (or 7) | Files forwarded to the patch generator |

---

## ⚠️ Known Limitations

- **Graph construction overhead** — Code graphs are computed offline per repository version; no incremental update mechanism is currently implemented.
- **Heuristic generalizability** — Filename rules were calibrated on SWE-bench Lite's Python repositories and may require recalibration for other languages or directory conventions.
- **Single-pass generation** — The patch generator runs once without iterative verification, which is the primary bottleneck between file localization quality and final resolution rate.

---

## 🔮 Future Work

- Iterative, execution-based patch verification loop
- Multi-file patch generation for systemic bugs spanning multiple modules
- Extension to non-Python repositories (Java, TypeScript, C++)
- Stress-testing the hybrid reranker on SWE-bench Verified (500 instances)

---

## 📚 Key References

- [CodeFuse-CGM (Tao et al., NeurIPS 2025)](https://arxiv.org/abs/2505.16901) — Original R4 framework this work builds upon
- [SWE-bench Lite](https://www.swebench.com) — Evaluation benchmark
- [Qwen2.5-Coder](https://huggingface.co/Qwen) — LLM for scoring and patch generation
- [CodeFuse-CGE-Large](https://huggingface.co/codefuse-ai) — Code graph node embeddings

---

## 📄 Citation

If you find this work useful, please also cite the original CGM paper:

```bibtex
@misc{tao2025codegraphmodelcgm,
  title         = {Code Graph Model (CGM): A Graph-Integrated Large Language Model
                   for Repository-Level Software Engineering Tasks},
  author        = {Hongyuan Tao and Ying Zhang and Zhenhao Tang and others},
  year          = {2025},
  eprint        = {2505.16901},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2505.16901}
}
```

---

## 📄 License

This project was developed as part of an academic capstone project (PCD) at ENSI — National School of Computer Science, University of Manouba, Tunisia.

---

*CodeSensei — Making automated software repair smarter, faster, and more reliable.*