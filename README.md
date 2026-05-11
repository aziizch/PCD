## 🚀 Enhanced Pipeline (CodeSensei / PCD-2)

This fork introduces a **Hybrid Reranker** that extends the original Stage 2 
of the R4 pipeline with three complementary signals:

| Signal | Description |
|--------|-------------|
| LLM Relevance Score | Qwen-32B scoring with uncertainty penalty (ε=0.3) |
| Graph Proximity | BFS distance to nearest FUNCTION node |
| Heuristic Rules | Filename-based penalties/boosts |

### Results on SWE-bench Lite (223 instances)

| Metric | Baseline | Ours | Gain |
|--------|----------|------|------|
| Stage 2 Recall | 83.65% | 87.44% | +4.53% |
| Precision | 64.13% | 67.26% | +4.87% |
| MRR | 73.54% | 75.62% | +2.83% |
| Resolution Rate | 20.61% | 21.06% | +2.18% |

### How to Use the Hybrid Reranker

\```python
from hybrid_reranker.hybrid_scorer import HybridReranker

reranker = HybridReranker(alpha=0.75, beta=0.20, delta=0.05, epsilon=0.3)
ranked_files = reranker.score(stage1_shortlist, graph, issue)
\```