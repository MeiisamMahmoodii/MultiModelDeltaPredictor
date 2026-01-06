# ISD-CP Research Papers - Quick Links & Annotations

## 🔴 ABSOLUTE MUST READ (10 papers)

### Core Architecture & Fundamentals
1. **Attention Is All You Need** (2017)
   - Link: https://arxiv.org/abs/1706.03762
   - PDF: https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
   - Why: Foundation of everything. Read first.

2. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021)
   - Link: https://arxiv.org/abs/2104.09864
   - Why: Explains RoPE mechanism used for causal graph position encoding

### Causal Discovery Core
3. **DAGs with NO TEARS** (2018)
   - Link: https://arxiv.org/abs/1803.01422
   - PDF: https://proceedings.neurips.cc/paper_files/paper/2018/file/e859cd4305a7c5a9b78b5155b3c3abc5-Paper.pdf
   - Why: The h-function constraint foundation. Zheng's work is seminal.

4. **Causality: Models, Reasoning and Inference** (Book, 2009)
   - Amazon: https://www.amazon.com/Causality-Reasoning-Inference-Judea-Pearl/dp/0521895685
   - Why: Pearl's foundational book. Read Chapters 1-3 minimum.

### Deep Learning Techniques
5. **Categorical Reparameterization with Gumbel-Softmax** (2016)
   - Link: https://arxiv.org/abs/1611.01144
   - Why: Explains hard/soft discrete sampling in expert routing

6. **Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer** (2017)
   - Link: https://arxiv.org/abs/1701.06538
   - Why: MoE architecture foundation with modifications

7. **GLU Variants Improve Transformer** (2020)
   - Link: https://arxiv.org/abs/2002.05202
   - Why: SwiGLU activation in expert blocks

### Physics & Embeddings
8. **Fourier Features Let Networks Learn High Frequency Functions** (2020)
   - Link: https://arxiv.org/abs/2006.10739
   - PDF: https://proceedings.neurips.cc/paper_files/paper/2020/file/55651da9bcc6fa3b045290333f3a14d3-Paper.pdf
   - Why: Hybrid embedding strategy (Linear + Fourier + MLP)

### Causal Inference & Learning
9. **Elements of Causal Inference** (Book, 2017)
   - Amazon: https://www.amazon.com/Elements-Causal-Inference-Foundations-Learning/dp/0262037319
   - MIT Website: https://mitpress.mit.edu/9780262037319/elements-of-causal-inference/
   - Why: Twin-world concept comes from interventional data handling

10. **Learning Sparse Nonparametric DAGs (NOTEARS-MLP)** (2020)
    - Link: https://arxiv.org/abs/1909.13189
    - Why: Primary baseline for non-linear causal discovery

---

## 🟡 STRONGLY RECOMMENDED (17 papers)

### Optimization & Training
11. **Decoupled Weight Decay Regularization (AdamW)** (2017)
    - Link: https://arxiv.org/abs/1711.05101
    - PDF: https://openreview.net/pdf?id=Syx4wnEtvH
    - Used in: main.py training loop

12. **SGDR: Stochastic Gradient Descent with Warm Restarts** (2016)
    - Link: https://arxiv.org/abs/1608.03983
    - Used in: Learning rate scheduling

13. **Curriculum Learning** (2009)
    - Link: https://dl.acm.org/doi/10.1145/1553374.1553380
    - PDF: https://ronan.collobert.com/pub/matos/2009_curriculum_learning.pdf
    - Used in: curriculum.py multi-dimensional scheduling

14. **Training Deep Nets with Sublinear Memory Cost** (2016)
    - Link: https://arxiv.org/abs/1604.06174
    - Used in: Gradient checkpointing for large models

### Causal Discovery Baselines
15. **DAG-GNN: DAG Structure Learning with Graph Neural Networks** (2019)
    - Link: https://arxiv.org/abs/1904.10098
    - Conference: ICML 2019
    - Status: Implemented in wrappers.py for comparison

16. **Gradient-Based Neural DAG Learning** (2019)
    - Link: https://arxiv.org/abs/1906.02226
    - Why: Alternative gradient-based approach

17. **On the Role of Sparsity and DAG Constraints (GOLEM)** (2020)
    - Link: https://arxiv.org/abs/2006.10201
    - Conference: NeurIPS 2020
    - Why: Linear baseline for structure learning

### Normalization & Architecture
18. **Root Mean Square Layer Normalization** (2019)
    - Link: https://arxiv.org/abs/1910.07468
    - Conference: NeurIPS 2019
    - Used in: Expert block normalization

19. **Layer Normalization** (2016)
    - Link: https://arxiv.org/abs/1607.06450
    - Used in: Attention layer normalization

### Discrete Sampling (Related)
20. **The Concrete Distribution** (2016)
    - Link: https://arxiv.org/abs/1611.00712
    - Why: Related to Gumbel-Softmax, alternative formulation

### Classical Causal Discovery
21. **Causation, Prediction, and Search** (Book, 2000)
    - Amazon: https://www.amazon.com/Causation-Prediction-Search-Peter-Spirtes/dp/0262194545
    - Why: PC algorithm foundation (constraint-based methods)

22. **Optimal Structure Identification with Greedy Search (GES)** (2002)
    - Link: https://jmlr.org/papers/v3/chickering02a.html
    - Why: Score-based classical baseline

23. **The Max-Min Hill-Climbing Bayesian Network** (2006)
    - Link: https://link.springer.com/article/10.1007/s10994-006-6889-7
    - Why: SHD metric standard reference

### Modern Neural Methods
24. **Semi-Supervised Classification with GCN** (2016)
    - Link: https://arxiv.org/abs/1609.02907
    - Conference: ICLR 2017
    - Why: Foundation for graph neural networks (DAG-GNN uses GCN)

25. **Neural Ordinary Differential Equations** (2018)
    - Link: https://arxiv.org/abs/1806.07522
    - Conference: NeurIPS 2018
    - Why: Alternative dynamics learning approach

26. **Physics-Informed Neural Networks** (2019)
    - Link: https://doi.org/10.1016/j.jcp.2018.10.045
    - Journal: Journal of Computational Physics
    - Why: Physics-aware learning comparison point

27. **AVICI: Variational Autoencoder for Causal Inference** (2021)
    - Link: https://arxiv.org/abs/2106.07635
    - Status: Implemented in wrappers.py as baseline

---

## 🟢 NOVEL INTEGRATION CANDIDATES (9 papers)

### Advanced Structure Learning
**28. Masked Autoencoder Density Estimation** (2023)
   - Link: https://arxiv.org/abs/2301.13867
   - Idea: Use masked learning for progressive edge discovery

**29. Transformers for Multivariate Time Series Forecasting** (2022)
   - Link: https://arxiv.org/abs/2205.13504
   - Idea: Advanced sequential prediction for delta learning

### Interventional Data & Confounders
**30. Causal Inference with Hidden Variables via GAMM** (2023)
   - Link: https://arxiv.org/abs/2306.08821
   - Idea: Handle latent confounders with interventional data

### Scalability Breakthroughs
**31. Efficient Neural Causal Discovery without Acyclicity Constraints** (2023)
   - Link: https://arxiv.org/abs/2309.11821
   - Idea: Relax DAG constraints for better scaling

### Uncertainty Quantification
**32. Bayesian Causal Forests for Individual Treatment Effects** (2019)
   - Link: https://arxiv.org/abs/1904.10320
   - Idea: Add uncertainty estimates to delta predictions

### Advanced Graph Learning
**33. Learning to Simulate Complex Physics with Graph Networks** (2021)
   - Link: https://arxiv.org/abs/2002.09405
   - Idea: Iterative graph refinement via message-passing

### Mixed Discrete-Continuous Systems
**34. Differentiable Causal Discovery from Interventional Data** (2021)
   - Idea: Extend to mixed discrete-continuous structures

### Few-Shot & Transfer Learning
**35. Meta-Learning for Causal Structure Learning** (2022+)
   - Idea: Few-shot causal discovery via meta-learning

### Real-World Benchmarks
**36. Causal Protein-Signaling Networks (Sachs Dataset)** (2005)
   - Link: https://doi.org/10.1126/science.1105809
   - Status: Standard real-world benchmark not yet tested

### Efficient Expert Systems
**37. Switch Transformers: Scaling to Trillion Parameters** (2021)
   - Link: https://arxiv.org/abs/2101.03961
   - Conference: ICLR 2021
   - Idea: Advanced sparse routing for massive expert pools

---

## 📚 BY READING ORDER (Recommended Sequence)

### Week 1: Foundations
- [ ] Vaswani et al. (2017) - Attention Is All You Need
- [ ] Pearl - Causality (Book, Chapters 1-3)
- [ ] Zheng et al. (2018) - DAGs with NO TEARS

### Week 2: ISD-CP Core
- [ ] Su et al. (2021) - RoPE
- [ ] Jang et al. (2016) - Gumbel-Softmax
- [ ] Shazeer et al. (2017) - MoE
- [ ] Tancik et al. (2020) - Fourier Features

### Week 3: Extensions & Optimization
- [ ] Shazeer (2020) - GLU Variants
- [ ] Loshchilov & Hutter (2017) - AdamW
- [ ] Loshchilov & Hutter (2016) - SGDR
- [ ] Bengio et al. (2009) - Curriculum Learning

### Week 4: Baselines & Comparisons
- [ ] Zheng et al. (2020) - NOTEARS-MLP
- [ ] Yu et al. (2019) - DAG-GNN
- [ ] Peters et al. (2017) - Elements of Causal Inference
- [ ] Lachapelle et al. (2019) - GraN-DAG

### Week 5+: Advanced Topics
- [ ] Choose from Novel Integration papers (28-37)
- [ ] Real-world benchmarks (Sachs Dataset)
- [ ] Implementation of baselines

---

## 🎓 BY TOPIC CLUSTERING

### Transformers & Attention (3 papers)
1. Vaswani et al. (2017) - Attention
2. Su et al. (2021) - RoPE
3. Dosovitskiy et al. (2020) - ViT (mentioned, not core)

### Mixture of Experts (4 papers)
1. Shazeer et al. (2017) - Sparsely-Gated MoE
2. Jang et al. (2016) - Gumbel-Softmax (routing)
3. Maddison et al. (2016) - Concrete Distribution (related)
4. Lepikhin et al. (2021) - Switch Transformers (advanced)

### Causal Discovery (8 papers)
1. Pearl (2009) - Causality (foundational book)
2. Spirtes et al. (2000) - Causation, Prediction, Search (classical)
3. Zheng et al. (2018) - DAGs with NO TEARS (optimization)
4. Zheng et al. (2020) - NOTEARS-MLP (non-linear)
5. Yu et al. (2019) - DAG-GNN (neural)
6. Lachapelle et al. (2019) - GraN-DAG (gradient-based)
7. Lorch et al. (2021) - AVICI (variational)
8. Chickering (2002) - GES (score-based)

### Optimization & Learning (5 papers)
1. Loshchilov & Hutter (2017) - AdamW
2. Loshchilov & Hutter (2016) - SGDR
3. Bengio et al. (2009) - Curriculum Learning
4. Chen et al. (2016) - Gradient Checkpointing
5. Ba et al. (2016) - Layer Normalization

### Physics & Embeddings (3 papers)
1. Tancik et al. (2020) - Fourier Features
2. Raissi et al. (2019) - Physics-Informed NNs
3. Chen et al. (2018) - Neural ODEs

### Interventional Data (2 papers)
1. Peters et al. (2017) - Elements of Causal Inference
2. Eberhardt & Scheines (2007) - Interventions and Causal Inference

### Normalization & Activations (3 papers)
1. Ba et al. (2016) - Layer Normalization
2. Zhang & Sennrich (2019) - RMSNorm
3. Shazeer (2020) - GLU Variants

### Benchmarks & Datasets (3 papers)
1. Sachs et al. (2005) - Sachs Dataset (real-world)
2. Lauritzen & Spiegelhalter (1988) - Asia Network
3. Beinlich et al. (1989) - ALARM Network

---

## 🔗 AUTHOR TRACKING (For Citations & Updates)

### Most Cited Authors in Project
1. **Zheng, Aragam, Ravikumar** - NOTEARS family
2. **Shazeer** - MoE and GLU
3. **Su et al.** - RoPE
4. **Tancik et al.** - Fourier Features
5. **Pearl, Spirtes** - Causal theory founders
6. **Jang, Gu, Poole** - Gumbel-Softmax
7. **Loshchilov & Hutter** - Optimization

**Google Scholar Search Strategy**:
- Search "Zheng causal discovery" to find NOTEARS variants
- Search "Shazeer mixture of experts" for MoE improvements
- Search "causal discovery 2023" for latest methods

---

## 📊 PAPER DEPENDENCY GRAPH

```
Pearl (2009) Causality
    ↓
Zheng et al. (2018) NOTEARS
    ├→ Zheng et al. (2020) NOTEARS-MLP ← ISD-CP PRIMARY BASELINE
    └→ Yu et al. (2019) DAG-GNN

Vaswani et al. (2017) Attention
    ↓
Su et al. (2021) RoPE ← USED IN ISD-CP
    ↓
ISD-CP: Causal Transformer

Jang et al. (2016) Gumbel-Softmax ← USED IN ISD-CP
    ↓
Shazeer et al. (2017) MoE ← USED IN ISD-CP
    ↓
Hard routing in ISD-CP

Tancik et al. (2020) Fourier Features ← USED IN ISD-CP
    ↓
Hybrid embeddings in ISD-CP

Peters et al. (2017) Elements of Causal Inference
    ↓
Twin-world concept in ISD-CP
```

---

## 🎯 EXAM-STYLE QUESTIONS (To test understanding)

1. **Why does ISD-CP use RoPE instead of absolute position embeddings?**
   - Answer: Causal graphs have no inherent sequential order. RoPE encodes relative distances.

2. **How does hard Gumbel-Softmax (k=1) differ from soft MoE?**
   - Answer: Forces exactly one expert per token, preventing expert collapse.

3. **What is the h-function in NOTEARS?**
   - Answer: h(A) = tr(e^(A⊙A)) - d enforces acyclicity through matrix exponential.

4. **How does twin-world variance reduction work?**
   - Answer: Same noise in observational and interventional data enables Δ = f(X|do) - f(X|obs).

5. **What are the three main novel contributions of ISD-CP?**
   - Answer: (1) Unified structure+function learning, (2) Interleaved token encoding, (3) Hard MoE for physics.

---

## ⚠️ COMMONLY CONFUSED PAPERS

| Confusion | Resolution |
|-----------|-----------|
| NOTEARS vs NOTEARS-MLP | NOTEARS is linear; NOTEARS-MLP handles non-linear functions |
| Gumbel-Softmax vs Concrete | Same idea, different names (Gumbel more common) |
| DAGs with NO TEARS vs GraN-DAG | Same h-function, different optimization (Lagrangian vs direct) |
| RoPE vs Absolute Positional | RoPE uses rotation; Absolute adds position embeddings |
| SHD vs SID vs F1 | SHD = edit distance; SID = intervention equivalence; F1 = precision-recall |

---

Generated: January 6, 2026  
Total Links: 37+ papers with direct arXiv/URL references  
Reading Time Estimate: 40-60 hours for complete coverage  
