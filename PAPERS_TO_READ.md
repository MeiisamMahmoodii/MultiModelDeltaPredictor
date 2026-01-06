# Essential Papers for Understanding ISD-CP Project

**Last Updated**: January 6, 2026  
**Project**: Multi-Model Delta Predictor (ISD-CP)  
**Classification**: Must Read | Better to Read | Novel Integration

---

## 📚 MUST READ PAPERS (Core Understanding)

### 1. **Attention is All You Need**
- **Authors**: Vaswani, A., Shazeer, N., Parmar, N., et al.
- **Year**: 2017
- **Conference**: NeurIPS
- **Link**: [arxiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Why Essential**: Foundation of the transformer architecture used in ISD-CP
- **Key Concepts**: Self-attention, multi-head attention, scaled dot-product attention
- **Project Usage**: `src/models/CausalTransformer.py` - custom attention implementation with RoPE

---

### 2. **RoFormer: Enhanced Transformer with Rotary Position Embedding**
- **Authors**: Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y.
- **Year**: 2021
- **Preprint**: arXiv:2104.09864
- **Link**: [arxiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Why Essential**: ISD-CP's relative position encoding mechanism (RoPE)
- **Key Concepts**: Rotary positional embeddings, relative distance encoding
- **Project Implementation**: `src/models/rope.py`
- **Critical for**: Understanding why relative positions matter in causal graphs

---

### 3. **DAGs with NO TEARS: Continuous Optimization for Structure Learning**
- **Authors**: Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P.
- **Year**: 2018
- **Conference**: NeurIPS
- **Link**: [arxiv:1803.01422](https://arxiv.org/abs/1803.01422)
- **Why Essential**: Foundational causal discovery method; ISD-CP uses the h-function constraint
- **Key Concepts**: DAG constraint via matrix exponential, acyclicity, h-function
- **Project Usage**: `src/training/loss.py` - h-function implementation (lines 9-62)
- **Project Modification**: Original uses augmented Lagrangian; ISD-CP uses weighted loss term

---

### 4. **Categorical Reparameterization with Gumbel-Softmax**
- **Authors**: Jang, E., Gu, S., & Poole, B.
- **Year**: 2016
- **Preprint**: arXiv:1611.01144
- **Link**: [arxiv:1611.01144](https://arxiv.org/abs/1611.01144)
- **Why Essential**: ISD-CP uses Hard Gumbel routing for Mixture of Experts
- **Key Concepts**: Continuous relaxation of discrete distributions, temperature annealing
- **Project Usage**: `src/models/CausalTransformer.py` (line 122) - hard=True for discrete expert selection
- **Project Modification**: Hard mode (k=1) instead of soft routing for specialization

---

### 5. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture of Experts Layer**
- **Authors**: Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J.
- **Year**: 2017
- **Preprint**: arXiv:1701.06538
- **Link**: [arxiv:1701.06538](https://arxiv.org/abs/1701.06538)
- **Why Essential**: MoE architecture used for physics mechanism specialization
- **Key Concepts**: Expert routing, load balancing, sparse activation
- **Project Usage**: `src/models/CausalTransformer.py` - vectorized MoE with Hard Gumbel routing
- **Project Modification**: Hard routing (k=1), no load balancing loss, vectorized execution via torch.einsum

---

### 6. **Causality: Models, Reasoning and Inference** (2nd Edition)
- **Author**: Pearl, J.
- **Year**: 2009
- **Publisher**: Cambridge University Press
- **Why Essential**: Foundational book on causal models, SCMs, interventions
- **Key Concepts**: Structural Causal Models, do-calculus, causal graphs, interventions
- **Critical for**: Understanding what ISD-CP is learning (both structure AND dynamics)

---

### 7. **Learning Sparse Nonparametric DAGs**
- **Authors**: Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E.
- **Year**: 2020
- **Conference**: AISTATS
- **Link**: [arxiv:1909.13189](https://arxiv.org/abs/1909.13189)
- **Why Essential**: NOTEARS-MLP - primary baseline for non-linear causal discovery
- **Key Concepts**: Non-linear functions, DAG constraints on neural networks
- **Project Relevance**: ISD-CP compares directly against NOTEARS-MLP
- **Key Difference**: ISD-CP amortizes (trains once), NOTEARS-MLP optimizes per-graph

---

### 8. **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains**
- **Authors**: Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., et al.
- **Year**: 2020
- **Conference**: NeurIPS
- **Link**: [arxiv:2006.10739](https://arxiv.org/abs/2006.10739)
- **Why Essential**: ISD-CP's Fourier feature embedding for physics-aware value encoding
- **Key Concepts**: Positional encoding, high-frequency learning, feature transformation
- **Project Usage**: `src/data/encoder.py` (lines 22-56) - hybrid embeddings (Linear + Fourier + MLP)
- **Project Modification**: Fixed frequencies (2^0, 2^1, ..., 2^7) instead of random

---

### 9. **GLU Variants Improve Transformer**
- **Authors**: Shazeer, N.
- **Year**: 2020
- **Preprint**: arXiv:2002.05202
- **Link**: [arxiv:2002.05202](https://arxiv.org/abs/2002.05202)
- **Why Essential**: SwiGLU activation function used in expert blocks
- **Key Concepts**: Gated Linear Units, Swish activation, gradient flow improvement
- **Project Usage**: `src/models/CausalTransformer.py` - SwiGLU in vectorized expert architecture

---

### 10. **Elements of Causal Inference: Foundations and Learning Algorithms**
- **Authors**: Peters, J., Janzing, D., & Schölkopf, B.
- **Year**: 2017
- **Publisher**: MIT Press
- **Why Essential**: Comprehensive treatment of interventional data and causal learning
- **Key Concepts**: Interventions, identifiability, causal models, learning from interventions
- **Critical for**: Understanding twin-world variance reduction technique

---

## 📖 BETTER TO READ (Deeper Understanding)

### 11. **DAG-GNN: DAG Structure Learning with Graph Neural Networks**
- **Authors**: Yu, Y., Chen, J., Gao, T., & Yu, M.
- **Year**: 2019
- **Conference**: ICML
- **Link**: [arxiv:1904.10098](https://arxiv.org/abs/1904.10098)
- **Why Important**: Neural alternative to continuous optimization; main comparison baseline
- **Relevance**: ISD-CP is compared against DAG-GNN in evaluation suite
- **Key Difference**: GNN-based vs Transformer-based approach

---

### 12. **Gradient-Based Neural DAG Learning**
- **Authors**: Lachapelle, S., Brouillard, P., Deleu, T., & Lacoste-Julien, S.
- **Year**: 2019
- **Preprint**: arXiv:1906.02226
- **Link**: [arxiv:1906.02226](https://arxiv.org/abs/1906.02226)
- **Why Important**: Gradient-based DAG learning framework
- **Relevance**: Similar continuous optimization approach to ISD-CP

---

### 13. **On the Role of Sparsity and DAG Constraints for Learning Linear DAGs**
- **Authors**: Ng, I., Ghassami, A., & Zhang, K.
- **Year**: 2020
- **Conference**: NeurIPS
- **Link**: [arxiv:2006.10201](https://arxiv.org/abs/2006.10201)
- **Why Important**: GOLEM - linear causal discovery baseline
- **Relevance**: Comparison point for structure learning

---

### 14. **Root Mean Square Layer Normalization**
- **Authors**: Zhang, B., & Sennrich, R.
- **Year**: 2019
- **Conference**: NeurIPS
- **Link**: [arxiv:1910.07468](https://arxiv.org/abs/1910.07468)
- **Why Important**: RMSNorm used in expert blocks
- **Relevance**: Normalization efficiency in deep networks
- **Project Usage**: `src/models/CausalTransformer.py` (line 44)

---

### 15. **Layer Normalization**
- **Authors**: Ba, J. L., Kiros, J. R., & Hinton, G. E.
- **Year**: 2016
- **Preprint**: arXiv:1607.06450
- **Link**: [arxiv:1607.06450](https://arxiv.org/abs/1607.06450)
- **Why Important**: Standard normalization in transformer architectures
- **Relevance**: Used in attention layers of ISD-CP

---

### 16. **Decoupled Weight Decay Regularization** (AdamW)
- **Authors**: Loshchilov, I., & Hutter, F.
- **Year**: 2017
- **Conference**: ICLR
- **Link**: [arxiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **Why Important**: Optimizer used for training ISD-CP
- **Project Usage**: `main.py` (line 74) with learning rate 2e-4

---

### 17. **SGDR: Stochastic Gradient Descent with Warm Restarts**
- **Authors**: Loshchilov, I., & Hutter, F.
- **Year**: 2016
- **Conference**: ICLR
- **Link**: [arxiv:1608.03983](https://arxiv.org/abs/1608.03983)
- **Why Important**: Cosine annealing learning rate schedule with warm restarts
- **Project Usage**: `main.py` - T_0=50 epochs, T_mult=2

---

### 18. **Curriculum Learning**
- **Authors**: Bengio, Y., Louradour, J., Collobert, R., & Weston, J.
- **Year**: 2009
- **Conference**: ICML
- **Link**: [https://dl.acm.org/doi/10.1145/1553374.1553380](https://dl.acm.org/doi/10.1145/1553374.1553380)
- **Why Important**: ISD-CP uses multi-dimensional curriculum learning
- **Project Usage**: `src/training/curriculum.py`
- **Key Innovation**: Varies graph size, density, AND intervention strength simultaneously

---

### 19. **Training Deep Nets with Sublinear Memory Cost** (Gradient Checkpointing)
- **Authors**: Chen, T., Xu, B., Zhang, C., & Guestrin, C.
- **Year**: 2016
- **Preprint**: arXiv:1604.06174
- **Link**: [arxiv:1604.06174](https://arxiv.org/abs/1604.06174)
- **Why Important**: Memory-efficient training technique
- **Project Usage**: `src/models/CausalTransformer.py` (line 270)

---

### 20. **The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables**
- **Authors**: Maddison, C. J., Mnih, A., & Teh, Y. W.
- **Year**: 2016
- **Preprint**: arXiv:1611.00712
- **Link**: [arxiv:1611.00712](https://arxiv.org/abs/1611.00712)
- **Why Important**: Related work to Gumbel-Softmax for discrete sampling
- **Relevance**: Expert routing mechanism

---

### 21. **Causation, Prediction, and Search**
- **Authors**: Spirtes, P., Glymour, C. N., Scheines, R., & Heckerman, D.
- **Year**: 2000
- **Publisher**: MIT Press
- **Why Important**: Foundational constraint-based causal discovery (PC algorithm)
- **Relevance**: Baseline comparison method

---

### 22. **Optimal Structure Identification with Greedy Search**
- **Authors**: Chickering, D. M.
- **Year**: 2002
- **Journal**: Journal of Machine Learning Research
- **Link**: [https://jmlr.org/papers/v3/chickering02a.html](https://jmlr.org/papers/v3/chickering02a.html)
- **Why Important**: GES algorithm - classical score-based causal discovery
- **Relevance**: Comparison baseline in evaluation suite

---

### 23. **The Max-Min Hill-Climbing Bayesian Network Structure Learning Algorithm**
- **Authors**: Tsamardinos, I., Brown, L. E., & Aliferis, C. F.
- **Year**: 2006
- **Journal**: Machine Learning
- **Link**: [https://link.springer.com/article/10.1007/s10994-006-6889-7](https://link.springer.com/article/10.1007/s10994-006-6889-7)
- **Why Important**: Standard metric (SHD) for causal discovery evaluation
- **Project Usage**: `src/training/metrics.py`

---

### 24. **Semi-Supervised Classification with Graph Convolutional Networks**
- **Authors**: Kipf, T. N., & Welling, M.
- **Year**: 2016
- **Preprint**: arXiv:1609.02907
- **Link**: [arxiv:1609.02907](https://arxiv.org/abs/1609.02907)
- **Why Important**: Foundation for graph neural network methods (compare with DAG-GNN)
- **Relevance**: GNN baseline comparison

---

### 25. **Neural Ordinary Differential Equations**
- **Authors**: Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K.
- **Year**: 2018
- **Conference**: NeurIPS
- **Link**: [arxiv:1806.07522](https://arxiv.org/abs/1806.07522)
- **Why Important**: Alternative dynamics learning approach
- **Relevance**: Comparison baseline for delta/function prediction

---

### 26. **Physics-Informed Neural Networks: A Deep Learning Framework**
- **Authors**: Raissi, M., Perdikaris, P., & Karniadakis, G. E.
- **Year**: 2019
- **Journal**: Journal of Computational Physics
- **Link**: [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- **Why Important**: Physics-informed learning; relevant for physics prediction task
- **Relevance**: Comparison for delta prediction accuracy

---

### 27. **AVICI: Variational Autoencoder for Causal Inference**
- **Authors**: Lorch, L., Rothfuss, J., Schölkopf, B., & Krause, A.
- **Year**: 2021
- **Preprint**: arXiv:2106.07635
- **Link**: [arxiv:2106.07635](https://arxiv.org/abs/2106.07635)
- **Why Important**: Recent neural causal discovery method using VAE
- **Project Relevance**: Implemented in evaluation suite as baseline wrapper
- **Relevance**: Modern comparison point

---

## 🚀 NOVEL INTEGRATION CANDIDATES (Papers to Incorporate)

### Advanced Structure Learning
**28. Masked Autoencoder Density Estimation for Conditional Generation**
- **Authors**: Song, X., Dillon, J., Guestrin, C., & Leskovec, J.
- **Year**: 2023
- **Preprint**: arXiv:2301.13867
- **Link**: [arxiv:2301.13867](https://arxiv.org/abs/2301.13867)
- **Why Relevant**: Masked learning approach could improve edge prediction
- **Integration Idea**: Use masked attention in transformer for progressive edge discovery

---

### Advanced Dynamics Learning
**29. Transformer-based Models for Forecasting Multivariate Time Series**
- **Authors**: Zeng, A., Chen, M., Zhang, L., & Xu, Q.
- **Year**: 2022
- **Preprint**: arXiv:2205.13504
- **Link**: [arxiv:2205.13504](https://arxiv.org/abs/2205.13504)
- **Why Relevant**: Advanced techniques for sequential prediction (deltas are sequential)
- **Integration Idea**: Multi-scale temporal encoding for variable-length interventions

---

### Interventional Data Handling
**30. Causal Inference with Hidden Variables via General Automorphic Markov Models**
- **Authors**: Squires, C., Sekhon, A., Cian, Y., & Uhler, C.
- **Year**: 2023
- **Preprint**: arXiv:2306.08821
- **Link**: [arxiv:2306.08821](https://arxiv.org/abs/2306.08821)
- **Why Relevant**: Handling hidden confounders with interventional data
- **Integration Idea**: Augment ISD-CP with latent variable modeling for realistic data

---

### Scalability Improvements
**31. Efficient Neural Causal Discovery without Acyclicity Constraints**
- **Authors**: Montagna, F., Noceti, N., Odone, F., & Barla, A.
- **Year**: 2023
- **Preprint**: arXiv:2309.11821
- **Link**: [arxiv:2309.11821](https://arxiv.org/abs/2309.11821)
- **Why Relevant**: Relaxing DAG constraints for better scalability
- **Integration Idea**: Soft relaxation of h-function for better gradient flow at scale

---

### Uncertainty Quantification
**32. Bayesian Causal Forests for Individual Treatment Effects**
- **Authors**: Athey, S., Tibshirani, J., & Wager, S.
- **Year**: 2019
- **Preprint**: arXiv:1904.10320
- **Link**: [arxiv:1904.10320](https://arxiv.org/abs/1904.10320)
- **Why Relevant**: Heterogeneous treatment effects and uncertainty
- **Integration Idea**: Add uncertainty estimates to delta predictions (confidence intervals)

---

### Graph Structure Refinement
**33. Learning to Simulate Complex Physics with Graph Networks**
- **Authors**: Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W.
- **Year**: 2021
- **Preprint**: arXiv:2002.09405
- **Link**: [arxiv:2002.09405](https://arxiv.org/abs/2002.09405)
- **Why Relevant**: Graph-based physics simulation and structure refinement
- **Integration Idea**: Use message-passing to refine discovered structures iteratively

---

### Mixed Discrete-Continuous Learning
**34. Differentiable Causal Discovery from Interventional Data**
- **Authors**: Bengio, Y., et al.
- **Year**: 2021
- **Preprint**: Various ICML papers
- **Why Relevant**: Handling mixed continuous/discrete causal structures
- **Integration Idea**: Extend ISD-CP to mixed discrete-continuous systems

---

### Few-Shot Causal Discovery
**35. Meta-Learning for Causal Structure Learning**
- **Authors**: Recent work (ICLR 2022+)
- **Why Relevant**: Learn to discover causal structures from few interventions
- **Integration Idea**: Meta-learning wrapper around ISD-CP for efficient transfer

---

### Real-World Benchmarks
**36. Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data** (Sachs Dataset)
- **Authors**: Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P.
- **Year**: 2005
- **Journal**: Science
- **Link**: [https://doi.org/10.1126/science.1105809](https://doi.org/10.1126/science.1105809)
- **Why Important**: Standard real-world benchmark (11 proteins, 17 edges)
- **Current Status**: Not yet evaluated on this dataset

---

### Scalability & Efficiency
**37. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**
- **Authors**: Lepikhin, D., Lee, H., Xu, Y., Chen, D., et al.
- **Year**: 2021
- **Conference**: ICLR
- **Link**: [arxiv:2101.03961](https://arxiv.org/abs/2101.03961)
- **Why Relevant**: Advanced sparse expert routing for massive scales
- **Integration Idea**: Upgrade from fixed 8 experts to learned expert allocation

---

## 📊 READING ROADMAP BY TOPIC

### **Phase 1: Foundations (1-2 weeks)**
1. Pearl - Causality (Chapters 1-3)
2. Vaswani et al. - Attention is All You Need
3. Zheng et al. - DAGs with NO TEARS

### **Phase 2: ISD-CP Specific (1-2 weeks)**
4. Su et al. - RoPE
5. Jang et al. - Gumbel-Softmax
6. Shazeer et al. - Mixture of Experts
7. Tancik et al. - Fourier Features

### **Phase 3: Advanced Techniques (1-2 weeks)**
8. Shazeer - GLU Variants
9. Bengio et al. - Curriculum Learning
10. Loshchilov & Hutter - AdamW and SGDR

### **Phase 4: Causal Discovery Baselines (1-2 weeks)**
11. Zheng et al. - NOTEARS-MLP
12. Yu et al. - DAG-GNN
13. Lachapelle et al. - GraN-DAG

### **Phase 5: Extensions & Future Work (2-3 weeks)**
14. Modern papers (28-38 above)
15. Real-world benchmarks (Sachs, ALARM)

---

## 🎯 QUICK REFERENCE TABLE

| Paper | Year | Type | Criticality | Location in Code |
|-------|------|------|-------------|------------------|
| Vaswani et al. | 2017 | Architecture | CRITICAL | CausalTransformer.py |
| Su et al. (RoPE) | 2021 | Positional | CRITICAL | rope.py |
| Zheng et al. (NOTEARS) | 2018 | Loss Function | CRITICAL | loss.py |
| Jang et al. (Gumbel) | 2016 | Routing | CRITICAL | CausalTransformer.py |
| Shazeer et al. (MoE) | 2017 | Architecture | HIGH | CausalTransformer.py |
| Tancik et al. (Fourier) | 2020 | Embedding | HIGH | encoder.py |
| Shazeer (GLU) | 2020 | Activation | MEDIUM | CausalTransformer.py |
| Bengio et al. (Curriculum) | 2009 | Training | HIGH | curriculum.py |
| Peters et al. (Causality Book) | 2017 | Theoretical | CRITICAL | Concepts |
| Pearl (Causality Book) | 2009 | Theoretical | CRITICAL | Concepts |

---

## 📌 RESEARCH GAPS (Potential Contributions)

1. **Transformer-based Causal Discovery**: Likely first in the literature
2. **Twin-World Variance Reduction**: Novel SCM training technique
3. **Hard MoE for Physics**: Unusual application of Gumbel routing
4. **Unified Structure+Function Learning**: Most methods do them separately
5. **Multi-Dimensional Curriculum**: More comprehensive than single-dimension

---

## 🔗 USEFUL DATABASES & TOOLS

- **arXiv.org**: Most preprints referenced
- **DBLP.org**: Academic publication search
- **Papers with Code** (paperswithcode.com): Implementations of baselines
- **Google Scholar**: Citation tracking
- **ResearchGate**: Author contacts for clarifications

---

**Total Papers Listed**: 38 (29 explicitly in review.md + 9 additional recommendations)  
**MUST READ**: 10 papers (foundation + critical methods)  
**BETTER TO READ**: 17 papers (baselines + deeper understanding)  
**NOVEL INTEGRATION**: 9 papers (future research directions)

