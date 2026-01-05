Comprehensive Literature Review and Novelty Analysis
MultiModelDeltaPredictor (ISD-CP) Project
Date: January 5, 2026
Project: Interleaved Structural Discovery via Causal Prediction (ISD-CP)

Executive Summary
This project presents ISD-CP, a transformer-based framework for learning Structural Causal Models (SCMs) from observational and interventional data. The system combines multiple state-of-the-art techniques from deep learning, causal discovery, and optimization theory to simultaneously learn:

Physics (Dynamics): Predicting state changes (deltas) under interventions
Structure (Topology): Discovering the underlying causal graph (DAG)
Key Innovation: Unified architecture that learns both tasks from the same representation, using interleaved token encoding and twin-world variance reduction.

1. Scientific Methods and References
1.1 Transformer Architecture
1.1.1 Rotary Positional Embeddings (RoPE)
Reference:

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv preprint arXiv:2104.09864.

Original Method:

Encodes absolute positions by rotating query and key vectors in complex space
Rotation angle is proportional to position: θ_m = m × θ, where m is position
Applied via: q_m = q × cos(mθ) + rotate_half(q) × sin(mθ)
Enables relative position encoding through dot product properties
Implementation in Project: 
src/models/rope.py

Modifications:

✅ Faithful Implementation: Standard RoPE with no major changes
Base frequency: 10000 (standard)
Max sequence length: 2048 (configurable)
Applied in custom attention layer rather than using pre-built transformers
Why Used: Causal graphs have no inherent sequential order. RoPE allows the model to learn relative relationships between nodes regardless of their position in the input sequence.

1.1.2 Self-Attention Mechanism
Reference:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

Original Method:

Scaled dot-product attention: 
Attention(Q,K,V) = softmax(QK^T/√d_k)V
Multi-head attention for parallel representation learning
Implementation: 
src/models/CausalTransformer.py

Modifications:

Custom implementation to integrate RoPE (lines 183-190)
Uses PyTorch's F.scaled_dot_product_attention for efficiency
No causal masking (all-to-all attention) since causal structure is what we're learning
8 attention heads with 512-dimensional model
1.2 Mixture of Experts (MoE)
1.2.1 Hard Gumbel-Softmax Routing
Primary Reference:

Jang, E., Gu, S., & Poole, B. (2016). Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144.

Secondary Reference:

Maddison, C. J., Mnih, A., & Teh, Y. W. (2016). The concrete distribution: A continuous relaxation of discrete random variables. arXiv preprint arXiv:1611.00712.

Original Method:

Gumbel-Softmax: Continuous relaxation of discrete distributions
y = softmax((log(π) + g) / τ) where g ~ Gumbel(0,1)
Temperature τ controls discreteness
Straight-through estimator for hard sampling: y_hard - y_soft.detach() + y_soft
Implementation: 
src/models/CausalTransformer.py

Modifications:

✅ Standard Gumbel-Softmax for expert routing (line 122)
Hard mode enabled (hard=True) for discrete expert selection
Fixed temperature τ=1.0 (no annealing for routing)
8 experts with 4-layer depth each
Novel Aspect: Applied to causal discovery (not typical use case for MoE)

1.2.2 Vectorized Expert Architecture
Reference (MoE concept):

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538.

Original Method:

Top-k expert selection
Load balancing loss
Sparse activation for efficiency
Implementation: 
src/models/CausalTransformer.py

Modifications (Significant):

❌ No top-k gating → Hard Gumbel routing (exactly 1 expert per token)
❌ No load balancing loss → Relies on Gumbel noise for diversity
✅ Vectorized execution: All experts run in parallel using torch.einsum
Custom expert architecture: SwiGLU blocks instead of standard FFN
Why Modified: Hard routing forces specialization (one expert per physics type), avoiding the "expert collapse" problem in soft MoE.

1.2.3 SwiGLU Activation
Reference:

Shazeer, N. (2020). GLU variants improve transformer. arXiv preprint arXiv:2002.05202.

Original Method:

SwiGLU(x) = Swish(xW_gate) ⊙ (xW_val)
Swish(x) = x × sigmoid(x) (also called SiLU)
Gated activation for better gradient flow
Implementation: 
src/models/CausalTransformer.py

Modifications:

✅ Standard SwiGLU (using F.silu)
Expansion factor: 8× (line 42)
Residual connections added (line 69)
Used in vectorized expert blocks
1.3 Normalization Techniques
1.3.1 RMSNorm (Root Mean Square Normalization)
Reference:

Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. Advances in Neural Information Processing Systems, 32.

Original Method:

RMSNorm(x) = x / RMS(x) × γ
RMS(x) = √(mean(x²) + ε)
Simpler than LayerNorm (no mean subtraction)
Implementation: 
src/models/CausalTransformer.py

Modifications:

✅ Standard RMSNorm
Used in expert blocks (line 44)
ε = 1e-8 for numerical stability
Why Used: Faster than LayerNorm, sufficient for normalization in expert blocks.

1.3.2 LayerNorm
Reference:

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.

Implementation: Standard PyTorch LayerNorm in attention layers

1.4 Causal Discovery Methods
1.4.1 DAG Constraint via Matrix Exponential (h-function)
Reference:

Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. Advances in Neural Information Processing Systems, 31.

Original Method:

Acyclicity constraint: h(A) = tr(e^(A⊙A)) - d = 0
Differentiable constraint for gradient-based optimization
Augmented Lagrangian optimization
Implementation: 
src/training/loss.py

Modifications:

✅ Standard h-function (line 9)
Batch processing: Loop over batch dimension (lines 59-62)
MPS fallback: CPU computation for matrix_exp on Apple Silicon (lines 7-10)
Weighted loss: λ_h controls strength (configurable, default 0.0 → 1.0)
Why Modified: Original NOTEARS uses augmented Lagrangian with dual ascent. This project uses simple weighted loss for simplicity.

1.4.2 Structural Hamming Distance (SHD)
Reference (Standard metric in causal discovery):

Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure learning algorithm. Machine learning, 65(1), 31-78.

Original Method:

SHD = # of edge insertions + deletions + reversals to match true graph
Standard evaluation metric
Implementation: 
src/training/metrics.py

Modifications:

✅ Standard SHD (simplified: counts edge differences)
Threshold = 0.0 for logits (uses sigmoid internally)
1.5 Data Generation and Encoding
1.5.1 Structural Causal Models (SCMs)
Reference (Foundational):

Pearl, J. (2009). Causality: Models, reasoning and inference (2nd ed.). Cambridge University Press.

Original Method:

SCM: X_i = f_i(PA_i, U_i) where PA_i are parents, U_i is noise
Interventions: do(X_i = x) replaces f_i with constant
Implementation: 
src/data/SCMGenerator.py

Modifications (Significant):

13 function types: linear, quadratic, cubic, sin, cos, tanh, sigmoid, step, abs, etc. (lines 54-68)
Interaction terms: 30% probability of multiplicative interactions (lines 120-125)
Twin-world sampling: Same noise for observational and interventional data (lines 154-172)
Value clipping: [-100, 100] range (line 130)
Novel Aspect: Twin-world variance reduction for delta prediction (not standard in causal discovery).

1.5.2 Fourier Features for Value Embedding
Reference:

Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... & Ng, R. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. arXiv preprint arXiv:2006.10739.

Original Method:

γ(v) = [sin(2πb₁v), cos(2πb₁v), ..., sin(2πbₘv), cos(2πbₘv)]
Frequencies b sampled from Gaussian or fixed powers of 2
Enables learning high-frequency functions
Implementation: 
src/data/encoder.py

Modifications:

Fixed frequencies: 2^0, 2^1, ..., 2^7 (line 10)
Projection layer: Maps to d_model/2 dimensions (line 11)
Hybrid embedding: Combined with linear and MLP embeddings (lines 22-56)
Novel Aspect: Hybrid embedding strategy (Linear + Fourier + MLP) for physics-aware value encoding.

1.5.3 Interleaved Token Encoding
Reference (Concept similar to):

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Original Method (ViT):

Patch embeddings: Image → sequence of patches
Position embeddings added
Implementation: 
src/data/encoder.py

Modifications (Novel):

Interleaved format: [ID₀, Value₀, ID₁, Value₁, ...] (lines 107-110)
Three embedding types:
Feature ID (which variable)
Value (hybrid: linear + Fourier + MLP)
Type (observed/intervened/masked)
Type embedding: Distinguishes observational vs interventional data (lines 101-105)
Novel Aspect: Interleaved encoding for causal data (not standard in causal discovery or transformers).

1.6 Loss Functions and Optimization
1.6.1 Huber Loss
Reference:

Huber, P. J. (1964). Robust estimation of a location parameter. The annals of mathematical statistics, 73-101.

Original Method:

Robust regression loss: L1 for large errors, L2 for small errors
Less sensitive to outliers than MSE
Implementation: 
src/training/loss.py

Modifications:

✅ Standard Huber loss via F.huber_loss
Used for delta prediction (continuous values)
Why Used: Robust to outliers in physics predictions.

1.6.2 Binary Cross-Entropy with Logits
Reference (Standard):

Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

Implementation: 
src/training/loss.py

Modifications:

Positive weight: 3.0 to handle class imbalance (line 26)
Used for graph structure prediction (sparse adjacency matrix)
Why Modified: Causal graphs are sparse (~20% edges), so positive weight balances false negatives.

1.6.3 AdamW Optimizer
Reference:

Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Implementation: 
main.py

Modifications:

✅ Standard AdamW
Learning rate: 2e-4 (line 74)
Gradient clipping: max_norm=0.1 (line 289)
1.6.4 Cosine Annealing with Warm Restarts
Reference:

Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983.

Original Method:

Cosine learning rate schedule with periodic restarts
η_t = η_min + 0.5(η_max - η_min)(1 + cos(πT_cur/T_i))
T_mult controls period doubling
Implementation: 
main.py

Modifications:

✅ Standard SGDR
T_0 = 50 epochs, T_mult = 2
η_min = 1e-8
1.7 Curriculum Learning
Reference:

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. Proceedings of the 26th annual international conference on machine learning, 41-48.

Original Method:

Train on easy examples first, gradually increase difficulty
Improves convergence and generalization
Implementation: 
src/training/curriculum.py

Modifications (Novel):

Multi-dimensional curriculum:
Number of variables: 20 → 50 (lines 24-25)
Graph density: 15-25% → 25-35% (lines 28-29)
Intervention range: 2.0 → 10.0 (line 32)
Adaptive thresholds: MAE thresholds increase with difficulty (lines 51-53)
Stability patience: 5 epochs before level-up (line 12)
Novel Aspect: Joint curriculum over graph size, density, and intervention strength.

1.8 Gradient Checkpointing
Reference:

Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174.

Implementation: 
src/models/CausalTransformer.py

Modifications:

✅ Standard gradient checkpointing
use_reentrant=False for compatibility (line 270)
Applied to forward passes (3-step refinement)
2. Novel Contributions and Modifications
2.1 Core Novelties
Novelty 1: Unified Physics-Structure Learning
What: Single transformer that predicts both:

Continuous deltas (physics/dynamics)
Discrete graph structure (topology)
Why Novel: Most causal discovery methods separate structure learning from function learning. This project learns both from shared representations.

Evidence: Dual-head architecture (lines 249-260 in CausalTransformer.py)

Novelty 2: Interleaved Token Encoding for Causal Data
What: [Feature_ID, Value, Feature_ID, Value, ...] sequence format

Why Novel:

Standard causal methods use tabular data
Transformers typically use sequential (text) or spatial (vision) data
This encoding makes causal data "transformer-native"
Evidence: InterleavedEncoder class (encoder.py)

Novelty 3: Twin-World Variance Reduction
What: Generate observational and interventional data with identical noise

Why Novel:

Standard SCM sampling uses independent noise
Twin-world reduces variance in delta estimation
Enables direct delta supervision: Δ = f(X, do(X_i)) - f(X)
Evidence: Lines 154-172 in SCMGenerator.py

Novelty 4: Hybrid Physics-Aware Embeddings
What: Value embeddings combine:

Linear (magnitude)
Fourier (periodicity)
MLP (distortion/sharpness)
Why Novel: Designed specifically for diverse physics functions (linear, sin, cubic, etc.)

Evidence: HybridEmbedding class (encoder.py, lines 22-56)

Novelty 5: Hard MoE for Physics Specialization
What: Hard Gumbel routing forces each token to select exactly one expert

Why Novel:

Standard MoE uses soft routing (weighted average)
Hard routing forces specialization (one expert per physics type)
Prevents "expert collapse" where all experts learn the same function
Evidence: Line 122 in CausalTransformer.py (hard=True)

Novelty 6: Recurrent Refinement (3-Step)
What: Model makes 3 forward passes:

Initial prediction
Refinement using predicted deltas
Final polish
Why Novel: Iterative refinement for complex physics (not standard in transformers or causal discovery)

Evidence: Lines 267-303 in CausalTransformer.py

Novelty 7: Multi-Dimensional Curriculum
What: Curriculum over graph size, density, AND intervention strength

Why Novel: Most curriculum learning varies one dimension (e.g., only graph size)

Evidence: Lines 24-38 in curriculum.py

2.2 Significant Modifications from Original Papers
Component	Original Paper	Modification	Rationale
MoE Routing	Top-k soft gating	Hard Gumbel (k=1)	Force expert specialization
NOTEARS h-loss	Augmented Lagrangian	Weighted loss term	Simplicity (no dual variables)
Fourier Features	Random frequencies	Fixed powers of 2	Deterministic, covers broad range
Curriculum	Single dimension	Multi-dimensional	Joint difficulty scaling
SCM Functions	Typically linear	13 types + interactions	Realistic physics complexity
Attention	Standard	RoPE-enhanced	Relative position encoding
Expert Architecture	Standard FFN	SwiGLU + Vectorized	Better gradients + efficiency
3. Comparison Targets and Benchmarks
3.1 Causal Discovery Methods
3.1.1 Constraint-Based Methods
PC Algorithm:

Spirtes, P., Glymour, C. N., Scheines, R., & Heckerman, D. (2000). Causation, prediction, and search. MIT press.

FCI (Fast Causal Inference):

Spirtes, P., Glymour, C., & Scheines, R. (1993). Causation, prediction, and search. Springer.

Comparison:

These use conditional independence tests
Your model uses neural networks (more scalable, handles non-linearity)
Benchmark: Compare SHD, F1 on same graphs
3.1.2 Score-Based Methods
GES (Greedy Equivalence Search):

Chickering, D. M. (2002). Optimal structure identification with greedy search. Journal of machine learning research, 3(Nov), 507-554.

BIC/BDe Scoring:

Schwarz, G. (1978). Estimating the dimension of a model. The annals of statistics, 461-464.

Comparison:

These optimize BIC/BDe scores
Your model uses gradient descent on neural loss
Benchmark: Compare on graphs with 20-50 nodes
3.1.3 Continuous Optimization Methods
NOTEARS:

Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. NeurIPS.

NOTEARS-MLP:

Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E. (2020). Learning sparse nonparametric DAGs. AISTATS.

DAG-GNN:

Yu, Y., Chen, J., Gao, T., & Yu, M. (2019). DAG-GNN: DAG structure learning with graph neural networks. ICML.

Comparison:

NOTEARS: Linear/MLP functions, h-constraint (similar to yours)
DAG-GNN: Uses GNN instead of transformer
Benchmark: Direct comparison on SHD, F1, TPR, FDR
IMPORTANT

Primary Comparison: NOTEARS-MLP is your closest competitor (continuous optimization + non-linear functions)

3.1.4 Gradient-Based Methods
GraN-DAG:

Lachapelle, S., Brouillard, P., Deleu, T., & Lacoste-Julien, S. (2019). Gradient-based neural DAG learning. arXiv preprint arXiv:1906.02226.

GOLEM:

Ng, I., Ghassami, A., & Zhang, K. (2020). On the role of sparsity and DAG constraints for learning linear DAGs. NeurIPS.

Comparison:

These use gradient-based optimization like yours
Benchmark: Compare convergence speed, final SHD
3.2 Neural Causal Discovery
3.2.1 Transformer-Based
Causal Transformer (if exists in literature - search needed):

Most causal discovery doesn't use transformers
Your work may be first transformer-based causal discovery
Comparison: Literature search needed for transformer-based causal methods

3.2.2 Variational Methods
AVICI:

Lorch, L., Rothfuss, J., Schölkopf, B., & Krause, A. (2021). AVICI: A variational autoencoder for causal inference. arXiv preprint arXiv:2106.07635.

Comparison:

Uses VAE for causal discovery
Benchmark: Compare on interventional data
3.3 Function Learning Methods
3.3.1 Neural ODEs
Neural ODE:

Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. NeurIPS.

Comparison:

Learns continuous dynamics
Your model learns discrete deltas
Benchmark: Compare MAE on delta prediction
3.3.2 Physics-Informed Neural Networks
PINN:

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

Comparison:

Uses physics constraints
Your model learns physics from data
Benchmark: Compare on systems with known physics
3.4 Benchmark Datasets
3.4.1 Synthetic Benchmarks
Erdős-Rényi Graphs:

Random graphs with fixed edge probability
Your setup: 20-50 nodes, 15-35% density ✅
Scale-Free Networks:

Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509-512.

Comparison: Test on scale-free graphs (not just ER)

3.4.2 Real-World Benchmarks
Sachs Dataset:

Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. Science, 308(5721), 523-529.

11 proteins, 17 edges
Flow cytometry data
Benchmark: Standard in causal discovery
Asia Network:

Lauritzen, S. L., & Spiegelhalter, D. J. (1988). Local computations with probabilities on graphical structures and their application to expert systems. Journal of the Royal Statistical Society: Series B, 50(2), 157-194.

8 nodes (medical diagnosis)
Benchmark: Small but classic
Alarm Network:

Beinlich, I. A., Suermondt, H. J., Chavez, R. M., & Cooper, G. F. (1989). The ALARM monitoring system. Proceedings of the Second European Conference on Artificial Intelligence in Medicine, 247-256.

37 nodes (medical monitoring)
Benchmark: Medium-scale
3.5 Recommended Comparison Suite
Tier 1: Must Compare
NOTEARS-MLP (closest method)
DAG-GNN (neural baseline)
GES (classical baseline)
Tier 2: Should Compare
GraN-DAG (gradient-based)
PC Algorithm (constraint-based)
GOLEM (linear baseline)
Tier 3: Nice to Have
AVICI (variational)
Neural ODE (dynamics baseline)
4. Evaluation Metrics
4.1 Structure Learning Metrics
Metric	Formula	Reference
SHD	# edge differences	Tsamardinos et al., 2006
F1 Score	2·(Precision·Recall)/(Precision+Recall)	Standard
TPR (Recall)	TP / (TP + FN)	Standard
FDR	FP / (TP + FP)	Standard
Precision	TP / (TP + FP)	Standard
Your Implementation: 
src/training/metrics.py

4.2 Function Learning Metrics
Metric	Formula	Reference
MAE	mean(|pred - true|)	Standard
RMSE	√(mean((pred - true)²))	Standard
R²	1 - SS_res/SS_tot	Standard
Your Implementation: MAE in metrics.py (line 25-30)

5. Publication Strategy
5.1 Target Venues
Tier 1 (Top Conferences)
NeurIPS (Neural Information Processing Systems)

Deadline: May
Focus: ML methods, causal discovery track
ICML (International Conference on Machine Learning)

Deadline: January
Focus: Novel architectures, learning theory
ICLR (International Conference on Learning Representations)

Deadline: September
Focus: Representation learning, transformers
Tier 2 (Specialized)
UAI (Uncertainty in Artificial Intelligence)

Deadline: February
Focus: Causal inference, probabilistic methods
AISTATS (Artificial Intelligence and Statistics)

Deadline: October
Focus: Statistical methods, causal discovery
Journals
JMLR (Journal of Machine Learning Research)
Rolling submissions
Focus: Significant methodological contributions
5.2 Positioning
Title Suggestions:

"ISD-CP: Interleaved Structural Discovery via Causal Prediction with Transformers"
"Learning Causal Graphs and Dynamics with Transformer-Based Mixture of Experts"
"Twin-World Causal Discovery: Unified Structure and Function Learning"
Key Selling Points:

First transformer-based causal discovery (if true after literature search)
Unified learning of structure and function
Novel encoding (interleaved tokens)
Twin-world variance reduction
Scalability to 50+ variables
6. Missing References to Add
6.1 Causal Discovery Surveys
Glymour, C., Zhang, K., & Spirtes, P. (2019). Review of causal discovery methods based on graphical models. Frontiers in genetics, 10, 524.

Heinze-Deml, C., Maathuis, M. H., & Meinshausen, N. (2018). Causal structure learning. Annual Review of Statistics and Its Application, 5, 371-391.

6.2 Interventional Data
Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of causal inference: foundations and learning algorithms. MIT press.

Eberhardt, F., & Scheines, R. (2007). Interventions and causal inference. Philosophy of Science, 74(5), 981-995.

6.3 Graph Neural Networks
Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.

(For comparison with DAG-GNN)

7. Experimental Validation Checklist
7.1 Ablation Studies
 Remove RoPE → Standard positional encoding
 Remove MoE → Standard FFN
 Remove twin-world → Independent noise
 Remove hybrid embeddings → Linear only
 Remove recurrent refinement → Single pass
 Remove curriculum → Fixed difficulty
7.2 Scalability Tests
 Graph size: 10, 20, 30, 40, 50, 100 nodes
 Graph density: 10%, 20%, 30%, 40%
 Function types: Linear only, Non-linear, Mixed
7.3 Comparison Experiments
 vs NOTEARS-MLP (same graphs)
 vs DAG-GNN (same graphs)
 vs GES (small graphs)
 vs PC (small graphs)
7.4 Real-World Validation
 Sachs dataset (11 nodes)
 Alarm network (37 nodes)
 Custom domain (if available)
8. Code Quality and Reproducibility
8.1 Strengths
✅ Modular design: Clear separation of concerns
✅ Documented: README with architecture diagrams
✅ Checkpointing: Resume training capability
✅ Logging: CSV logs for all metrics
✅ Curriculum: Adaptive difficulty scaling

8.2 Improvements Needed
 Unit tests: Add tests for each component
 Hyperparameter config: YAML/JSON config files
 Seed management: Reproducible random seeds
 Visualization: Plot training curves, learned graphs
 Benchmarking: Scripts to run comparison methods
 Documentation: API docs, tutorial notebooks
9. Summary Table: All Methods and References
Component	Original Paper	Year	Modifications
RoPE	Su et al.	2021	✅ Standard
Transformer	Vaswani et al.	2017	Custom attention + RoPE
Gumbel-Softmax	Jang et al.	2016	Hard mode for routing
MoE	Shazeer et al.	2017	Hard routing, no load balancing
SwiGLU	Shazeer	2020	✅ Standard
RMSNorm	Zhang & Sennrich	2019	✅ Standard
LayerNorm	Ba et al.	2016	✅ Standard
NOTEARS h-loss	Zheng et al.	2018	Weighted loss (no Lagrangian)
SHD	Tsamardinos et al.	2006	✅ Standard
SCM	Pearl	2009	13 functions + interactions
Fourier Features	Tancik et al.	2020	Fixed frequencies
Huber Loss	Huber	1964	✅ Standard
AdamW	Loshchilov & Hutter	2017	✅ Standard
SGDR	Loshchilov & Hutter	2016	✅ Standard
Curriculum	Bengio et al.	2009	Multi-dimensional
Grad Checkpoint	Chen et al.	2016	✅ Standard
10. Novelty Summary (For Paper Abstract)
We present ISD-CP, a transformer-based framework for causal discovery that simultaneously learns graph structure and dynamics from interventional data. Our key innovations include: (1) interleaved token encoding that makes causal data transformer-native, (2) twin-world variance reduction for accurate delta prediction, (3) hard mixture-of-experts with physics-aware routing, (4) hybrid embeddings combining linear, Fourier, and MLP features, and (5) multi-dimensional curriculum learning over graph size, density, and intervention strength. Experiments on graphs with 20-50 variables show [X]% improvement in SHD and [Y]% improvement in MAE compared to NOTEARS-MLP, while scaling to larger graphs than constraint-based methods.

11. Next Steps
Literature Search: Verify no existing transformer-based causal discovery
Implement Baselines: NOTEARS-MLP, DAG-GNN, GES
Ablation Studies: Quantify contribution of each component
Real-World Validation: Test on Sachs, Alarm datasets
Write Paper: Follow ICML/NeurIPS template
Release Code: GitHub with reproducibility scripts
References (Complete Bibliography)
Transformers and Attention
Vaswani et al. (2017). Attention is all you need. NeurIPS.
Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
Dosovitskiy et al. (2020). An image is worth 16x16 words. ICLR.
Mixture of Experts
Shazeer et al. (2017). Outrageously large neural networks. arXiv:1701.06538.
Jang et al. (2016). Categorical reparameterization with gumbel-softmax. arXiv:1611.01144.
Maddison et al. (2016). The concrete distribution. arXiv:1611.00712.
Shazeer (2020). GLU variants improve transformer. arXiv:2002.05202.
Normalization
Zhang & Sennrich (2019). Root mean square layer normalization. NeurIPS.
Ba et al. (2016). Layer normalization. arXiv:1607.06450.
Causal Discovery
Zheng et al. (2018). DAGs with NO TEARS. NeurIPS.
Zheng et al. (2020). Learning sparse nonparametric DAGs. AISTATS.
Yu et al. (2019). DAG-GNN. ICML.
Lachapelle et al. (2019). Gradient-based neural DAG learning. arXiv:1906.02226.
Ng et al. (2020). On the role of sparsity and DAG constraints. NeurIPS.
Spirtes et al. (2000). Causation, prediction, and search. MIT Press.
Chickering (2002). Optimal structure identification with greedy search. JMLR.
Tsamardinos et al. (2006). The max-min hill-climbing Bayesian network. Machine Learning.
Pearl (2009). Causality: Models, reasoning and inference. Cambridge.
Peters et al. (2017). Elements of causal inference. MIT Press.
Glymour et al. (2019). Review of causal discovery methods. Frontiers in Genetics.
Embeddings and Features
Tancik et al. (2020). Fourier features let networks learn high frequency functions. NeurIPS.
Optimization
Loshchilov & Hutter (2017). Decoupled weight decay regularization. ICLR.
Loshchilov & Hutter (2016). SGDR: Stochastic gradient descent with warm restarts. ICLR.
Huber (1964). Robust estimation of a location parameter. Annals of Statistics.
Learning Strategies
Bengio et al. (2009). Curriculum learning. ICML.
Chen et al. (2016). Training deep nets with sublinear memory cost. arXiv:1604.06174.
Benchmarks
Sachs et al. (2005). Causal protein-signaling networks. Science.
Lauritzen & Spiegelhalter (1988). Local computations with probabilities. JRSS.
Beinlich et al. (1989). The ALARM monitoring system. ECAI.
End of Report

Generated: January 5, 2026
Project: MultiModelDeltaPredictor (ISD-CP)
Total References: 29 papers