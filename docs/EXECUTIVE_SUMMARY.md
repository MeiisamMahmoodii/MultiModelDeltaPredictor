# Executive Summary & Technical Report

## 1. Project Overview
**Goal:** Build a "Universal Causal Simulator" that predicts how a complex system changes ($\Delta$) after an intervention ($do(X_i=v)$), *without* knowing the underlying causal graph beforehand.
**Philosophy:** "Physics First" — If the model can accurately predict the consequences of interventions, it must have implicitly understood the causal structure.

## 2. Core Innovations (The "Secret Sauce")

### A. Data & Training
*   **Twin World Generation:** We generate pairs of samples $(X_{base}, X_{int})$ from the exact same exogenous noise ($\epsilon$). This isolates the *pure causal effect* of the intervention, removing noise variance.
*   **Curriculum Learning:** We train on increasing complexity (Levels), starting from small graphs/easy mechanisms and scaling up to 50+ variables and non-linear interactions.
*   **Decoupled Training Strategy:**
    1.  **Phase 1 (Physics Only)**: Train with `lambda_dag=0`. Force the model to master the input-output mapping $f(X, I) \to \Delta$.
    2.  **Phase 2 (Structure Refinement)**: Gently introduce DAG constraints (`lambda_dag`) to prune false positive edges from the learned internal representation.

### B. Model Architecture (The "ISD-CP" Transformer)
*   **Interleaved Pattern Encoding:** instead of standard tabular embedding, we treat the row as a sequence of `(FeatureID, Value)` pairs. This allows the Transformer to serve as a proper Graph Neural Network (GNN).
*   **Learned Causal Mask:** We project the internal attention weights to an explicit Adjacency Matrix ($A_{pred} = Q K^T$). This matrix is then used to *mask* the attention in subsequent layers (Iterative Refinement), forcing the computation to follow the predicted causal graph.
*   **Hard-Gumbel Mixture of Experts (MoE)**: To handle diverse mechanism types (Linear, Sigmoid, Tanh, etc.), we use a Sparse MoE layer.
    *   **Innovation:** We added a `router_tau` parameter to control expert specialization vs. diversity.

## 3. Current Status (The Pivot)

**Critical Finding (Jan 2026): The Zero-Inflation Problem**
*   **Observation**: In sparse causal graphs ($p=0.2$), ~45% of all intervention effects are **exactly zero** (non-descendants).
*   **Failure**: Choosing `HuberLoss` (which is soft near zero) caused the model to output small noise instead of true zeros, failing to beat a "Predict Zero" baseline (MAE 3.7).
*   **Solution**: We pivoted to **L1 Loss** (Laplace Prior).
*   **Result**: The model immediately beat the baseline, achieving **MAE 3.23** (train) and **2.87** (easy val). The Physics Engine is now viable.

## 4. Problems & Challenges

1.  **Structure Oscillation**: In Phase 2, the Structural Hamming Distance (SHD) oscillates (e.g., 500 $\leftrightarrow$ 350). This "Thrashing" happens because `lambda_dag` fights against the `MAE` loss necessary for physics accuracy.
2.  **Sparsity vs. Gradient**: Gradients for "Zero Effect" are hard to learn with standard regression heads. The model struggles to output *hard* zeros.

## 5. Suggestions & Future Directions

### A. Immediate Improvements (The "Zero-Inflated" Architecture)
The current regression head tries to do two jobs: detect *if* there is an effect, and predict *how much*. We should split this:
*   **Head 1 (Gating)**: A Classifier that outputs $P(\text{Effect} \neq 0)$. Trained with BCE Loss.
*   **Head 2 (Regression)**: A Regressor that outputs $\Delta$. Trained with MSE/L1.
*   **Result**: $\Delta_{final} = \text{Head2} \times \mathbb{I}(\text{Head1} > 0.5)$. This explicitly models the 45% sparsity.

### B. Advanced Structure Learning (Factor Graphs)
*   **Concept**: Instead of learning $N^2$ edges, learn Low-Rank Factors ($U, V$).
*   **Status**: We partially implemented this via the Attention Mechanism ($Q K^T$), which is rank-deficient.
*   **Next Step**: Formalize this by restricting the rank of the Predicted Adjacency Matrix, improving scalability to $N=1000$.

### C. Differentiable Sorting (Top-K)
*   To enforce DAG acyclicity more efficiently than the expensive Matrix Exponential (`trace(exp(A))`), we could use Differentiable Sorting/Permutation learning (e.g., Gumbel-Sinkhorn) to impose a topological order.

## 6. Conclusion
The project has successfully navigated from a theoretical concept to a working "Physics Engine" that outperforms trivial baselines. The key insight was realizing that **Loss Function Geometry** (L1 vs Huber) matters more than architecture when dealing with sparse causal effects. The next frontier is explicitly modeling the "Existence of Effect" separate from the "Magnitude of Effect."
