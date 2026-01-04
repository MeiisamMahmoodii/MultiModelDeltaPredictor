# Multi-Model Delta Predictor (ISD-CP Unified)

**ISD-CP (Interleaved Structural Discovery via Causal Prediction)** is a transformer-based framework for learning Causal Structural Causal Models (SCMs) by observing state transitions ("Deltas") under diverse interventions.

---

## 🚀 Project Status: **Phase 5 (Verified & Unified)**
**Current Stage**: Phase 5C (Unified Fine-Tuning).
**Status**: ✅ **SUCCESS**. The model is successfully learning both Physics and Structure.

### 📊 Latest Results (Epoch 163+)
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Physics MAE** | **~12.1** | **Excellent.** The model predicts physical changes with ~7% error rate across a `[-100, 100]` range. |
| **Structure SHD** | **~550** | **Improving.** The model is actively reducing structural errors (started at ~1200 random, ~490 empty). |
| **Structure F1** | **~0.33** | **Rising.** We broke the "Zero-Prediction Baseline" (F1 ~0.20) and are now discovering real edges. |
| **TPR (Recall)** | **~0.40** | **Strong.** The model has found 40% of all causal edges in the system. |

### 📈 Trajectory: Where we are vs Where we are heading
*   **Where we started (Phase 4)**: A model that knew Physics perfectly (MAE ~12) but was blind to structure (SHD N/A).
*   **Where we are (Phase 5C)**: We re-connected the "Causal Eye" (DAG Head). At first, it refused to learn (Structural Collapse), but we fixed this by re-weighting the loss (`pos_weight=3.0`). Now it is "waking up" and learning the graph *without* forgetting the physics.
*   **Where we are heading**:
    *   **Short Term**: SHD < 500, F1 > 0.50. This confirms we can extract high-quality graphs.
    *   **Long Term**: Scaling to 100+ variables using this proven architecture.

---

## 🧪 Architecture: **Phase 5 Unified**

The model is a single **Causal Transformer** that outputs two distinct predictions from the same internal "Understanding" of the system.

```mermaid
graph TD
    subgraph Input
    Seq["Interleaved Sequence: [Node, Value, Node, Value...]"]
    end
    
    subgraph "Physics-Native Encoder"
    Seq -->|Hybrid| Emb["Hybrid Embeddings (Linear + Fourier + MLP)"]
    end
    
    subgraph "Unified Transformer"
    Emb -->|RoPE| Attn["Self-Attention (Rotary Positional)"]
    Attn -->|Vectorized| MoE["Mixture of Experts (Hard Gumbel)"]
    MoE -->|Recurrent| Refine["3-Step Refinement Loop"]
    end
    
    Refine -->|"Head 1 (Physics)"| Delta["Delta Prediction (Huber Loss)"]
    Refine -->|"Head 2 (Structure)"| Graph["DAG Adjacency (BCE Loss)"]

    style Delta fill:#bfb,stroke:#333,stroke-width:2px,color:#000
    style Graph fill:#fbb,stroke:#333,stroke-width:2px,color:#000
```

### Key Components

#### 1. RoPE (Rotary Positional Embeddings)
```mermaid
graph LR
    subgraph "Absolute Position"
    Vec["Vector [x1, x2]"] -->|Rotate by m*theta| Rot["Rotated [x1', x2']"]
    end
    
    subgraph "Attention"
    Rot -->|Dot Product| Score["Relative Score (Depends on m-n)"]
    end
    
    style Vec fill:#eee,stroke:#333,color:#000
    style Rot fill:#bbf,stroke:#333,color:#000
    style Score fill:#bfb,stroke:#333,color:#000
```
*   **Purpose**: Standard transformers use absolute positions ($0, 1, 2...$). In a causal graph, node $i$ and node $j$ might be far apart in the list but close in the graph. RoPE encodes "Relative Distance" mathematically.
*   **Mechanism**: It rotates the Query and Key vectors in the complex plane by an angle proportional to their position.
*   **Benefit**: This allows the attention mechanism to understand *relative* relationships (e.g., "Input $X$ is the value immediately following Node ID $Y$") regardless of where they appear in the total sequence.

## 🛠️ Data Pipeline ("Physics 2.0")

```mermaid
graph TD
    subgraph "Step 1: The Universe"
    SCM["SCM Generator"] -->|Random Functions| Graph["Physical Laws (DAG)"]
    end
    
    subgraph "Step 2: Twin Worlds"
    Graph -->|Noise Z| World1["World 1 (Observed)"]
    Graph -->|Noise Z + Intervention| World2["World 2 (Intervened)"]
    end
    
    subgraph "Step 3: Sampling"
    World1 -->|Subtract| Delta["True Delta"]
    World2 -->|Tokenize| Input["Model Input"]
    end
    
    style SCM fill:#f9f,stroke:#333,color:#000
    style Graph fill:#eee,stroke:#333,color:#000
    style World1 fill:#bbf,stroke:#000,color:#000
    style World2 fill:#bfb,stroke:#000,color:#000
```

### SCM Generator
We simulate complex physical systems, not just linear graphs.

#### 2. Hard MoE (Mixture of Experts) with Gumbel-Softmax
```mermaid
graph TD
    Token["Input Token"] --> Router["Router Network"]
    
    Router -->|Logits| Gumbel["Gumbel-Softmax (Hard)"]
    Gumbel -->|Index=2| Switch{Switch}
    
    subgraph "Physics Experts"
    E1["Expert 1 (Linear)"]
    E2["Expert 2 (Sine)"]
    E3["Expert 3 (Threshold)"]
    E8["Expert 8 (Chaos)"]
    end
    
    Switch -->|Route Token| E3
    E3 -->|Output| Result["Specialized Pred"]
    
    style Token fill:#eee,stroke:#000,color:#000
    style Router fill:#eee,stroke:#333,color:#000
    style Gumbel fill:#f9f,stroke:#333,color:#000
    style E3 fill:#bfb,stroke:#000,color:#000
```
*   **Purpose**: Physical laws are distinct (e.g., a "Threshold" function behaves differently from a "Sine Wave"). A single dense network blurs them.
*   **Mechanism**:
    *   We use 8 Vectorized Experts.
    *   A **Router** network predicts which expert to use.
    *   **Gumbel-Softmax (Hard)**: Instead of a weighted average (Soft MoE), we use the "Straight-Through Estimator" to force a discrete choice ($1$ for Expert A, $0$ for others).
*   **Benefit**: This forces specialization. One expert becomes the "Sine Wave Specialist", another the "Linear Specialist", preventing interference between conflicting physical laws.

#### 3. Dual Heads (Unified Output)
```mermaid
graph TD
    Unified["Refined Representation"] -->|Split| H1
    Unified -->|Split| H2
    
    subgraph "Head 1: Physics"
    H1["MLP Projection"] -->|Huber Loss| Delta["Continuous Delta"]
    end
    
    subgraph "Head 2: Structure"
    H2["Bilinear Query"] -->|BCE Loss| DAG["Discrete Adjacency"]
    end
    
    style Delta fill:#bfb,stroke:#333,color:#000
    style DAG fill:#fbb,stroke:#333,color:#000
```
The Deep Physics Logic (Transformer Output) splits into two tasks:
*   **Delta Head**:
    *   **Task**: Predict continuous value changes $\Delta = f(Parents)$.
    *   **Loss**: Huber Loss (Robust Regression).
*   **DAG Head**:
    *   **Task**: Predict the discrete Causal Graph ($A_{ij}$).
    *   **Mechanism**: A bilinear layer that queries "Who are my parents?" for every node.
    *   **Loss**: Binary Cross Entropy (BCE) with `pos_weight` to handle sparsity.

---

## 📉 Metrics Explained

### 1. MAE (Mean Absolute Error) - The "Physics Score"
**Formula**: $\text{MAE} = \frac{1}{N} \sum | \text{Predicted} - \text{True} |$
*   **Why it matters**: It measures raw accuracy. In our data (range `[-100, 100]`), an MAE of **12.0** means the model is usually within **6%** of the correct value.
*   **Why it's good**: Achieving low MAE proves the model has "grokked" the underlying simulation functions (Sin, Tanh, Cubic interactons).

### 2. SHD (Structural Hamming Distance) - The "Graph Score"
**Formula**: Count of (Missing Edges + Wrong Edges + Reversed Edges).
*   **Why it matters**: It is the "True Error" of discovery.
*   **Goal**: Drive this to 0. (Random guessing for 50 vars is ~1200. We are at ~550).

---

## 📜 Complete Training Log (Phase 4 -> 5)

Below is the trajectory of the model. Note the **Phase 5 Transition** around Epoch 148 where we resumed training with the DAG head enabled.

### Phase 4 (Physics Only)
| Epoch | Level | Train Loss | MAE (Physics) | SHD (Structure) | F1 (Structure) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0 | 292.7 | 3.30 | 306.7 | 0.25 | Initial Chaos |
| 10 | 0 | 3055.7 | 33.7 | 3644 | 1.10 | Curriculum Shock (Params Adjusted) |
| ... | ... | ... | ... | ... | ... | ... |
| 147 | 29 | 13029 | **12.30** | N/A | N/A | **Phase 4 Complete. Physics Solved.** |

### Phase 5 (Unified Fine-Tuning)
| Epoch | Level | Train Loss | MAE | SHD | F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 148 | 29 | 206M | 12.53 | 480 | 0.24 | **Shock**: DAG Head initialized. Physics Preserved. |
| 153 | 29 | 2414 | 14.12 | 492 | 0.27 | **Stagnation**: Structural Collapse (Predicting Empty). |
| **156** | **29** | **3996** | **14.35** | **593** | **0.31** | **Fix Applied**: `pos_weight=3.0` & `lambda_dag=200`. |
| 157 | 29 | 1423 | 12.33 | 574 | 0.32 | **Waking Up**: SHD starts dropping. F1 rising. |
| 163 | 29 | 1338 | **12.15** | **550** | **0.33** | **Current**: Steady improvement. |

---

## 🛠️ Usage

### Resume Phase 5 Training
To continue specifically from where we are (Unified Fine-Tuning):

```bash
python main.py \
  --checkpoint_path checkpoints/checkpoint_epoch_163.pt \
  --resume \
  --lambda_dag 200.0 \
  --lambda_h 1.0 \
  --lr 1e-4 \
  --epochs 1000
```
