# ISD-CP Unified: Hyper-Experimental Causal Discovery

**ISD-CP Unified** is a next-generation causal discovery framework designed to learn causal graphs and intervention effects from large-scale synthetic data.

Unlike traditional methods that treat interventions as passive data tokens, this project introduces **Hyper-Interventional Experts**, a novel architecture where the model's weights are dynamically re-programmed by a Hyper-Network based on the intervention target.

## 🌟 Key Research Novelty

### 1. Hyper-Interventional Experts (Model F)
Standard Transformers treat "Intervention on Node X" as just another input feature. We argue an intervention is a **System Modification**.
*   **Our Approach**: We use a **Hyper-Network** that takes the intervention target ID and generates modulation weights for the final prediction layer.
*   **Result**: The model effectively "swaps out" its physical laws to match the intervened reality, providing a stronger inductive bias for perfect interventions.

### 2. "Twin World" Variance Reduction
Learning $\Delta = Y_{do} - Y_{obs}$ is noisy because standard generators use different $\epsilon$ for $Y_{do}$ and $Y_{obs}$.
*   **Our Approach**: We generate the noise matrix $\epsilon$ *once*.
*   **Result**: The noise term cancels out in the subtraction, leaving a pure causal signal. Variance drops to near zero, speeding up convergence by orders of magnitude.

### 3. Interleaved Tokenization
Instead of one vector per variable, we use the TabPFN approach:
*   Sequence: `[Feature_ID_0, Value_0, Feature_ID_1, Value_1, ...]`
*   **Feature Tokens**: Learn structural identity and adjacency (Graph Discovery).
*   **Value Tokens**: Learn immediate state and delta predictions (Effect Prediction).

---

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Experiment (Single Command)
This command handles everything: data generation, curriculum learning, and multi-GPU distribution.

```bash
# Run on a single GPU (or CPU)
python main.py --epochs 1000 --max_vars 50

# Run on Multiple GPUs (DDP) - e.g., 4x A100
torchrun --nproc_per_node=4 main.py --epochs 5000 --max_vars 100
```

### Options
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--min_vars` | 20 | Starting number of variables (Curriculum Level 0) |
| `--max_vars` | 50 | Maximum variables to scale up to |
| `--batch_size` | 32 | Batch size per GPU |
| `--dry_run` | False | Run a single step to verify pipeline |

---

## 📂 Project Structure

*   `src/models/CausalTransformer.py`: **The Core Model**. Contains the Interleaved Encoder, Hyper-Network, and Expert Heads.
*   `src/data/SCMGenerator.py`: **The Engine**. Generates random SCMs and implements "Twin World" noise logic.
*   `src/data/CausalDataset.py`: **The Feeder**. Infinite iterable dataset that yields paired (Base, Twin) samples.
*   `src/training/curriculum.py`: **The Coach**. Manages difficulty scaling (Vertices, Density, Ranges) based on validation MAE.
*   `src/analysis/probe.py`: **The Auditor**. Linear probes to check if the model is learning implicit DAG structures.

## 🔮 Future Work
*   **Steering**: Actively maximizing Probe Accuracy during training to force standard transformers to learn structural representations.
*   **Real Data**: Adapting the Hyper-Expert fine-tuning for real-world gene knockout datasets.
