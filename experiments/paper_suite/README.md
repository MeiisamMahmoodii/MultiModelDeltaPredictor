# ISD-CP Benchmark Suite

This suite benchmarks ISD-CP against AVICI, CausalFormer, GEARS, and TabPFN.

## Structure

- `run_suite.py`: Main harness. Generates data, runs models, saves results.
- `wrappers.py`: Adapters for each model.
- `report.py`: Generates markdown summary from results CSV.
- `results/`: Output directory.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install causal-learn  # For PC
   pip install avici         # For AVICI
   pip install gears-cgl     # For GEARS (or clone repo)
   pip install tabpfn        # For TabPFN
   ```

2. **Checkpoints**:
   - Ensure ISD-CP checkpoint is at `checkpoints/checkpoint_epoch_253.pt`.
   - For other models, provide paths if using pretrained checkpoints (default wrappers might assume pip package loading).

## Usage

Run the full suite:

```bash
python experiments/paper_suite/run_suite.py \
    --sizes 10,20,30,40,50 \
    --num_graphs 5 \
    --run_isdcp \
    --run_avici \
    --run_gears \
    --run_tabpfn \
    --device cuda
```

Generate Report:

```bash
python experiments/paper_suite/report.py --results_dir experiments/paper_suite/results
```

## Adding New Models

Modify `wrappers.py` to add a new class inheriting from `ModelWrapper`, and update `run_suite.py` to instantiate it.
