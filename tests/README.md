# Verification Tests

This folder contains scripts to verify the functionality of the `ISD-CP-Learning` codebase.

## `verify_config.py`

This script verifies that the model configuration flags working as intended.

**Checks:**
1.  **Model Depth**: Initializes `CausalTransformer` with `num_layers=24` and confirms the encoder has 24 layers.
2.  **Fixed Variables**: Initializes `SCMGenerator` and `CausalDataset` with fixed 50 variables and confirms batch shapes.
3.  **Forward Pass**: Runs a dummy batch through the deep model to ensure no runtime errors.

**Usage:**
```bash
python3 tests/verify_config.py
```
**Expected Output:**
```
✅ Depth Check Passed
✅ Variable Count Check Passed
✅ Forward Pass Passed
```
