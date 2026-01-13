# Critical Issues Found - Deep Code Audit (Session 3)

## Summary
**Total Issues Identified: 15** (Previously Fixed: 13, New: 2)

All 13 previously identified issues are confirmed fixed. However, two **CRITICAL NEW ISSUES** discovered that completely prevent structure learning and intervention awareness.

---

## 🔴 CRITICAL NEW ISSUES (Must Fix Immediately)

### **Issue #14: int_node_idx Parameter Unused**
**Severity**: HIGH  
**Location**: `src/models/CausalTransformer.py:382`  
**Description**:
- Function signature accepts `int_node_idx` parameter (which node was intervened)
- Parameter is **never used** in the forward pass
- Data pipeline correctly provides it via `collate_fn_pad` (stacked to shape `(B,S)`)
- **Impact**: Model has no intervention awareness. It treats all samples equally regardless of which node was intervened.

**Evidence**:
```python
# CausalTransformer.py:382
def forward(self, base_samples, int_samples, target_row, int_mask, int_node_idx=None, mcm_mask=None):
    # ... 80+ lines of code ...
    # int_node_idx is NEVER referenced anywhere
    # Should be: instr = self.int_embedding(int_node_idx)
    # (as shown in notebook ISD-CP.ipynb:965)
```

**Fix Strategy**:
1. Keep `int_node_idx` parameter
2. Create `int_embedding` layer in `__init__` (or use target_row-based encoding)
3. Use in `_forward_pass` to condition the model on intervened node

---

### **Issue #15: Dummy Zeros Override Structure Predictions**
**Severity**: CRITICAL  
**Location**: `src/models/CausalTransformer.py:463-468`  
**Description**:
- Model computes `logits_final` from Pass 3 (DAG head output)
- **Line 463 creates a dummy all-zero tensor** and overwrites it:
```python
dummy_adj = torch.zeros(B, N, N, device=base_samples.device)
# ... 
return deltas_final, logits_final, dummy_adj, None, total_aux  # Returns DUMMY, not logits_final!
```
- Result: Training receives all-zero structure predictions regardless of model output
- **Impact**: Structure learning is completely disabled. DAG head never receives learning signal.

**Why This Exists**:
- Code comment says "for API compatibility with main.py metrics"
- But it's using the WRONG variable name

**Current Code Flow** (BROKEN):
```
Pass 1: logits_1 → causal_mask
Pass 2: logits_2 → refined_mask  
Pass 3: logits_final ← Computed by DAG head ✓
Return: dummy_adj ← torch.zeros() ✗ (Wrong!)
```

**Correct Code Flow** (Should Be):
```
Return: logits_final ← Use actual predictions
```

**Fix**:
```python
# Line 463-468
if len(base_samples.shape) == 3:
    B, S, N = base_samples.shape
else:
    B, N = base_samples.shape

total_aux = aux_1 + aux_2 + aux_3

# REMOVE: dummy_adj = torch.zeros(...)
# CHANGE: Return logits_final directly
return deltas_final, logits_final, logits_final, None, total_aux
```

---

## ✅ PREVIOUSLY FIXED ISSUES (All Verified)

### **Issues 1-8** (First Audit - All Fixed)
1. ✅ No weight_decay in optimizer
2. ✅ Gradient clipping too tight (1.0 → 10.0)
3. ✅ DAG loss starved by default lambda_dag=0
4. ✅ Router not reinitialized on fresh start
5. ✅ pos_weight clamping too restrictive
6. ✅ Validation cache not respecting density changes
7. ✅ Scheduler only per-epoch (should be per-batch)
8. ✅ AdamW parameters not explicit

### **Issues 9-13** (Second Audit - All Fixed)
9. ✅ Missing dtype preservation in loss tensors
10. ✅ Router weights not reassigned after resume
11. ✅ No DDP synchronization for validation metrics
12. ✅ NaN loss replacement without gradient preservation
13. ✅ Validation loader cache not invalidated on curriculum change

---

## 📊 Impact Assessment

| Issue | Prevents Learning | Prevents Training | Must Fix First |
|-------|------------------|------------------|----------------|
| #14 (unused int_node_idx) | Yes - No intervention awareness | No | 2 |
| #15 (dummy zeros override) | Yes - Structure learning disabled | No | 1 |

**Training can proceed, but:**
- Model learns ONLY physics (delta predictions)
- Model learns ZERO structure (DAG predictions all-zero)
- Model is UNAWARE of which node was intervened

---

## 🔧 Recommended Fix Order

1. **Fix #15 FIRST** (1 line change): Return `logits_final` instead of `dummy_adj`
   - Enables structure learning immediately
   - Can be verified by checking loss gradients

2. **Fix #14 SECOND** (10-15 lines): Embed and use `int_node_idx`
   - Adds intervention awareness
   - Improves physics head conditioning

3. **Validate**: Run `python main.py --dry_run` to verify forward/backward pass

---

## 💾 Code Locations to Review

### int_node_idx Parameter Flow
- **Data Gen**: `src/data/CausalDataset.py:92` - Creates `int_node_idx = torch.argmax(int_mask)`
- **Collate**: `src/data/collate.py:47-48` - Stacks to `(B,S)` 
- **Model**: `src/models/CausalTransformer.py:382` - **Parameter accepted but ignored**
- **Training**: `main.py:485` - Passes to model but doesn't verify it's used

### Dummy Tensor Override
- **Issue**: `src/models/CausalTransformer.py:463` - Creates dummy zeros
- **Usage**: `main.py:487` - Receives dummy instead of predictions
- **Metrics**: `main.py:532-535` - Computes metrics on dummy zeros (always same)

---

## 🎯 Validation Checklist

After fixes:
- [ ] `logits_final` has non-zero values during training
- [ ] Structure loss (`loss_dag`) changes with batch
- [ ] Metrics (F1, SHD) improve over epochs (not stuck at constant)
- [ ] `int_node_idx` is used in forward pass (add debug print)
- [ ] Model parameters update with structure gradients

---

## 📝 Notes

- Both issues are **recent regressions** (from notebook ISD-CP.ipynb which uses int_embedding correctly)
- Issues #14 & #15 explain why model can't learn structure despite curriculum and loss functions
- Physics head (deltas) working correctly, only structure head broken
- DDP synchronization and other training fixes are solid

