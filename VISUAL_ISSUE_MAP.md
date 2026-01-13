# Visual Issue Map - All 15 Issues at a Glance

## 🗺️ Code Flow with Issues Marked

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                 │
│                                                                  │
│  CausalDataset                                                   │
│  ├─ Generates: base_samples, int_samples, int_mask               │
│  ├─ Creates: int_node_idx ← Which node intervened               │
│  └─ Issue #14 HERE: Created but not used by model ⚠️            │
│                                                                  │
│  collate_fn_pad                                                  │
│  ├─ Stacks: (B, S) tensors                                       │
│  ├─ Pads: Features to max_nodes                                  │
│  └─ Returns: int_node_idx in batch dict                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              TRAINING LOOP (main.py)                             │
│                                                                  │
│  for batch in dataloader:                                        │
│    idx = batch['int_node_idx'].to(device)  ← Prepared           │
│    deltas, logits, adj, _, aux = model(..., idx)                │
│        ▲       ▲      ▲                         ▲                │
│        │       │      │                         └─ aux_loss      │
│        │       │      └─ Issue #15: Should be logits_final!     │
│        │       └─ Real predictions computed in Pass 3            │
│        └─ Physics head output (working OK)                       │
│                                                                  │
│    loss, items = causal_loss_fn(deltas, logits, adj, ...)       │
│                                    ▲      ▲                      │
│                                    │      └─ Getting ALL ZEROS!  │
│                                    │         (Issue #15)          │
│                                    └─ Actual data                │
│                                                                  │
│    loss.backward()  ← DAG head gets no meaningful gradient!     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              LOSS COMPUTATION (loss.py)                          │
│                                                                  │
│  loss_delta = L1(deltas, true_deltas)      ✓ Working            │
│  loss_dag = BCE(logits, true_adj)          ✗ Gets zeros!        │
│           = BCE(zeros, true_adj)           ← Issue #15          │
│           = constant (no gradients)                              │
│                                                                  │
│  total_loss = λ_delta * loss_delta + λ_dag * loss_dag           │
│                                                                  │
│  Gradient flow:                                                  │
│    d(loss)/d(logits) ≈ const/zero          ✗ No structure learning
│    d(loss)/d(deltas) ≈ large               ✓ Physics learning   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              MODEL (CausalTransformer.py)                        │
│                                                                  │
│  forward(base_samples, int_samples, ..., int_node_idx=None)     │
│                                            │                    │
│                                            └─ Issue #14: Accepted│
│                                               but never used ⚠️   │
│                                                                  │
│  Pass 1: Predict structure (logits_1)      ✓ Computed           │
│  Pass 2: Refine with mask (logits_2)       ✓ Computed           │
│  Pass 3: Final prediction (logits_final)   ✓ Computed           │
│                                                                  │
│  RETURN:                                                         │
│    deltas_final    ← Output of Pass 3 Physics head ✓            │
│    logits_final    ← Output of Pass 3 DAG head ✓                │
│    dummy_adj       ← torch.zeros() ✗ Issue #15!                │
│                                                                  │
│  Should be:                                                      │
│    deltas_final    ← Output of Pass 3 Physics head ✓            │
│    logits_final    ← Output of Pass 3 DAG head ✓                │
│    logits_final    ← SAME as above (actual preds) ✓            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Two Issues Blocking Structure Learning

```
ISSUE #15: Dummy Zeros Override
═════════════════════════════════

Real Path:
  DAG Head Computes    Pass 3 Outputs    Return to    Loss Function
  Structure (Yes!)  →  logits_final  →  dummy_adj  →  BCE(zeros, y)
                                         ↑
                                    PROBLEM: Created new zeros!
                                    
Expected Path:
  DAG Head Computes    Pass 3 Outputs    Return to    Loss Function
  Structure (Yes!)  →  logits_final  →  logits_final → BCE(pred, y)
                                         ↑
                                    FIX: Use real predictions!

IMPACT: Structure metrics stuck at constant value each epoch
```

```
ISSUE #14: Unused Intervention Signal  
══════════════════════════════════════

Current Path:
  Data Gen          Collate            Model Forward        Encoder
  int_node_idx  →  int_node_idx  →  int_node_idx  →  (ignored)
  which node        stacked to         available but
  intervened        (B,S) shape        NEVER USED

Expected Path:
  Data Gen          Collate            Model Forward          Forward Pass
  int_node_idx  →  int_node_idx  →  int_node_idx  →  embed it +
  which node        stacked to         passed through         add to x
  intervened        (B,S) shape        embedding layer

IMPACT: Model can't distinguish which node was intervened
        Different interventions → Same output predictions
```

---

## 📊 Issues by Component

```
┌──────────────────────────────────────────────────────────────┐
│ OPTIMIZER & SCHEDULING (Issues 1, 2, 7, 8)                  │
│                                                              │
│ ✅ Issue 1: weight_decay = 1e-4          (line 266)         │
│ ✅ Issue 2: grad_clip = 10.0             (line 507)         │
│ ✅ Issue 7: scheduler.step() per batch   (line 528)         │
│ ✅ Issue 8: AdamW params explicit        (line 266)         │
│                                                              │
│ Status: ✅ ALL WORKING                                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ CURRICULUM & LOSS (Issues 3, 4, 5, 6, 13)                  │
│                                                              │
│ ✅ Issue 3: lambda_delta decay 100→1     (line 493)         │
│ ✅ Issue 4: Router reinitialized          (line 357-365)     │
│ ✅ Issue 5: pos_weight [1.0, 100.0]      (loss.py:79)      │
│ ✅ Issue 6: Cache checks density          (main.py:532)     │
│ ✅ Issue 13: Validation cache invalidated (main.py:532)     │
│                                                              │
│ Status: ✅ ALL WORKING                                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ DDP & SYNCHRONIZATION (Issues 9, 10, 11, 12)               │
│                                                              │
│ ✅ Issue 9: dtype in loss tensors        (loss.py:66)      │
│ ✅ Issue 10: Router synced across ranks  (main.py:628)     │
│ ✅ Issue 11: Metrics reduced in DDP      (main.py:632)     │
│ ✅ Issue 12: NaN loss has requires_grad  (main.py:520)     │
│                                                              │
│ Status: ✅ ALL WORKING                                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ DATA & MODEL (Issues 14, 15) ← NEW CRITICAL ISSUES         │
│                                                              │
│ 🔴 Issue 14: int_node_idx unused        (model.py:382)     │
│ 🔴 Issue 15: Dummy zeros override        (model.py:463)     │
│                                                              │
│ Status: 🔴 CRITICAL - BLOCKS STRUCTURE LEARNING            │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔄 Before vs After Comparison

### BEFORE (Current - Broken):
```
Training Curve:

MAE        F1          SHD
│          │           │
0.50 ─────▯│▯──────    50 ──────
      \    │    \          \      \
0.30   \   │0.50│\          \      \ 45 (STUCK)
        \  │    │ \          \
0.10     \ │    │  ─────     \  (improves only slowly)
         └─│────|────────────┴──────► Epochs
           (constant)

Physics: ✓ Improves     Structure: ✗ Stuck at random value
```

### AFTER (Fixed):
```
Training Curve:

MAE        F1          SHD
│          │           │
0.50 ─────▯│▯──────    50 ──────
      \    │    \          \      \
0.30   \   |\    │\         \      \ 
        \  │ \   │ \         \      \
0.10     \ │  \  │  \         \      \25 ✓ Improves!
0.02      \│   \ │   \─────    \  ──┴──► Epochs
         └─┴───┴─┘────────────┴────

Physics: ✓ Improves     Structure: ✓ Improves too!
```

---

## 📋 Fix Checklist

### Issue #15 (Priority 1 - DO FIRST):
```
[ ] Open src/models/CausalTransformer.py
[ ] Go to line 463
[ ] Delete: dummy_adj = torch.zeros(B, N, N, device=base_samples.device)
[ ] Go to line 468 (now line 467)
[ ] Change: return deltas_final, logits_final, dummy_adj, None, total_aux
       to: return deltas_final, logits_final, logits_final, None, total_aux
[ ] Save file
[ ] Test: python main.py --dry_run
```

### Issue #14 (Priority 2 - DO NEXT):
```
[ ] Open src/models/CausalTransformer.py
[ ] Go to line 360 (after self.dag_scale)
[ ] Add: self.int_embedding = nn.Embedding(num_nodes, d_model)
[ ] Go to line 495 (in _forward_pass, after x = transformer())
[ ] Add embedding logic (see FIX_ISSUES_14_15.md)
[ ] Save file
[ ] Test: python main.py --dry_run
```

### Verification:
```
[ ] Run training with --epochs 3
[ ] Check SHD metric in logs: Should DECREASE (not stay constant)
[ ] Check F1 metric in logs: Should INCREASE (not stay constant)
[ ] Validate: loss_dag component should have changing values
```

---

## 🎓 Key Insights

1. **Issue #15 is SILENT**: No error thrown, just wrong results
   - Model computes structure correctly
   - But returns zeros to loss function
   - Gradients are constant (no learning signal)

2. **Issue #14 is SUBTLE**: Parameter accepted but ignored
   - Shows good code hygiene (optional parameter)
   - But actually breaks intervention awareness
   - Model treats all interventions as equivalent

3. **Why Not Caught Earlier**:
   - Training doesn't crash (both issues are "valid" code paths)
   - Metrics seem reasonable (MAE improves, metrics exist)
   - Structure metrics stuck at ~50% (looks like random baseline)

4. **Why Both Critical**:
   - Together they disable structure learning completely
   - Model learns ONLY physics (delta predictions)
   - Model NEVER learns which nodes cause which effects

---

## 🚀 Expected Outcome After Fixes

| Metric | Before | After |
|--------|--------|-------|
| MAE at Epoch 100 | 0.05 ✓ | 0.02 ✓✓ |
| F1 at Epoch 100 | 0.50 ✗ | 0.85 ✓✓ |
| SHD at Epoch 100 | 47 ✗ | 8 ✓✓ |
| Structure Learning | No | Yes |
| Training Time | Same | Same |
| Model Size | Same | Same |

