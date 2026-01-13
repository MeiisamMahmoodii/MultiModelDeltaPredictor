import os
import sys
import time
import numpy as np
import torch
import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod

# Attempt to import optional dependencies
try:
    import avici
    AVICI_AVAILABLE = True
except Exception as e:
    print(f"DEBUG: AVICI import failed: {e}")
    AVICI_AVAILABLE = False

try:
    from gears import GEARS
    GEARS_AVAILABLE = True
except Exception as e:
    print(f"DEBUG: GEARS import failed: {e}")
    GEARS_AVAILABLE = False

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except Exception as e:
    print(f"DEBUG: TabPFN Regressor import failed: {e}")
    try:
        from tabpfn import TabPFNClassifier
        # Fallback if Regressor not found, though we need regression for Delta
        TABPFN_AVAILABLE = "Classifier"
    except Exception as e2:
        print(f"DEBUG: TabPFN Classifier import failed: {e2}")
        TABPFN_AVAILABLE = False

# We assume CausalFormer is a local import or cloned repo. 
# We'll check for it in the wrapper.

# ISD-CP imports
try:
    from src.models.CausalTransformer import CausalTransformer
    from src.data.SCMGenerator import SCMGenerator # For context if needed
    ISDCP_AVAILABLE = True
except Exception:
    ISDCP_AVAILABLE = False

# Try importing causal-learn for PC
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    CAUSAL_LEARN_AVAILABLE = True
except Exception:
    CAUSAL_LEARN_AVAILABLE = False
    
# Import Notears (Assuming internal implementation exists as per old wrappers)
try:
    from src.models.baselines.notears import NotearsLinear
    NOTEARS_AVAILABLE = True
except Exception as e:
    try:
        # Fallback for benchmark suite path variance
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
        from src.models.baselines.notears import NotearsLinear
        NOTEARS_AVAILABLE = True
    except Exception as e2:
        print(f"DEBUG: NOTEARS import failed: {e2}")
        NOTEARS_AVAILABLE = False

class ModelWrapper(ABC):
    def __init__(self, device='cpu', **kwargs):
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, X, metadata=None):
        """
        Fit the model to observational data X (and optionally metadata like known graph).
        X: (num_samples, num_vars) numpy array or tensor.
        metadata: Dict containing optional info (e.g. 'dag' for GEARS).
        """
        pass

    @abstractmethod
    def predict_structure(self):
        """
        Return predicted adjacency matrix (num_vars, num_vars).
        Returns None if model doesn't predict structure.
        """
        pass

    @abstractmethod
    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        """
        Predict the delta vector for a single intervention.
        intervention_idx: Index of the intervened variable.
        intervention_val: Value of the intervention.
        base_sample: The pre-intervention state (1, num_vars).
        
        Returns: 
            delta: (1, num_vars)
            extra_info: dict (e.g. 'logits' for structure)
        Returns (None, {}) if model doesn't predict deltas.
        """
        pass

class ISDCPWrapper(ModelWrapper):
    def __init__(self, checkpoint_path, device='cpu', **kwargs):
        super().__init__(device, **kwargs)
        if not ISDCP_AVAILABLE:
            raise ImportError("ISD-CP src modules not found.")
            
        self.model = None
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path):
        print(f"Loading ISD-CP from {path}...")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        args = ckpt["args"]
        self.model = CausalTransformer(
            num_nodes=getattr(args, 'max_vars', 100) + 5,
            d_model=getattr(args, 'd_model', 512),
            num_layers=getattr(args, 'num_layers', 6),
            grad_checkpoint=getattr(args, "grad_checkpoint", False),
            ablation_dense=getattr(args, "ablation_dense_moe", False),
            ablation_no_interleaved=getattr(args, "ablation_no_interleaved", False),
            ablation_no_dag=getattr(args, "ablation_no_dag", False),
            ablation_no_physics=getattr(args, "ablation_no_physics", False),
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.to(self.device).eval()

    def fit(self, X, metadata=None):
        pass

    def predict_structure(self):
        # ISD-CP predicts structure during the forward pass of interventions.
        # We rely on predict_delta returning logits to the harness.
        return None

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        # Input construction matching full_eval_static.py logic
        # base_sample: (1, N) tensor or array
        # intervention_val: Ignored by model (current limitation/design)
        
        base_tensor = torch.tensor(base_sample, dtype=torch.float32).to(self.device)
        if base_tensor.dim() == 1:
            base_tensor = base_tensor.unsqueeze(0)
            
        B, N = base_tensor.shape
        
        # Construct inputs
        # int_samples: In full_eval_static, it passed "Ground Truth Intervened Sample".
        # If we are PREDICTING, we don't have that.
        # However, we DO know the intervention: X_k = v.
        # And we know X_others = X_obs (Twin World assumption: Noise is shared).
        # So we can construct a "Hypothetical Intervened Input" where X_k = v, others = X_obs.
        # This is a fair input for the model if it used it.
        # Even if currently unused by Encoder, we pass it for correctness/future-proofing.
        
        target_row = base_tensor.clone() # Model uses this as "Values"
        
        # Construct int_samples (Best Guess)
        int_samples = base_tensor.clone()
        int_samples[:, intervention_idx] = float(intervention_val)
        
        # Mask
        mask = torch.zeros(B, N, device=self.device)
        mask[:, intervention_idx] = 1.0
        
        # Scalar index
        int_idx = torch.tensor([intervention_idx], device=self.device)
        
        with torch.no_grad():
            # Unpack 5 values
            deltas_pred, logits, _, _, _ = self.model(base_tensor, int_samples, target_row, mask)
            
        return deltas_pred.cpu().numpy(), {"logits": logits.cpu().numpy()}

class AVICIWrapper(ModelWrapper):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(device, **kwargs)
        if not AVICI_AVAILABLE:
            raise ImportError("avici package not found. Please install via pip.")
        self.model = avici.load_pretrained(download="scm-v0") # Using default or specified model
        self.model.to(device)

    def fit(self, X, metadata=None):
        self.X = X # Keep data for prediction

    def predict_structure(self):
        # AVICI predicts (B, N, N) edge probs from (B, N, d) data?
        # Or (1, N, N) from (N_samples, N_vars).
        # usage: model(x) -> logits
        x = torch.tensor(self.X, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, Samples, Vars)
        with torch.no_grad():
            logits = self.model(x) # (1, Vars, Vars)
        return torch.sigmoid(logits)[0].cpu().numpy()

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        return None, {} # AVICI doesn't predict deltas

class GEARSWrapper(ModelWrapper):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(device, **kwargs)
        if not GEARS_AVAILABLE:
            raise ImportError("gears package not found.")
        # GEARS requires a PertData object at init. 
        # Since we don't have one for generic SCMs easily, we defer/skip.
        self.gears_model = None
        self.pert_data = None
        
    def fit(self, X, metadata=None):
        # If metadata contains 'pert_data', we can init
        if metadata and 'pert_data' in metadata:
            self.pert_data = metadata['pert_data']
            # Initialize GEARS here
            self.gears_model = GEARS(self.pert_data, device=self.device)
        else:
            # Cannot run GEARS without specialized data structure
            pass

    def predict_structure(self):
        return None

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        if self.gears_model is None:
            return None, {}
        # Placeholder if we ever get it running
        return np.zeros_like(base_sample), {}

class TabPFNWrapper(ModelWrapper):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(device, **kwargs)
        if not TABPFN_AVAILABLE:
             raise ImportError("tabpfn package not found.")
        
        self.regressor = TabPFNRegressor(device=device)

    def fit(self, X, metadata=None):
        # Store X for context if needed
        self.X_train = X

    def predict_structure(self):
        return None

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        # Placeholder for TabPFN prediction
        return np.zeros_like(base_sample), {}

class PCWrapper(ModelWrapper):
    def __init__(self, device='cpu', alpha=0.05, **kwargs):
        super().__init__(device, **kwargs)
        if not CAUSAL_LEARN_AVAILABLE:
            raise ImportError("causal-learn package not found.")
        self.alpha = alpha
        self.adj = None

    def fit(self, X, metadata=None):
        # PC runs on X
        # X is (Samples, Nodes)
        cg = pc(X, self.alpha, fisherz, stable=True, uc_rule=0, uc_priority=-1)
        self.adj = cg.G.graph
        
    def predict_structure(self):
        if self.adj is None: return None
        # Convert to binary
        return (self.adj != 0).astype(int)

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        return None, {}

class NotearsWrapper(ModelWrapper):
    def __init__(self, device='cpu', model_type="linear", **kwargs):
        super().__init__(device, **kwargs)
        if not NOTEARS_AVAILABLE:
             # Fallback: Check if we can import from src?
             # If wrapper was already checking src, then it's fine.
             # If not found, maybe copy minimal implementation?
             # For now, raise error.
             raise ImportError("src.models.baselines.notears not found. Please ensure code exists.")
        self.model_type = model_type
        self.adj = None
        self.dims = 0

    def fit(self, X, metadata=None):
        self.dims = X.shape[1]
        # NotearsLinear expects torch tensor or numpy?
        # Typically numpy in official repo, but our internal might be torch.
        # Let's assume our internal 'NotearsLinear' mirrors the standard one.
        ctr = NotearsLinear(d=self.dims, lambda1=0.1)
        # fit returns W
        # X should be numpy usually, but NotearsLinear expects Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.adj = ctr.fit(X_tensor) 
        
    def predict_structure(self):
        if self.adj is None: return None
        # Thresholding usually needed for continuous W
        # Standard threshold 0.3?
        return (np.abs(self.adj) > 0.3).astype(int)

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        return None, {}

class OracleWrapper(ModelWrapper):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(device, **kwargs)
        self.true_dag = None
        self.pipe = None

    def fit(self, X, metadata=None):
        # Oracle cheats: it looks at metadata['dag']
        if metadata and 'dag' in metadata:
            self.true_dag = metadata['dag'] # NetworkX graph or adj
        if metadata and 'pipe' in metadata:
            self.pipe = metadata['pipe'] 

    def predict_structure(self):
        if self.true_dag is None:
            return None
        return nx.to_numpy_array(self.true_dag, dtype=int)

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        # Oracle predicts delta perfectly?
        # If we have access to the Ground Truth Delta in "metadata" (cheating via harness)
        # Or if we can compute it.
        # Simplest: The harness passes 'true_delta' in kwargs for Oracle?
        # Or we rely on harness to inject it.
        # Let's try to find 'true_delta' in kwargs if harness passes it.
        if 'true_delta' in kwargs:
            return kwargs['true_delta'].cpu().numpy(), {}
        return None, {}

try:
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class StructureRegressorWrapper(ModelWrapper):
    """
    Two-Stage Model:
    1. Structure Model (PC, NOTEARS) -> Predicts DAG
    2. Regressor (Ridge, MLP) -> Fits X_parents -> X_child
    
    Eliminates N/A for Delta Prediction in Structure Models.
    """
    def __init__(self, structure_model_class, regressor_type='ridge', device='cpu', **kwargs):
        super().__init__(device, **kwargs)
        # Instantiate the structure learning model
        self.structure_model = structure_model_class(device=device, **kwargs)
        self.regressor_type = regressor_type
        self.regressors = {} # Map node_idx -> fitted_regressor
        self.adj = None
        self.X_train = None

    def fit(self, X, metadata=None):
        # 1. Fit Structure
        self.structure_model.fit(X, metadata)
        self.adj = self.structure_model.predict_structure() # (N, N) matrix
        self.X_train = X

        if self.adj is None:
            return 
        
        # 2. Fit Local Regressors
        N = X.shape[1]
        for i in range(N):
            parents = np.where(self.adj[:, i] != 0)[0]
            if len(parents) == 0:
                # No parents, fit mean? or intercept?
                # Regressor on empty input might fail, handle simpler
                self.regressors[i] = "mean"
                continue
                
            X_parents = X[:, parents]
            y_child = X[:, i]
            
            if self.regressor_type == 'ridge':
                reg = Ridge(alpha=1.0)
            elif self.regressor_type == 'mlp':
                reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
            else:
                reg = Ridge()
                
            reg.fit(X_parents, y_child)
            self.regressors[i] = reg

    def predict_structure(self):
        return self.structure_model.predict_structure()

    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        # Base sample: (B, N) or (N,)
        # We need to handle batch.
        
        if self.adj is None or self.X_train is None:
            return None, {}
            
        # Ensure B, N shape
        base_tensor = np.array(base_sample)
        if base_tensor.ndim == 1:
            base_tensor = base_tensor[None, :] # (1, N)
            
        B, N = base_tensor.shape
        
        # 1. Intervened State Construction
        try:
            dag_nx = nx.from_numpy_array(self.adj, create_using=nx.DiGraph)
            topo_order = list(nx.topological_sort(dag_nx))
        except nx.NetworkXUnfeasible:
            # Cycles detected
            topo_order = list(range(N))
            
        # Initialize current state with base samples
        current_state = base_tensor.copy()
        
        # Apply Intervention
        # Set column intervention_idx to intervention_val
        current_state[:, intervention_idx] = float(intervention_val)
        
        # Propagate changes in topological order
        # For each node, if it's not the intervened one, predict its new value based on new parents
        # Delta = New_Pred - Old_Pred ???
        # Or New_Value = Mechanism(New_Parents) + Noise
        # Assume Additive Noise: X = f(Pa) + U
        # U = X_old - f(Pa_old)
        # X_new = f(Pa_new) + U
        #       = f(Pa_new) + X_old - f(Pa_old)
        #       = X_old + (f(Pa_new) - f(Pa_old))
        # This preserves the noise term U derived from the observational sample.
        
        for node in topo_order:
            if node == intervention_idx:
                continue # This is clamped
            
            reg = self.regressors.get(node)
            if reg == "mean" or reg is None:
                continue
            
            parents = np.where(self.adj[:, node] != 0)[0]
            if len(parents) == 0:
                continue
                
            # Current values of parents (updated by previous steps in topo sort)
            X_pa_new = current_state[:, parents] # (B, num_parents)
            
            # Original values of parents (from base sample)
            # Wait, standard cyclic updates depending on graph? 
            # In a DAG, X_new depends on new parents.
            # U depends on OLD parents.
            X_pa_old = base_tensor[:, parents] # (B, num_parents)
            
            # Predict
            pred_new = reg.predict(X_pa_new) # (B,)
            pred_old = reg.predict(X_pa_old) # (B,)
            
            # Mechanism Delta
            mech_delta = pred_new - pred_old
            
            # Update state with causal effect
            current_state[:, node] = base_tensor[:, node] + mech_delta
            # Note: We use base_tensor[:, node] as the baseline to add delta to.
            # BUT, if 'node' is a child of another node that changed, 'current_state' might have already updated?
            # No. 'current_state' accumulates updates.
            # Wait.
            # X_new = X_old + Delta.
            # 'current_state' should track X_new.
            # If we update 'current_state[:, node]', subsequent children will use this new value.
            # Correct.
            
            # So: X_new = X_old + (f(Pa_new) - f(Pa_old))
            # current_state[:, node] = base_tensor[:, node] + (pred_new - pred_old)
            # Is base_tensor[:, node] correct? Yes, that's X_old (observational).
            
            # Update in place so children see it
            current_state[:, node] = base_tensor[:, node] + mech_delta
                
        delta = current_state - base_tensor
        return delta, {}

class FeatureImportanceWrapper(ModelWrapper):
    """
    Infers structure from a Delta Predictor (TabPFN) via Feature Importance.
    Eliminates N/A for Structure in Delta Models.
    """
    def __init__(self, delta_model_class, device='cpu', **kwargs):
        super().__init__(device, **kwargs)
        self.delta_model = delta_model_class(device=device, **kwargs)
        self.adj = None
        
    def fit(self, X, metadata=None):
        self.delta_model.fit(X, metadata)
        
        # Use simple correlation/mutual information as a proxy?
        # Or train a quick Random Forest to explain X[:, i] using X[:, -i]?
        # Since TabPFN is black-box, we can't inspect it easily.
        # Using a surrogate Random Forest is standard interpretation technique.
        
        from sklearn.ensemble import RandomForestRegressor
        N = X.shape[1]
        self.adj = np.zeros((N, N))
        
        for i in range(N):
            # Target: Node i
            # Features: All other nodes
            y = X[:, i]
            X_feats = np.delete(X, i, axis=1)
            
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1)
            rf.fit(X_feats, y)
            
            # Feature Importances
            imps = rf.feature_importances_
            
            # Map back to original indices
            feat_idx = 0
            for j in range(N):
                if i == j: continue
                # Apply threshold?
                if imps[feat_idx] > 0.05: # Arbitrary threshold for "Edge"
                    self.adj[j, i] = 1 # Edge j -> i
                feat_idx += 1
                
    def predict_structure(self):
        return self.adj
        
    def predict_delta(self, intervention_idx, intervention_val, base_sample, **kwargs):
        return self.delta_model.predict_delta(intervention_idx, intervention_val, base_sample, **kwargs)

