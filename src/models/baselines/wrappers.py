import numpy as np
import networkx as nx
import torch
import time
from abc import ABC, abstractmethod

# Try importing causal-learn
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.cit import fisherz
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("Warning: causal-learn not found. PC and GES will fail if used.")

# Import ISD-CP components
from src.models.CausalTransformer import CausalTransformer
from src.data.encoder import InterleavedEncoder

class BaselineModel(ABC):
    @abstractmethod
    def fit(self, X, **kwargs):
        """
        Fit the model to data X. 
        X is (Samples, Nodes) numpy array.
        """
        pass
        
    @abstractmethod
    def predict_adj(self):
        """
        Return the learned adjacency matrix (N, N) as numpy array.
        """
        pass

class PCWrapper(BaselineModel):
    def __init__(self, alpha=0.05, max_k=2):
        self.alpha = alpha
        self.adj = None
        self.max_k = max_k
        
    def fit(self, X, **kwargs):
        if not CAUSAL_LEARN_AVAILABLE:
            raise ImportError("causal-learn is required for PC")
        
        # PC returns a CausalGraph object
        # fisherz is standard for continuous data
        cg = pc(X, self.alpha, fisherz, stable=True, uc_rule=0, uc_priority=-1, max_k=self.max_k)
        self.adj = cg.G.graph # extract graph structure
        
    def predict_adj(self):
        # Convert causal-learn graph to binary adj; treat undirected as bidirectional for SHD fairness
        return (self.adj != 0).astype(int)

class GESWrapper(BaselineModel):
    def __init__(self):
        self.adj = None
        
    def fit(self, X, **kwargs):
        if not CAUSAL_LEARN_AVAILABLE:
            raise ImportError("causal-learn is required for GES")
            
        record = ges(X)
        self.adj = record['G'].graph
        
    def predict_adj(self):
        return (self.adj != 0).astype(int)

class NotearsWrapper(BaselineModel):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.adj = None
        
    def fit(self, X, **kwargs):
        # Using the project's own reference implementation if available, 
        # or a simplified internal one.
        # Check src/models/baselines/notears.py
        from src.models.baselines.notears import NotearsLinear
        
        # Use linear for speed or nonlinear for fairness? 
        # Paper compares against NOTEARS-MLP (Nonlinear)
        # But standard NOTEARS is often Linear.
        # Let's try Nonlinear if available, else Linear.
        
        # Assumption: X is (Samples, Nodes)
        nt = NotearsLinear(d=X.shape[1], lambda1=0.1)
        self.adj = nt.fit(torch.tensor(X, dtype=torch.float32))
        # TODO: switch to notears_nonlinear wrapper if needed
        
    def predict_adj(self):
        return (self.adj != 0).astype(int)

class ISDCPWrapper(BaselineModel):
    """
    Wraps our Causal Transformer to be compatible with the benchmark runner.
    """
    def __init__(self, device='cpu', checkpoint_path=None, d_model=256):
        self.device = device
        self.model = None
        self.d_model = d_model
        
    def fit(self, X, **kwargs):
        """
        ISD-CP is typically pre-trained or trained on stream.
        For fair 'training from scratch' on dataset X:
        1. Initialize model
        2. Run training loop on X (treating X as the 'observational' world)
        
        HOWEVER, ISD-CP relies on *interventional* data for its full power.
        If X is purely observational, ISD-CP degenerates to standard Transformer.
        
        If X contains intervention metadata (kwargs), we use it.
        """
        num_vars = X.shape[1]
        self.model = CausalTransformer(num_nodes=num_vars + 5, d_model=self.d_model)
        self.model.to(self.device)
        self.model.train()
        
        # Simple training loop for benchmark (fine-tuning on this dataset)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Convert X to tensor
        data = torch.tensor(X, dtype=torch.float32).to(self.device)
        # X is (B, N). We need interleaved inputs?
        # Actually main.py handles data loading from SCM.
        
        # For this wrapper, let's assume we run a minimal loop to adapt to X
        # But realistically, benchmarking meant evaluating the *pretrained* model 
        # or training it fully.
        # If training fully, it takes time.
        pass # Placeholder for actual training logic integration
        
    def predict_adj(self):
        # Run one inference pass to get logits
        # Extract DAG logits
        # return binarized adj
        return np.zeros((10, 10)) # Placeholder
