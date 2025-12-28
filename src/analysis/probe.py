import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from rich.console import Console

console = Console()

class CausalProber:
    """
    Analyzes a trained Causal Transformer to see if it implicitly learns the DAG structure.
    Method: Train a linear classifier on pair-wise embeddings (h_i, h_j) -> Edge(i->j).
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.model.eval()
        self.device = device
        self.activations = {}
        
    def get_activations(self, batch):
        """
        Runs a forward pass and captures embeddings from the last transformer layer.
        """
        self.model.eval()
        with torch.no_grad():
            # Hooks are tricky with different models, let's rely on the model returning features if possible.
            # Most of our models have self.transformer or self.backbone.
            
            # Standard Interface for all our models:
            # x = encoder(...)
            # x = transformer(x)
            
            # We will manually run the encoder + transformer part here
            base = batch['base_samples'].to(self.device)
            int_s = batch['int_samples'].to(self.device)
            target = batch['target_row'].to(self.device)
            mask = batch['int_mask'].to(self.device)
            
            enc_out = self.model.encoder(base, int_s, target, mask)
            
            # Handle different model attributes
            if hasattr(self.model, 'transformer'):
                # Model A, B, C, F
                features = self.model.transformer(enc_out)
            elif hasattr(self.model, 'backbone'):
                # Model E
                features = self.model.backbone(enc_out)
            else:
                raise ValueError("Model structure unknown: cannot find transformer/backbone.")
                
            return features # (B, N, d_model)

    def collect_probing_dataset(self, dataloader, max_batches=50):
        """
        Runs the model on the dataloader and constructs a dataset of (Embedding_Pair, Edge_Label).
        X = [Concat(h_i, h_j)]
        y = [A_ij]
        """
        X_list = []
        y_list = []
        
        console.print(f"[bold blue]Collecting probing data from {max_batches} batches...[/bold blue]")
        
        for i, batch in enumerate(dataloader):
            if i >= max_batches: break
            
            # Get Embeddings associated with these graphs
            emb = self.get_activations(batch) # (B, N, D)
            
            # Get True Adjacency
            adj = batch['adj'].to(self.device) # (B, N, N)
            
            B, N, D = emb.shape
            
            # Create pairs (i, j)
            # We want to predict if i -> j
            # Input: concat(h_i, h_j)
            
            # Broadcast to create pairs efficiently?
            # Lets loop for safety and clarity first, optimization later if needed.
            # Actually, we can use torch combination.
            
            # emb_i: (B, N, 1, D) -> repeat to (B, N, N, D)
            emb_i = emb.unsqueeze(2).expand(B, N, N, D)
            # emb_j: (B, 1, N, D) -> repeat to (B, N, N, D)
            emb_j = emb.unsqueeze(1).expand(B, N, N, D)
            
            # Concat: (B, N, N, 2D)
            pairs = torch.cat([emb_i, emb_j], dim=-1)
            
            # Flatten to (B*N*N, 2D)
            X_batch = pairs.reshape(-1, 2 * D)
            y_batch = adj.reshape(-1)
            
            X_list.append(X_batch.cpu())
            y_list.append(y_batch.cpu())
            
        return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

    def train_probe(self, train_loader, val_loader):
        """
        Trains a Logistic Regression probe.
        """
        # 1. Collect Data
        X_train, y_train = self.collect_probing_dataset(train_loader, max_batches=20)
        X_val, y_val = self.collect_probing_dataset(val_loader, max_batches=10)
        
        console.print(f"Train Size: {X_train.shape[0]} pairs. Val Size: {X_val.shape[0]} pairs.")
        
        # 2. Define Probe (Simple Linear Layer)
        input_dim = X_train.shape[1]
        probe = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        # 3. Train Loop
        probe_batch_size = 4096
        probe_dataset = TensorDataset(X_train, y_train)
        probe_loader = DataLoader(probe_dataset, batch_size=probe_batch_size, shuffle=True)
        
        console.print("[bold green]Training Linear Probe...[/bold green]")
        for epoch in range(5):
            probe.train()
            total_loss = 0
            for bx, by in probe_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = probe(bx).squeeze()
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Val Acc
            probe.eval()
            with torch.no_grad():
                vx, vy = X_val.to(self.device), y_val.to(self.device)
                v_pred = probe(vx).squeeze()
                auc = roc_auc_score(vy.cpu().numpy(), v_pred.cpu().numpy())
                
            console.print(f"Epoch {epoch+1}: Loss {total_loss:.4f} | Val AUC: {auc:.4f}")
            
        return probe, auc
