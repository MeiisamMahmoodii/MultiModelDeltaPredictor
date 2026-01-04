import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from rich.console import Console
import argparse
import os
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_fn_pad

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
        
    def get_activations(self, batch):
        """
        Runs a forward pass and captures embeddings from the last transformer layer.
        """
        self.model.eval()
        with torch.no_grad():
            base = batch['base_samples'].to(self.device)
            int_s = batch['int_samples'].to(self.device)
            target = batch['target_row'].to(self.device)
            mask = batch['int_mask'].to(self.device)
            idx = batch['int_node_idx'].to(self.device) # Needed for Phase 4 interface
            
            # Encoder
            # Note: Phase 4 CausalTransformer uses interleaved tokens (2N length)
            # x = [Features, Values, Features, Values ...]
            enc_out = self.model.encoder(base, int_s, target, mask)
            
            # Transformer
            features = self.model.transformer(enc_out) # (B, 2N, D)
            
            # Extract Value Tokens (Indices 1, 3, 5...) which carry the physics info
            # This matches the logic in CausalTransformer.forward: value_tokens = x[:, 1::2, :]
            value_features = features[:, 1::2, :] # (B, N, D)
                
            return value_features

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
            
            # Get Embeddings
            emb = self.get_activations(batch) # (B, N, D)
            
            # Get True Adjacency
            adj = batch['adj'].to(self.device) # (B, N, N)
            
            B, N, D = emb.shape
            
            # Create pairs (i, j): Predict if i -> j
            # emb_i: (B, N, 1, D) -> Source
            emb_i = emb.unsqueeze(2).expand(B, N, N, D)
            # emb_j: (B, 1, N, D) -> Dest
            emb_j = emb.unsqueeze(1).expand(B, N, N, D)
            
            # Concat: (B, N, N, 2D)
            # We predict A_ij from (Node_i, Node_j)
            pairs = torch.cat([emb_i, emb_j], dim=-1)
            
            # Flatten
            X_batch = pairs.reshape(-1, 2 * D)
            y_batch = adj.reshape(-1)
            
            # Downsample negatives? Adjacency is sparse (~20% positives).
            # Let's keep it balanced or just raw? Raw is fine for AUC.
            
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
        
        # 2. Define Probe
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
                try:
                    auc = roc_auc_score(vy.cpu().numpy(), v_pred.cpu().numpy())
                except ValueError:
                    auc = 0.5 # Handle single class edge case
                
            console.print(f"Epoch {epoch+1}: Loss {total_loss:.4f} | Val AUC: {auc:.4f}")
            
        return probe, auc

def load_checkpoint(path, device):
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    model = CausalTransformer(
        num_nodes=args.max_vars + 5,
        d_model=512,
        num_layers=args.num_layers
    )
    
    # DDP fix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model, train_args = load_checkpoint(args.checkpoint, device)
    
    # Generate Data (Use Max Vars to be rigorous)
    n_vars = train_args.max_vars
    gen = SCMGenerator(num_nodes=n_vars, edge_prob=0.25)
    
    train_ds = CausalDataset(gen, num_nodes_range=(n_vars, n_vars), samples_per_graph=32, infinite=False, validation_graphs=32)
    val_ds = CausalDataset(gen, num_nodes_range=(n_vars, n_vars), samples_per_graph=32, infinite=False, validation_graphs=16)
    
    train_loader = DataLoader(train_ds, batch_size=16, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn_pad)
    
    prober = CausalProber(model, device)
    probe, auc = prober.train_probe(train_loader, val_loader)
    
    print(f"\nFinal Probing AUC: {auc:.4f}")
    if auc > 0.70:
        print("SUCCESS: The model implicitly likely contains causal structure!")
    else:
        print("WARNING: Low probing accuracy. Structure might not be encoded explicitly.")

if __name__ == "__main__":
    main()
