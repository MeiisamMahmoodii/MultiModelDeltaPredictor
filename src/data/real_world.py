import pandas as pd
import numpy as np
import os
import networkx as nx

class RealWorldLoader:
    """
    Loader for standard Causal Discovery benchmarks: Sachs, Alarm, etc.
    """
    def __init__(self, data_dir='data/real_world'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_sachs(self):
        """
        Loads the Sachs protein signaling network (11 nodes).
        Returns: (data_df, ground_truth_adj)
        """
        # Placeholder: If file assumes local, we mock or error
        # In a real impl, we'd download from bnlearn repository
        # For now, let's verify if files exist, else return dummy for dry run compatibility
        
        path_data = os.path.join(self.data_dir, 'sachs.csv')
        path_adj = os.path.join(self.data_dir, 'sachs_adj.csv')
        
        if os.path.exists(path_data):
            df = pd.read_csv(path_data)
            adj = np.loadtxt(path_adj, delimiter=',')
            return df, adj
        else:
            print(f"Dataset Sachs not found in {self.data_dir}. Using Mock 11-node data.")
            # Mock
            return pd.DataFrame(np.random.randn(100, 11)), np.zeros((11,11))

    def load_alarm(self):
        """
        Loads the ALARM network (37 nodes).
        """
        path_data = os.path.join(self.data_dir, 'alarm.csv')
        if os.path.exists(path_data):
            return pd.read_csv(path_data), None # Adj often hard to get as CSV
        else:
            print(f"Dataset Alarm not found in {self.data_dir}. Using Mock 37-node data.")
            return pd.DataFrame(np.random.randn(100, 37)), np.zeros((37,37))
