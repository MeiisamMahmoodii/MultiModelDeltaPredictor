import numpy as np
import pandas as pd
import networkx as nx
import torch

from typing import List, Optional
from scipy.special import expit

class SCMGenerator:
    def __init__(
        self,
        num_nodes: int = 10,
        edge_prob: float = 0.2,
        noise_scale: float = 1.0,
        num_samples_per_intervention: int = 100,
        intervention_prob: float = 0.3,
        intervention_values: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.noise_scale = noise_scale
        self.num_samples_per_intervention = num_samples_per_intervention
        self.intervention_prob = intervention_prob
        if intervention_values is None:
            self.intervention_values = [5.0, 8.0, 10.0]
        else:
            self.intervention_values = list(intervention_values)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_dag(self, num_nodes: Optional[int] = None, edge_prob: Optional[float] = None, seed: Optional[int] = None):
        if seed is None: seed = self.seed
        if seed is not None: np.random.seed(seed)
        if num_nodes is None: num_nodes = self.num_nodes
        if edge_prob is None: edge_prob = self.edge_prob

        dag = nx.DiGraph()
        dag.add_nodes_from(range(num_nodes))
        topo_order = np.arange(num_nodes)
        np.random.shuffle(topo_order)
        position_to_node = {i: topo_order[i] for i in range(num_nodes)}

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_prob:
                    parent = position_to_node[i]
                    child = position_to_node[j]
                    dag.add_edge(parent, child)
        return dag

    def edge_parameters(self, dag):
        for u, v in dag.edges():
            eq = np.random.randint(1, 17) # Increased range
            # Assign type
            if eq == 1: dag[u][v]['type'] = "linear"
            elif eq == 2: dag[u][v]['type'] = "negative linear"
            elif eq == 3: dag[u][v]['type'] = "sin"
            elif eq == 4: dag[u][v]['type'] = "cos"
            elif eq == 5: dag[u][v]['type'] = "tan"
            elif eq == 6: dag[u][v]['type'] = "log"
            elif eq == 7: dag[u][v]['type'] = "exp"
            elif eq == 8: dag[u][v]['type'] = "sqrt"
            elif eq == 9: dag[u][v]['type'] = "quadratic"
            elif eq == 10: dag[u][v]['type'] = "cubic"
            elif eq == 11: dag[u][v]['type'] = "sigmoid"
            elif eq == 12: dag[u][v]['type'] = "step"
            elif eq == 13: dag[u][v]['type'] = "abs"
            elif eq == 14: dag[u][v]['type'] = "poly"
            elif eq == 15: dag[u][v]['type'] = "sawtooth"
            elif eq == 16: dag[u][v]['type'] = "rational"
        return dag

    def generate_data(self, dag, num_samples, noise_scale=None, intervention=None, noise=None):
        if noise_scale is None: noise_scale = self.noise_scale
        nodes = list(dag.nodes())
        
        # Twin World Logic: Use provided noise if available
        if noise is None:
            noise = np.random.normal(scale=noise_scale, size=(num_samples, len(nodes)))
        
        data = pd.DataFrame(noise, columns=nodes)
        
        try:
            sorted_nodes = list(nx.topological_sort(dag))
        except:
            return data # Cycle fallback

        for node in sorted_nodes:
            if intervention and node in intervention:
                data[node] = intervention[node]
                continue
            
            parents = list(dag.predecessors(node))
            if not parents: continue
            
            # Start with noise (Additive)
            # Future: noise_term = data[node].values * heteroscedastic_factor
            noise_term = data[node].values.copy() 
            
            terms = []
            for p in parents:
                func = dag[p][node].get('type', 'linear')
                pval = data[p].values
                term = 0
                if func == 'linear': term = 2.0 * pval
                elif func == 'negative linear': term = -2.0 * pval
                elif func == 'sin': term = np.sin(pval)
                elif func == 'cos': term = np.cos(pval)
                elif func == 'tan': term = np.tanh(pval)
                elif func == 'quadratic': term = np.clip(pval, -5, 5)**2
                elif func == 'cubic': term = np.clip(pval, -3, 3)**3
                elif func == 'sigmoid': term = expit(pval)
                elif func == 'step': term = (pval > 0).astype(float)
                elif func == 'abs': term = np.abs(pval)
                elif func == 'poly': 
                    # 0.5*x^2 + 0.5*x (Simple polynomial mix)
                    term = 0.5 * (pval**2) + 0.5 * pval
                elif func == 'sawtooth': 
                    # Sawtooth wave: x - floor(x) centered
                    term = 2.0 * (pval - np.floor(pval)) - 1.0
                elif func == 'rational': 
                    # x / (1 + |x|) (Softsign-ish but rational)
                    term = 2.0 * pval / (1.0 + np.abs(pval))
                else: term = pval # Fallback
                terms.append(term)
            
            # Combine Terms: Additive vs Multiplicative (Interaction)
            # Randomly decide for this node if it's an "Interaction Node" 
            # (In a real scenario, this would be fixed per DAG, but here we do it per generation for variety 
            # OR we should store it in DAG. Let's do simple probabilistic mixing for now)
            
            if len(terms) > 1 and np.random.rand() < 0.3:
                # Interaction: Product of first two terms + Sum of rest
                # Represents "Modulation" (e.g. A * B + C)
                interact = terms[0] * terms[1]
                remaining = sum(terms[2:]) if len(terms) > 2 else 0
                total = noise_term + (interact + remaining)
            else:
                # Standard Additive Model
                total = noise_term + sum(terms)
                
            data[node] = np.clip(total, -100, 100)
        return data, noise

    def generate_pipeline(self, num_nodes=None, edge_prob=None, num_samples_base=100, num_samples_per_intervention=100, intervention_prob=None, as_torch=True, use_twin_world=True, intervention_scale=1.0):
        if num_nodes is None: num_nodes = self.num_nodes
        dag = self.generate_dag(num_nodes, edge_prob)
        dag = self.edge_parameters(dag)
        
        # 1. Generate Global Noise (The "World State")
        # We generate enough noise for the interventions. 
        # Note: Ideally baseline and interventions share the SAME noise rows.
        # Let's assume we want N samples where row i is "Person i".
        # We observe Person i normally, AND we observe Person i under intervention.
        nodes = list(dag.nodes())
        noise_dim = len(nodes)
        
        # Generate noise for the largest batch we might need
        # Actually, let's just generate distinct noise for base? 
        # NO! The whole point of Twin World is variance reduction.
        # We want: Delta = (Mechanisms(Noise) + Noise) - (Mechanisms_Int(Noise) + Noise)
        # So we MUST reuse the noise.
        
        # Let's generate a fixed set of noise vectors to be used across all scenarios
        # shape: (num_samples, num_vars)
        # Fix: Ensure noise is large enough for both base and intervention batches
        max_samples = max(num_samples_base, num_samples_per_intervention)
        global_noise = np.random.normal(scale=self.noise_scale, size=(max_samples, noise_dim))
        
        # Base Data (Observational) using Global Noise
        # Note: If num_samples_base != num_samples_per_int, we have a mismatch.
        # Now we strictly slice the noise to match the requested base samples.
        df_base, _ = self.generate_data(dag, num_samples_base, noise=global_noise[:num_samples_base])
        
        # Interventions
        prob = intervention_prob if intervention_prob else self.intervention_prob
        num_targets = max(1, int(len(nodes) * prob))
        targets = np.random.choice(nodes, size=num_targets, replace=False)
        
        all_dfs = [df_base]
        all_masks = [np.zeros((num_samples_per_intervention, len(nodes)))]
        
        # Standard Deviation for Adaptive Scaling
        base_stds = df_base.std()
        
        for t in targets:
            # Adaptive Intervention: Based on Sigma of the observational distribution
            # This ensures interventions are relative to the variable's natural scale.
            sigma = base_stds[t]
            if sigma < 1e-4: sigma = 1.0 # Safety fallback for constant/zero-var nodes
            
            # Intervene at ±1σ and ±2σ, scaled by the curriculum difficulty (intervention_scale)
            # If intervention_scale=1.0, we test "Standard" shifts (1-2 sigmas).
            # If intervention_scale=50.0, we test "Extreme" shifts (50-100 sigmas).
            coeffs = [-2.0, -1.0, 1.0, 2.0] 
            loop_values = [c * sigma * intervention_scale for c in coeffs]
            for val in loop_values:
                # Intervened Data using SAME Global Noise (Twin World) OR Random Noise (Ablation)
                noise_for_int = global_noise[:num_samples_per_intervention]
                if not use_twin_world:
                    # Generate fresh noise for this intervention
                    noise_for_int = np.random.normal(scale=self.noise_scale, size=(num_samples_per_intervention, noise_dim))
                    
                df_int, _ = self.generate_data(dag, num_samples_per_intervention, intervention={t: val}, noise=noise_for_int)
                
                mask = np.zeros((num_samples_per_intervention, len(nodes)))
                mask[:, t] = 1.0
                all_dfs.append(df_int)
                all_masks.append(mask)
        
        result = {
            "dag": dag,
            "base_tensor": torch.tensor(df_base.values, dtype=torch.float32) if as_torch else df_base,
            # We return lists of DFs. Dataset will collate them.
            "all_dfs": all_dfs, 
            "all_masks": all_masks
        }
        return result
