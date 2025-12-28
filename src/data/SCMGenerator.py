import numpy as np
import pandas as pd
import networkx as nx
import torch

class SCMGenerator:
    def __init__(
        self,
        num_nodes: int = 10,
        edge_prob: float = 0.2,
        noise_scale: float = 1.0,
        num_samples_per_intervention: int = 100,
        intervention_prob: float = 0.3,
        intervention_values: list[float] | None = None,
        seed: int | None = None,
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

    def generate_dag(self, num_nodes: int | None = None, edge_prob: float | None = None, seed: int | None = None):
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
            eq = np.random.randint(1, 11)
            # Assign type based on eq (Simplified logic for brevity, expands to full map if needed)
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
        return dag

    def generate_data(self, dag, num_samples, noise_scale=None, intervention=None):
        if noise_scale is None: noise_scale = self.noise_scale
        nodes = list(dag.nodes())
        data = pd.DataFrame(np.random.normal(scale=noise_scale, size=(num_samples, len(nodes))), columns=nodes)
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
            
            total = data[node].values.copy()
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
                else: term = pval # Fallback
                total += term
            data[node] = np.clip(total, -20, 20)
        return data

    def generate_pipeline(self, num_nodes=None, edge_prob=None, num_samples_base=100, num_samples_per_intervention=100, intervention_prob=None, as_torch=True):
        if num_nodes is None: num_nodes = self.num_nodes
        dag = self.generate_dag(num_nodes, edge_prob)
        dag = self.edge_parameters(dag)
        
        df_base = self.generate_data(dag, num_samples_base)
        
        # Interventions
        nodes = list(dag.nodes())
        prob = intervention_prob if intervention_prob else self.intervention_prob
        num_targets = max(1, int(len(nodes) * prob))
        targets = np.random.choice(nodes, size=num_targets, replace=False)
        
        all_dfs = [df_base]
        all_masks = [np.zeros((1, len(nodes)))] # Base mask
        
        for t in targets:
            for val in self.intervention_values:
                df_int = self.generate_data(dag, num_samples_per_intervention, intervention={t: val})
                mask = np.zeros((num_samples_per_intervention, len(nodes)))
                mask[:, t] = 1.0
                all_dfs.append(df_int)
                all_masks.append(mask)
        
        result = {
            "dag": dag,
            "base_tensor": torch.tensor(df_base.values, dtype=torch.float32) if as_torch else df_base,
            "all_dfs": all_dfs, # Keep as DF for easier processing in dataset
            "all_masks": all_masks
        }
        return result
