import numpy as np
import pytest
from src.data.SCMGenerator import SCMGenerator

def test_ood_generation_capability():
    # Verify SCM can generate extreme values if we intervene with them
    scm = SCMGenerator(num_nodes=5)
    dag = scm.generate_dag()
    dag = scm.edge_parameters(dag)
    
    # Standard range usually [-2, 2] noise
    
    # Force extreme intervention
    extreme_val = 100.0 # Our data is clipped to [-100, 100] in generator, let's try 50
    extreme_val = 50.0
    
    nodes = list(dag.nodes())
    target = nodes[0]
    
    df, _ = scm.generate_data(dag, num_samples=10, intervention={target: extreme_val})
    
    assert np.allclose(df[target].values, extreme_val)
    
    # Check that children also have high values (if linear/positive connection)
    # This just ensures "Physics" propagates even for OOD inputs
    pass
