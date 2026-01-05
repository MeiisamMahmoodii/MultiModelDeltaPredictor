import numpy as np
import pytest
from src.data.SCMGenerator import SCMGenerator

def test_seed_stability():
    # Run pipeline with seed 42 twice
    scm1 = SCMGenerator(num_nodes=5, seed=42)
    res1 = scm1.generate_pipeline(as_torch=False)
    
    scm2 = SCMGenerator(num_nodes=5, seed=42)
    res2 = scm2.generate_pipeline(as_torch=False)
    
    # Compare DAGs
    assert list(res1['dag'].edges()) == list(res2['dag'].edges())
    
    # Compare Data
    assert np.allclose(res1['base_tensor'].values, res2['base_tensor'].values)
    
def test_seed_diversity():
    # Run pipeline with seed 42 vs 43
    scm1 = SCMGenerator(num_nodes=5, seed=42)
    res1 = scm1.generate_pipeline(as_torch=False)
    
    scm2 = SCMGenerator(num_nodes=5, seed=43)
    res2 = scm2.generate_pipeline(as_torch=False)
    
    # Should be different
    # Note: Small chance of collision on small graph, but very unlikely for data values
    assert not np.allclose(res1['base_tensor'].values, res2['base_tensor'].values)
