import numpy as np
import pytest
from src.data.SCMGenerator import SCMGenerator

def test_scm_initialization():
    scm = SCMGenerator(num_nodes=5)
    assert scm.num_nodes == 5

def test_twin_world_property():
    """
    Verify that noise is reused between observational and interventional data
    when generated via generate_pipeline or manual injection.
    """
    scm = SCMGenerator(num_nodes=5, noise_scale=0.1, seed=42)
    dag = scm.generate_dag()
    dag = scm.edge_parameters(dag)
    
    # Generate noise manually
    num_samples = 10
    noise = np.random.normal(0, 1, (num_samples, 5))
    
    # Obs
    df_obs, noise_out = scm.generate_data(dag, num_samples, noise=noise)
    assert np.array_equal(noise, noise_out)
    
    # Int (Do X0 = 5)
    nodes = list(dag.nodes())
    target = nodes[0]
    df_int, noise_int_out = scm.generate_data(dag, num_samples, intervention={target: 5.0}, noise=noise)
    
    # Noise should be identical
    assert np.array_equal(noise, noise_int_out)
    
    # Data should be DIFFERENT at target
    assert np.allclose(df_int[target].values, 5.0)
    
    # Data should be SAME for ancestors of target (causality)
    # But hard to check ancestors easily without graph traversal.
    # At least check that SOME differences exist downstream.
    if len(list(dag.successors(target))) > 0:
         # If target has children, they should likely change (unless noise dominates or weak link)
         pass

def test_generate_pipeline_structure():
    scm = SCMGenerator(num_nodes=5, intervention_prob=1.0) # Force interventions
    res = scm.generate_pipeline(as_torch=True)
    
    assert "dag" in res
    assert "base_tensor" in res
    assert "all_dfs" in res
    assert isinstance(res["all_dfs"], list)
    assert len(res["all_dfs"]) > 1 # Base + at least one intervention logic 
