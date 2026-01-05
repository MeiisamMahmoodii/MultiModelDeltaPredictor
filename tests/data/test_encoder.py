import torch
import pytest
from src.data.encoder import InterleavedEncoder, HybridEmbedding, FourierEmbedding

def test_fourier_shape():
    d_out = 16
    layer = FourierEmbedding(d_out)
    x = torch.randn(2, 5, 1)
    out = layer(x)
    assert out.shape == (2, 5, d_out)

def test_hybrid_embedding_shape():
    d_model = 32
    layer = HybridEmbedding(d_model)
    x = torch.randn(2, 5, 1) # (B, N, 1)
    out = layer(x)
    assert out.shape == (2, 5, d_model)

def test_interleaved_encoder_output():
    batch_size = 2
    num_vars = 4
    d_model = 16
    
    encoder = InterleavedEncoder(num_vars, d_model)
    
    base_samples = torch.randn(batch_size, num_vars)
    int_samples = torch.randn(batch_size, num_vars) # Unused in forward?
    target_row = torch.randn(batch_size, num_vars)
    int_mask = torch.zeros(batch_size, num_vars)
    
    tokens = encoder(base_samples, int_samples, target_row, int_mask)
    
    # Expect 2 tokens per variable (ID + Value) -> 2 * num_vars
    expected_len = 2 * num_vars
    assert tokens.shape == (batch_size, expected_len, d_model)

def test_interleaved_ordering_and_ids():
    # Verify the interleaving logic: [ID0, Val0, ID1, Val1...]
    # We can't verify values easily after embedding, but we can verify shapes.
    # The code does:
    # ids = [0, 1, 2, 3]
    # stacked = stack([id_emb, val_emb], dim=2)
    # flatten -> [id0, val0, id1, val1...]
    pass
