import torch
import pytest
from src.models.rope import RotaryEmbedding, apply_rotary_pos_emb

def test_rope_initialization():
    dim = 64
    rope = RotaryEmbedding(dim)
    assert rope.dim == dim
    assert rope.inv_freq.shape == (dim // 2,)

def test_rope_forward_caching():
    dim = 64
    rope = RotaryEmbedding(dim, max_position_embeddings=100)
    
    # Forward with seq_len within cache
    cos, sin = rope(torch.zeros(1, 1, 50, dim), seq_len=50)
    assert cos.shape == (1, 1, 50, dim)
    assert sin.shape == (1, 1, 50, dim)
    assert rope.max_seq_len_cached == 100
    
    # Forward with seq_len exceeding cache (should trigger re-cache)
    cos, sin = rope(torch.zeros(1, 1, 150, dim), seq_len=150)
    assert cos.shape == (1, 1, 150, dim)
    assert rope.max_seq_len_cached == 150

def test_apply_rotary_pos_emb():
    batch_size = 2
    heads = 4
    seq_len = 10
    head_dim = 32
    
    q = torch.randn(batch_size, heads, seq_len, head_dim)
    k = torch.randn(batch_size, heads, seq_len, head_dim)
    
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(q, seq_len=seq_len)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    
    # Test position 0 (should be unchanged theoretically if rotation logic holds 
    # but cos/sin are computed slightly differently. 
    # At pos 0, theta=0 => cos=1, sin=0. 
    # q_rot = q*1 + rotate(q)*0 = q)
    # Let's check the first position tokens
    
    # Cos at pos 0 should be all 1s, Sin at pos 0 should be all 0s
    assert torch.allclose(cos[:, :, 0, :], torch.ones_like(cos[:, :, 0, :]))
    # Sin calculation in rope.py: 
    # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    # t starts at 0. So freqs[0] is 0. sin(0) is 0.
    assert torch.allclose(sin[:, :, 0, :], torch.zeros_like(sin[:, :, 0, :]))
    
    # So q at pos 0 should be equal to q_rot at pos 0
    assert torch.allclose(q[:, :, 0, :], q_rot[:, :, 0, :], atol=1e-6)

def test_relative_invariance_concept():
    # RoPE enables relative position attention.
    # Dot product of rotated vectors at pos m and n should depend on m-n.
    
    dim = 64
    rope = RotaryEmbedding(dim)
    
    # Vectors a and b
    a = torch.randn(1, 1, 1, dim)
    b = torch.randn(1, 1, 1, dim)
    
    # Embed at positions 0 and 5
    cos_0, sin_0 = rope(a, seq_len=1) # Treat as pos 0 (index 0 in sequence of len 1? No, rope takes seq_len)
    # Wait, rope returns cached cos/sin sliced to seq_len. 
    # To get specific positions, we rely on the slicing.
    
    # Let's get full cache for 10 positions
    cos, sin = rope(a, seq_len=10)
    
    # Pos 0 and 5
    pos1 = 0
    pos2 = 5
    diff = pos2 - pos1
    
    # Apply rotation to 'a' at pos1
    # q, k args to apply_rotary are (B, H, S, D).
    # We will simulate single vector at specific pos.
    
    def rotate_vec_at_pos(vec, p):
        # vec: (1, 1, 1, D)
        # We need cos/sin at index p.
        # cos shape (1, 1, 10, D).
        c = cos[:, :, p:p+1, :]
        s = sin[:, :, p:p+1, :]
        res, _ = apply_rotary_pos_emb(vec, vec, c, s) # Passing same for q,k just to reuse function
        return res
        
    a_rot = rotate_vec_at_pos(a, pos1)
    b_rot = rotate_vec_at_pos(b, pos2)
    
    score_1 = (a_rot * b_rot).sum()
    
    # Now shift both by +2 -> Pos 2 and 7. Diff is still 5.
    pos3 = 2
    pos4 = 7
    
    a_rot_shifted = rotate_vec_at_pos(a, pos3)
    b_rot_shifted = rotate_vec_at_pos(b, pos4)
    
    score_2 = (a_rot_shifted * b_rot_shifted).sum()
    
    # Scores should be identical
    assert torch.allclose(score_1, score_2, atol=1e-5)
