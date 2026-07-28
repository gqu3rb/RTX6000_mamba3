import torch
from mamba_ssm import Mamba3
batch, length, dim = 2, 2048, 768
x = torch.randn(batch, length, dim).to(torch.bfloat16).to("cuda")
model = Mamba3(
    # This module uses roughly 6 * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=128,  # SSM state size
    headdim=64, # SSM headdim
    is_mimo=False, # Use MIMO mode
    mimo_rank=4, # MIMO rank when is_mimo=True
    chunk_size=16, # 64/mimo_rank if x is in bf16, else 32/mimo_rank
    is_outproj_norm=False, # Additional post SSM norm
    dtype=torch.bfloat16,
).to("cuda")
y = model(x)
assert y.shape == x.shape