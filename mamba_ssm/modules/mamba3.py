# Copyright (c) 2026, Dao AI Lab, Goombalab.

import math
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
except ImportError:
    mamba3_mimo_combined = None

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import apply_rotary_qk_inference_fwd

try:
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
except ImportError:    
    mamba3_step_fn = None


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Heavy-tail activation for data-dependent A.

    Using this activation can improve stability during WSD training and at
    higher learning rates.

        f(x) = 1 + x        if x >= 0
            = 1 / (1 - x)  if x < 0

    The function is positive, continuous, and differentiable at x = 0.
    """
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)

class Mamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        # ----------------------------------------
        # Mamba-3 configs
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        fuse_pregate_headwise_norm=True,
        #-------------------------------------------
        # Fused kernel and sharding options
        chunk_size=64, # Recommended: 64 for SISO, 64/mimo_rank for MIMO
        dropout=0.0,  # Just to absorb the kwarg
        layer_idx=None,  # Absorb kwarg for general module
        n_layer=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # model dimension, related to D in [Published Version] Mamba3.pdf, P.3
        self.d_model = d_model
        # state dimension, related to N in [Published Version] Mamba3.pdf, P.3
        # generally set to around 64 or 128, refer to Mamba2.pdf, P.24
        self.d_state = d_state
        # expansion factor, related to e in Mamba2.pdf, P.26
        # typically set to 2
        self.expand = expand
        # head dimension, related to P in [Published Version] Mamba3.pdf, P.9
        # The head dimension is generally set to around 64 or 128, refer to Mamba2.pdf, P.24
        # The definitions of head dimension in 
        # [Published Version] Mamba3.pdf, P.9
        # AND
        # Mamba2.pdf, P.24
        # are the same, but there is a more detailed explanation on Mamba2.pdf, P.24
        self.headdim = headdim
        # chunk size, related to Q in Mamba2.pdf, P.19 OR 
        # C in [Published Version] Mamba3.pdf, P.10
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm=is_outproj_norm
        self.is_mimo = is_mimo
        # mimo rank, related to R in [Published Version] Mamba3.pdf, P.27
        # Ctrl+F "mimo_rank not in" in mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py to know the supporting mimo_rank values
        self.mimo_rank = mimo_rank
        self.fuse_pregate_headwise_norm = bool(
            fuse_pregate_headwise_norm and self.is_mimo and self.is_outproj_norm
        )
        if not self.is_mimo:
            self.mimo_rank = 1
        else:
            assert mamba3_mimo_combined is not None, "Fails to import Mamba-3 MIMO kernels. Please ensure you installed the necessary dependencies, such as TileLang."

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        # number of heads sharing the same B and C
        # related to Multi-input SSM in Mamba2.pdf, P.24
        # AND
        # related to Grouped Head Patterns in Mamba2.pdf, P.25
        self.num_bc_heads = ngroups
        
        # RoPE flags
        assert rope_fraction in [0.5, 1.0]

        # Relatd to the '2' of h(t) \in N/2 in [Published Version] Mamba3.pdf, P.7
        self.rotary_dim_divisor = int(2/rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads * self.mimo_rank + 3 * self.nheads + self.num_rope_angles
        # map each vector of dimension d_model to a vector of dimension d_in_proj
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt_bias parameterization        
        # generate the initial bias parameter value of \Delta_t for each head
        # the generated the initial bias parameter values are log-uniform
        # See:
        # "D:\user\google drive - great9284@gmail.com.tw\My Drive\陽明交通大學\論文\paper\MAMBA3\src\Why Log Uniform Randomization is used to generate the Delta t bias parameter\Why Log Uniform Randomization is used to generate the Delta t bias parameter.pdf"
        # to understand why use log-uniform randomization
        _dt = torch.exp(
            # torch.rand generates self.nheads random numbers in the range of [0, 1)
            # and map the result of torch.rand to [ln(dt_min, ln(dt_max))
            torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # prevent from _dt too close to 0 or equals to 0, 
        # causing the result of log(-torch.expm1(-_dt)) in the next line approaches to -\Inf
        _dt = torch.clamp(_dt, min=dt_init_floor)
        # given the _dt that has been calculated, solve for _dt_bias such that
        # softplus(_dt_bias) == _dt
        # expm1(x) is functionally equivalet to $1 - exp(x)$, but is improved to avoid Catastrophic Cancellation problem
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        # make the dt_bias to be a trainable parameter
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        # disable weight decay during the training process of dt_bias
        self.dt_bias._no_weight_decay = True
        
        # B and C biases
        # set the initial trainning values
        # Their dimension can be referred to: 
        # [Published Version] "B, C Biases", Mamba3.pdf, P.11
        # setting initial value to all 1's just for simplicity
        # Refer to: [Published Version] "B, C Bias Parameterization", Mamba3.pdf, P.30
        self.B_bias = nn.Parameter(1+torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
        self.C_bias = nn.Parameter(1+torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
                                                       
        # RMS Norm for B and C
        # Refer to: [Published Version] "BC / QK Normalization.", Mamba3.pdf, P.11
        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        if self.is_mimo:
            # Initialize up/down MIMO projection (for x and z)
            # The description of their dimenison can be seen on:
            # Appendix C, [Published Version] Mamba3.pdf
            mimo_x_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            mimo_z_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
            mimo_o_init_weights = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank

            self.mimo_x = nn.Parameter(mimo_x_init_weights, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init_weights, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init_weights, requires_grad=True)
    
        # D "skip" parameter
        # This parameter is not instroduced in any versions of mamba theses,
        # but it is mentioned in
        # section 2.1, Efficiently Modeling Long Sequences with Structured State Spaces.pdf
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                **factory_kwargs
            )
            if self.fuse_pregate_headwise_norm:
                assert self.norm.weight.numel() == self.nheads * self.headdim, (
                    "Fused pregate headwise norm expects one norm weight per head/headdim element."
                )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    # cu_seqlens means if variable length input is enabled
    # related to: Mamba2.pdf, P.28
    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim)
        Returns: same shape as u
        """
        # batch is batch size
        # seqlen is the sequence length, related to T in [Published Version] Mamba3.pdf, P.3
        # dim is the model dimension, related to D in [Published Version] Mamba3.pdf, P.3
        batch, seqlen, dim = u.shape

        angle_dt_state, ssm_state, k_state, v_state  = None, None, None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            angle_dt_state, ssm_state, k_state, v_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _, _, _ = self.step(u, angle_dt_state, ssm_state, k_state, v_state)
                return out

        # Apply in_proj
        # shape of zxBCdtAtrap is "batch seqlen d_in_proj"
        # search for the first occurrence of d_in_proj in this file to understand what is it
        # The position of z, x, B, C, dd_A, and angles is related to the green note on Figure 2 in [Published Version] Mamba3.pdf, P.11
        # dd_dt is related to \Delta_t in [Published Version] Mamba3.pdf, P.5
        # dd_A is related to A_t in [Published Version] Mamba3.pdf, P.5
        # trap is related to \lambda in [Published Version] Mamba3.pdf, P.5
        # angles is related to \theta(t) in Proposition 2, [Published Version] Mamba3.pdf, P.7
        zxBCdtAtrap = self.in_proj(u)
        # the dimension of z is assigned as "batch seqlen self.d_inner"
        # the dimension of x is assigned as "batch seqlen self.d_inner"
        # ... etc
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner, 
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads, 
                self.num_rope_angles
            ],
            dim=-1)
        # change the dimension of z from "batch seqlen self.d_inner" = "batch seqlen h*p" to "batch seqlen h p"
        # search for the first occurrence of the names of all member variables (begin with self.) in this file to know what they mean
        # h is number of heads, related to self.nheads in this file
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        # search for the first occurrence of self.num_bc_heads in this file for its meaning
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        # Compute ADT, DT
        # dd_A may be positive, but the resulting \Delta A_t (i.e. ADT) must be negative.
        # Related to:
        # Mamba-2’s Parameterization, [Published Version] Mamba3.pdf, P.3 (Mamba3 inherits such design in Mamba2)
        # AND
        # 3.5.1 Connection to Gating Mechanisms, Mamba1.pdf, P.8
        # AND
        # C Mechanics of Selective SSMs, Mamba1.pdf, P.27
        # AND
        # the heavy_tail_activation() function defined at the top of this file
        # So we apply heavy_tail_activation to dd_A to make it be positive, and then add a negative
        # sign to ensure it is negative. (Note that applying such an activation to dd_A is not
        # explicitly described in [Published Version] Mamba3.pdf)
        # NOTE: upstream commit c69aaab "Stabilize data-dependent A with heavy-tail activation (#962)"
        # replaced the previous -F.softplus(dd_A) with -heavy_tail_activation(dd_A).
        # Both are positive-valued, so the sign reasoning above is unchanged; the heavy-tail
        # activation f(x) = 1+x (x>=0), 1/(1-x) (x<0) decays more slowly than softplus for very
        # negative x, which improves stability during WSD training and at higher learning rates.
        _A = -heavy_tail_activation(dd_A.to(torch.float32)) # (B, L, N) = (batch, seqlen, self.nheads)
        # set _A to -self.A_floor if it is larger than -self.A_floor
        # this line is used to prevent _A underflows and equals 0 (which makes ADT equal to 0,
        # violating the rule that \Delta A_t must be negative) when dd_A is very small
        # thi line is not included in any mamba papers
        _A = torch.clamp(_A, max=-self.A_floor)            
        # Refer to: Algorithm 2, Mamba1.pdf, P.6
        DT = F.softplus(dd_dt + self.dt_bias) # (B, L, N) = (batch, seqlen, self.nheads)
        # '*' is element-wise multiplication
        ADT = _A * DT
        # rearrange the dimension of DT and ADT to match the memory mapping and parallel processing setting of 
        # Triton/tilelang kernel 
        # Related to:
        # Ctrl+F "ADT, DT, Trap:              (batch, nheads, seqlen)"
        # in mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Compute angle — cast to float32 as required by the MIMO/SISO kernels
        # .unsqueeze(-2) changes the dimension of angles from (batch, seqlen, num_rope_angles) to (batch, seqlen, 1, num_rope_angles)
        # .expand brocast the num_rope_angles, changes the dimension of angles from (batch, seqlen, 1, num_rope_angles) to (batch, seqlen, self.nheads, num_rope_angles)
        # use float32 is due to the sensitivity of precision error of cos_approx and sin_approx calculation in mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32) # (B, L, N, S)

        # Apply RMS Norm on B and C
        B = self.B_norm(B)
        C = self.C_norm(C)
        
        # Apply Mamba-3 kernel
        if self.is_mimo:
            # call the mamba3_mimo function in mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py
            # related to:
            # Ctrl+F the first occurrence of "mamba3_mimo_combined" in this file
            y = mamba3_mimo_combined(
                Q=C,
                K=B,
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                # MIMO_Out=self.mimo_o if (self.fuse_pregate_headwise_norm is True)
                #                       OR (self.is_outproj_norm is False)
                # NOTE: this condition used to be just "self.is_outproj_norm is False".
                # Upstream commit 33e2849 "Fuse gated norm and reduce bwd memory I/O in Mamba-3 MIMO (#967)"
                # added the fused path: when self.fuse_pregate_headwise_norm is True the output RMSNorm
                # is computed inside the TileLang kernel instead of in Python (see the
                # "if self.is_outproj_norm and not self.fuse_pregate_headwise_norm" block below),
                # so the kernel needs mimo_o (and Z) to finish the whole output stage by itself.
                MIMO_Out=self.mimo_o if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
                Angles=angles,
                D=self.D,
                Z=z if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=ssm_state is not None,
                cu_seqlens=cu_seqlens,
                fuse_pregate_headwise_rms_norm=self.fuse_pregate_headwise_norm,
                outproj_norm_weight=self.norm.weight if self.fuse_pregate_headwise_norm else None,
                outproj_norm_eps=self.norm.eps if self.fuse_pregate_headwise_norm else 1e-5,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k, last_v, *rest = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_state.copy_(last_k)
                v_state.copy_(last_v)
            if self.is_outproj_norm and not self.fuse_pregate_headwise_norm:
                z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z = rearrange(z, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z)
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                # .squeeze(dim) removed the selected dim of a tensor if the selected dim is 1
                # C.squeeze(2) transform the dimension of C from (B L R G N) to
                # (B L G N) if R=1
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                # Z=z if self.is_outproj_norm is False
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                # see the comments in the starting of `mamba3_siso_combined()`` function
                # to know its usage
                Input_States=None,
                return_final_states=ssm_state is not None,
                cu_seqlens=cu_seqlens,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k, last_v, *rest = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_state.copy_(last_k.unsqueeze(1))
                v_state.copy_(last_v)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z)
        
        out = self.out_proj(y.to(x.dtype))
        return out
    

    def _preprocess(self, A_proj, dd_dt, B, C, x, z, trap_proj, angle_proj):
        _A = -heavy_tail_activation(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        rank = self.mimo_rank if self.is_mimo else 1
        B = rearrange(B, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)

        B = self.B_norm(B)
        C = self.C_norm(C)

        B = B.expand(-1, -1, self.nheads, -1) # (B, R, N, S)
        C = C.expand(-1, -1, self.nheads, -1) # (B, R, N, S)
    
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, trap, _A, angles

    def _postprocess(self, y, outpj, z, zpj, headdim):
        # y: (batch, R, H, D) — apply mimo_z to z, then norm, then mimo_o
        z_r = torch.einsum("bhp,rhp->brhp", z.float(), zpj)  # (batch, R, H, D)
        z_r = rearrange(z_r, "b r h p -> b r (h p)")
        y = rearrange(y, "b r h p -> b r (h p)").float()
        y = self.norm(y, z_r)
        y = rearrange(y, "b r (h p) -> b r h p", p=headdim)
        y = torch.einsum("brhp,rhp->bhp", y, outpj)  # (batch, H, D)
        return y

    def step(self, u, angle_state, ssm_state, k_state, v_state, **kwargs):
        """
        Decode function using CuteDSL kernel from mamba3_step_fn.py.
        Also modify the state vars in-place for the next step.

        NOTE: Only tested on H100. Compatibility with other hardware
        will be made available in the future.

        Args:
            u: (batch, d_model)
            angle_state: (batch, nheads, num_rope_angles)
            ssm_state: (batch, nheads, headdim, d_state)
            k_state: (batch, R, nheads, d_state), where R = mimo_rank (R=1 if not MIMO)
            v_state: (batch, nheads, headdim)
            **kwargs: ignored
        Returns:
            out: (batch, d_model)
            nxt_angle_state: (batch, nheads, num_rope_angles)
            state_out: (batch, nheads, headdim, d_state)
            nxt_k_state: (batch, R, nheads, d_state), where R = mimo_rank (R=1 if not MIMO)
            nxt_v_state: (batch, nheads, headdim)
        """
        assert mamba3_step_fn is not None, "Cute Mamba-3 step function is not available. Please ensure you installed the necessary dependencies, such as nvidia-cutlass-dsl and quack-kernels."

        # in_proj
        zxBCdt = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1)

        DT, B, C, x, z, trap, A, angles = self._preprocess(
            dd_A, dd_dt, B, C, x, z, trap, angles)

        bias_q = rearrange(self.C_bias, "h r n -> r h n")
        bias_k = rearrange(self.B_bias, "h r n -> r h n")

        # NOTE: MIMO calls the Tilelang kernel, 
        # which permute the blockwise rotation matrix so that
        # the i-th entry is paired with the i+N//2-th entry:
        rotate_pairwise = not self.is_mimo
        C, B, nxt_angle_state = apply_rotary_qk_inference_fwd(
            q=C, k=B, angle_state=angle_state, 
            angle_proj=angles, dt=DT, bias_q=bias_q, bias_k=bias_k, 
            conjugate=False, inplace=False, # NOTE: inplace is incompatible with self.nheads != self.num_bc_heads
            rotate_pairwise=rotate_pairwise)

        nxt_v_state = x
        nxt_k_state = B

        if self.is_mimo:
            xpj = rearrange(self.mimo_x, "h r p -> r h p", p=self.headdim).contiguous()
            zpj = rearrange(self.mimo_z, "h r p -> r h p", p=self.headdim).contiguous()
            outpj = rearrange(self.mimo_o, "h r p -> r h p", p=self.headdim).contiguous()
        else:
            xpj = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)
            zpj = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=z.device, dtype=z.dtype)
            outpj = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)

        if self.is_outproj_norm:
            batch = x.shape[0]
            y = torch.empty(batch, self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)
            mamba3_step_fn(
                ssm_state,
                k_state,
                v_state,
                A,
                B,
                C,
                self.D,
                x,
                DT,
                trap,
                xpj,
                outproj=None,
                state_out=None, # can be not in place if pass in state_out
                out=y,
                z=None,
                zproj=None,
                tile_D=64,
                num_warps=4,
            )
            y = self._postprocess(y, outpj, z, zpj, self.headdim)
        else:
            y = torch.empty_like(x)
            mamba3_step_fn(
                ssm_state,
                k_state,
                v_state,
                A,
                B,
                C,
                self.D,
                x,
                DT,
                trap,
                xpj,
                outproj=outpj,
                state_out=None, # can be not in place if pass in state_out
                out=y,
                z=z,
                zproj=zpj,
                tile_D=64,
                num_warps=4,
            )

        # out_proj
        out = rearrange(y, "b h p -> b (h p)")
        out = self.out_proj(out.to(x.dtype))

        angle_state.copy_(nxt_angle_state)
        # Uncomment the following if mamba3_step_fn is not in place:
        # state_out = torch.empty_like(ssm_state)
        # ssm_state.copy_(state_out) 
        k_state.copy_(nxt_k_state)
        v_state.copy_(nxt_v_state)

        return out, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state
    
    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, inplace_state=None, **kwargs):
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype

        # RoPE State
        angle_dt_state = torch.zeros(
            (batch_size, self.nheads, self.num_rope_angles),
            device=device,
            dtype=torch.float32,
        )

        # Mamba-3 Combined Kernel States
        # SSM State
        ssm_state = torch.zeros(
            (batch_size, self.nheads, self.headdim, self.d_state),
            device=device,
            dtype=torch.float32,
        )

        # K (=B) State
        if self.is_mimo:
            k_state = torch.zeros(
                (batch_size, self.mimo_rank, self.nheads, self.d_state),
                device=device,
                dtype=dtype,
            )
        else:
            k_state = torch.zeros(
                (batch_size, 1, self.nheads, self.d_state),
                device=device,
                dtype=dtype,
            )

        # V (=x) State
        v_state = torch.zeros(
            (batch_size, self.nheads, self.headdim),
            device=device,
            dtype=dtype,
        )

        return (angle_dt_state, ssm_state, k_state, v_state)
    
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        device = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype

        if self.layer_idx not in inference_params.key_value_memory_dict:
            angle_dt_state = torch.zeros(
                (batch_size, self.nheads, self.num_rope_angles),
                device=device,
                dtype=torch.float32,
            )
            ssm_state = torch.zeros(
                (batch_size, self.nheads, self.headdim, self.d_state),
                device=device,
                dtype=torch.float32,
            )
            if self.is_mimo:
                k_state = torch.zeros(
                    (batch_size, self.mimo_rank, self.nheads, self.d_state),
                    device=device,
                    dtype=dtype,
                )
            else:
                k_state = torch.zeros(
                    (batch_size, 1, self.nheads, self.d_state),
                    device=device,
                    dtype=dtype,
                )
            v_state = torch.zeros(
                (batch_size, self.nheads, self.headdim),
                device=device,
                dtype=dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (angle_dt_state, ssm_state, k_state, v_state)
        else:
            angle_dt_state, ssm_state, k_state, v_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                angle_dt_state.zero_()
                ssm_state.zero_()
                k_state.zero_()
                v_state.zero_()
        return angle_dt_state, ssm_state, k_state, v_state
