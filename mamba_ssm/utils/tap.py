"""
Two mutually independent modes, both driven by the same `tap()` call sites in
mamba_ssm/modules/mamba3.py:

  capture()    -- store every tapped tensor verbatim (detached / float32 / CPU) for one
                  forward pass. Used by mamba3_compare_nano4/probe.py to compare the
                  `nano` and `rtx6000-adapt` implementations numerically. Unchanged.

  accumulate() -- store NOTHING per-tensor; fold each tapped tensor into running
                  statistics kept on the GPU, keyed by (layer_idx, tap name). Survives
                  arbitrarily many forward passes, so it can stay switched on for a whole
                  lm-eval downstream-task run and report the activation distribution of
                  every Mamba-3 block. Used by mamba3_distribution/.
"""

import math
from contextlib import contextmanager

import torch

# The ONLY line that differs between branches. Each branch is pinned to one machine
# (`nano` -> H200, `rtx6000-adapt` -> RTX6000), so this identifies which implementation
# is loaded. run_side.py asserts it against --tag, which makes it impossible to compare
# a branch against itself after a stray checkout or a bad PYTHONPATH.
IMPL = "nano"

_ACTIVE = None
_ACC = None


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
# How many dimensions at the end of the tensor together form "one channel." for per-channel statistics
#     z, x            (b, l, h, p)
#     B, C            (b, l, r, g, n)
#     angles_exp      (b, l, h, s)
_CHANNEL_TRAILING_DIMS = {
    "z": 2, "x": 2, # setting h*p element for a channel of z, x is also applied with the same setting
    "B_raw": 3, "C_raw": 3, "B_normed": 3, "C_normed": 3,
    "angles_cumsum": 2, "B_rope": 3, "C_rope": 3,
    "ssm_out": 2, "ssm_mimo_ver_out": 3, "y_pre_outproj": 1, "out": 1,
    "dd_dt": 1, "dd_A": 1, "trap_raw": 1, "angles_raw": 1,
    "DT": 1, "A": 1, "ADT": 1, "angles_exp": 1,
    "u": 1,
    # Mamba-2 taps (mamba_ssm/modules/mamba2.py). B/C carry no mimo_rank axis there, so
    # their channel is g*n rather than Mamba-3's r*g*n.
    "xBC_preconv": 1, "xBC_postconv": 1,
    "B_prekernel": 2, "C_prekernel": 2,
}

# log2|x| histogram settings
# out-of-range values are clamped into the end bins.
# (16.0 - (-32.0))/768 = 0.0625 exponent per bin = 2^0.0625 - 1 = ~4.4% threshold granularity.
# 768 int64 bins is 6 KB per tap site, ~1.2 MB for a 12-layer run.
LOG2_LO, LOG2_HI, LOG2_BINS = -32.0, 16.0, 768

# An element counts as an outlier when it falls outside mean +/- SIGMA_K * std
SIGMA_K = 3.0


def is_active():
    return _ACTIVE is not None or _ACC is not None


def expect_impl(name):
    """Raise unless the loaded mamba_ssm is the implementation the caller expects."""
    if IMPL != name:
        raise RuntimeError(
            f"expected the {name!r} implementation but imported {IMPL!r} "
            f"(check the checked-out branch and PYTHONPATH)"
        )


def _key(name, layer):
    """`layer is None` reproduces the old flat key, so mamba3_compare_nano4 -- which
    builds a bare Mamba3(**config) with no layer_idx -- keeps its exact key names and
    compare.py needs no change. Inside MambaLMHeadModel every block has a layer_idx, so
    the keys namespace themselves automatically."""
    return name if layer is None else f"L{int(layer):02d}/{name}"


def tap(name, t, layer=None):
    """Record `t` under `name` when capture or accumulation is active; always return `t`
    unchanged so this can be inserted into forward() without altering the computation."""
    if _ACC is not None: # The value of _ACC is assigned by accumulate() in the beginning
        _ACC.add(_key(name, layer), name, t)
    if _ACTIVE is None: # The value of _ACTIVE is assigned by capture() in the beginning
        return t
    k = _key(name, layer)
    if k in _ACTIVE:
        raise KeyError(f"tap {k!r} recorded twice in one capture")
    _ACTIVE[k] = t.detach().float().cpu()
    return t


@contextmanager
def capture():
    """Yield a dict that every tap() writes into for the duration of the block."""
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("tap.capture() is not reentrant")
    _ACTIVE = {}
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = None


# ---------------------------------------------------------------------------
# accumulate() mode
# ---------------------------------------------------------------------------

def _count_ge(h, t, lo, hi, bins):
    """Count of elements in h that >= t.

    A threshold like mean + 3*std lands *inside* a bin. Dropping that whole bin
    undercounts the tail by ~20% at these bin widths -- the density just inside the
    threshold is much higher than the density beyond it. So the straddling bin is split
    proportionally in log space, which is the same assumption the bin already makes about
    how its own contents are laid out.
    """
    if t <= 0.0:
        return float(h.sum())
    w = (hi - lo) / bins
    x = (math.log2(t) - lo) / w
    if x <= 0.0:
        return float(h.sum())
    if x >= bins:
        return 0.0
    j = int(math.floor(x))
    return float(h[j + 1:].sum()) + (j + 1 - x) * float(h[j])


def outlier_counts(hist, hist_neg, n_zero, lo, hi, k, mean, std):
    """Elements outside mean +/- k*std, split into the two tails.

    `hist` is the total log2|x| histogram and `hist_neg` its negative half, so
    hist_pos = hist - hist_neg. Both tails are counted in magnitude space and then
    attributed by sign, which is what makes an asymmetric threshold answerable at all.

    distribution.py carries an independent copy of this logic so it can run without
    mamba_ssm imported.
    """
    hist_pos = (hist - hist_neg).round()
    n_pos, n_neg = float(hist_pos.sum()), float(hist_neg.sum())
    up = lambda h, t: _count_ge(h, t, lo, hi, hist.numel())
    dn = lambda h, t: float(h.sum()) - up(h, t)

    U, L = mean + k * std, mean - k * std
    # x > U
    if U > 0:      hi_tail = up(hist_pos, U)
    elif U == 0:   hi_tail = n_pos
    else:          hi_tail = n_pos + n_zero + dn(hist_neg, -U)
    # x < L
    if L < 0:      lo_tail = up(hist_neg, -L)
    elif L == 0:   lo_tail = n_neg
    else:          lo_tail = n_neg + n_zero + dn(hist_pos, L)
    return int(round(lo_tail)), int(round(hi_tail))


class _Stat:
    """Running statistics for one (layer, tap) site. Everything lives on the tapped
    tensor's own device and is only pulled to CPU by finalize()
    """

    def __init__(self, t, n_channels, device):
        self.dtype = str(t.dtype)
        self.shape_tail = tuple(t.shape[1:])   # batch dim varies between lm-eval batches
        self.n_channels = n_channels
        self.calls = 0
        self.count = 0                 # number of scalar elements folded in
        z64 = lambda: torch.zeros((), dtype=torch.int64, device=device)
        # counters stay on the GPU: reading them per tap would force a host sync on every
        # one of the ~360 taps a 24-layer forward fires, which dominates the eval runtime.
        self.n_zero = z64()
        self.n_nonfinite = z64()
        z = lambda: torch.zeros((), dtype=torch.float64, device=device)
        self.sum = z()
        self.sumsq = z()
        self.sumabs = z()
        self.sum3 = z()
        self.sum4 = z()
        self.vmin = torch.full((), float("inf"), dtype=torch.float64, device=device)
        self.vmax = torch.full((), float("-inf"), dtype=torch.float64, device=device)
        # per-channel absmax
        self.ch_absmax = torch.zeros(n_channels, dtype=torch.float32, device=device)
        self.ch_sumabs = torch.zeros(n_channels, dtype=torch.float64, device=device)
        self.hist = torch.zeros(LOG2_BINS, dtype=torch.int64, device=device)
        # Same bins, negative values only. The total histogram cannot answer an asymmetric
        # question like "how many x < mean - 3*std" because it discards the sign; keeping
        # the negative half separately makes both tails recoverable (hist_pos = hist - hist_neg).
        self.hist_neg = torch.zeros(LOG2_BINS, dtype=torch.float64, device=device)

    def update(self, flat):
        """flat: (rows, n_channels) float32 on the accumulator's device.

        Deliberately free of any `.item()` / boolean-mask indexing: both would sync the
        host with the GPU on every tap. Everything below stays a queued kernel.
        """
        self.calls += 1
        self.n_nonfinite += (~torch.isfinite(flat)).sum()
        flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        f64 = flat.to(torch.float64)
        a = f64.abs()
        self.count += int(flat.numel())          # a Python int; no sync
        self.sum += f64.sum()
        self.sumsq += (f64 * f64).sum()
        self.sumabs += a.sum()
        f2 = f64 * f64
        self.sum3 += (f2 * f64).sum()
        self.sum4 += (f2 * f2).sum()
        self.vmin = torch.minimum(self.vmin, f64.min())
        self.vmax = torch.maximum(self.vmax, f64.max())
        self.ch_absmax = torch.maximum(self.ch_absmax, a.amax(dim=0).to(torch.float32))
        self.ch_sumabs += a.sum(dim=0)
        n_zero = (a == 0).sum()
        self.n_zero += n_zero
        # Exact zeros would be log2 -> -inf. Clamp them into bin 0 and subtract their
        # count back out at the end, which avoids the masked gather a[a > 0] would need.
        idx = (torch.log2(a.clamp_min(2.0 ** LOG2_LO)) - LOG2_LO)
        idx = (idx * (LOG2_BINS / (LOG2_HI - LOG2_LO))).to(torch.int64).clamp_(0, LOG2_BINS - 1)
        idx_flat = idx.reshape(-1)
        self.hist += torch.bincount(idx_flat, minlength=LOG2_BINS)
        self.hist[0] -= n_zero
        # bincount() only takes float weights, so this counts in float64 -- exact up to
        # 2^53, far beyond any element count a run can reach. Zeros carry weight 0, so
        # hist_neg needs no zero correction.
        self.hist_neg += torch.bincount(
            idx_flat, weights=(f64 < 0).reshape(-1).to(torch.float64), minlength=LOG2_BINS)

    def finalize(self):
        n = max(self.count, 1)
        mean = (self.sum / n).item()
        m2 = (self.sumsq / n).item()
        m3 = (self.sum3 / n).item()
        m4 = (self.sum4 / n).item()
        var = max(m2 - mean * mean, 0.0)
        # Raw moments -> central moments. Accumulating central moments directly would
        # need the mean up front, which a single streaming pass does not have.
        mu3 = m3 - 3 * mean * m2 + 2 * mean ** 3
        mu4 = m4 - 4 * mean * m3 + 6 * mean * mean * m2 - 3 * mean ** 4
        sd = math.sqrt(var)
        skew = mu3 / sd ** 3 if sd > 0 else float("nan")
        # Excess kurtosis: 0 for a normal, positive for heavier tails.
        excess_kurt = mu4 / (var * var) - 3.0 if var > 0 else float("nan")
        hist_c = self.hist.detach().cpu()
        hist_neg_c = self.hist_neg.detach().cpu().round()
        n_zero = int(self.n_zero)
        lo_tail, hi_tail = outlier_counts(hist_c, hist_neg_c, n_zero,
                                          LOG2_LO, LOG2_HI, SIGMA_K, mean, sd)
        n_out = lo_tail + hi_tail
        return {
            "skew": skew,
            "excess_kurt": excess_kurt,
            # 3-sigma outliers, the z-score convention: |x - mean| > SIGMA_K * std.
            "sigma_k": SIGMA_K,
            "outlier_lo": mean - SIGMA_K * sd,      # lower bound of the inlier range
            "outlier_hi": mean + SIGMA_K * sd,      # upper bound
            "n_outlier_lo": lo_tail,                # elements below outlier_lo
            "n_outlier_hi": hi_tail,                # elements above outlier_hi
            "n_outlier": n_out,
            "outlier_pct": 100.0 * n_out / n,
            "dtype": self.dtype,
            "shape_tail": self.shape_tail,
            "calls": self.calls,
            "count": self.count,
            "n_zero": int(self.n_zero),
            "n_nonfinite": int(self.n_nonfinite),
            "min": self.vmin.item(),
            "max": self.vmax.item(),
            "absmax": max(abs(self.vmin.item()), abs(self.vmax.item())),
            "mean": mean,
            "std": math.sqrt(var),
            "mean_abs": (self.sumabs / n).item(),
            "ch_absmax": self.ch_absmax.detach().cpu(),
            "ch_meanabs": (self.ch_sumabs / max(self.count // self.n_channels, 1)
                           ).to(torch.float32).detach().cpu(),
            "hist": hist_c,
            "hist_neg": hist_neg_c.to(torch.int64),
            "hist_lo": LOG2_LO, "hist_hi": LOG2_HI, "hist_bins": LOG2_BINS,
        }


class _Accumulator:
    def __init__(self, stride=1, max_rows=None, device=None):
        self.stride = max(int(stride), 1)
        self.max_rows = max_rows          # subsample rows per call; None = use all
        self.device = device
        self.stats = {}
        self._seen = {}

    def add(self, key, name, t):
        c = self._seen.get(key, 0)
        self._seen[key] = c + 1
        if c % self.stride:
            return
        with torch.no_grad():
            flat = t.detach()
            trailing = _CHANNEL_TRAILING_DIMS.get(name, 1)
            trailing = min(trailing, flat.dim())
            n_channels = 1
            for d in flat.shape[flat.dim() - trailing:]:
                n_channels *= int(d)
            flat = flat.reshape(-1, n_channels).float()
            if self.max_rows is not None and flat.shape[0] > self.max_rows:
                sel = torch.randint(flat.shape[0], (self.max_rows,), device=flat.device)
                flat = flat.index_select(0, sel)
            if self.device is not None:
                flat = flat.to(self.device)
            st = self.stats.get(key)
            if st is None:
                st = self.stats[key] = _Stat(t, n_channels, flat.device)
            elif st.n_channels != n_channels:
                raise RuntimeError(
                    f"tap {key!r} changed channel count {st.n_channels} -> {n_channels}")
            st.update(flat)

    def finalize(self):
        return {k: v.finalize() for k, v in sorted(self.stats.items())}


@contextmanager
def accumulate(stride=1, max_rows=None, device=None):
    """Fold every tapped tensor into running statistics for the duration of the block.

    stride    -- only fold every N-th call at each tap site (cheap subsampling).
    max_rows  -- randomly subsample this many (token, ...) rows per call.
    device    -- where the accumulators live; default is the tapped tensor's device.

    Yields the live _Accumulator; call `.finalize()` (or tap.save()) after the block.
    """
    global _ACC
    if _ACC is not None:
        raise RuntimeError("tap.accumulate() is not reentrant")
    _ACC = _Accumulator(stride=stride, max_rows=max_rows, device=device)
    try:
        yield _ACC
    finally:
        _ACC = None


def save(acc, path, meta=None):
    """Write finalized statistics to `path` for mamba3_distribution/distribution.py."""
    torch.save({"impl": IMPL, "meta": meta or {}, "stats": acc.finalize()}, path)
    return path
