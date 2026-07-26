"""Opt-in capture of Mamba-3 forward-pass intermediates for cross-implementation
numerical comparison. A no-op unless explicitly started.

Used by mamba3_compare_nano4/probe.py to read named intermediates straight out of the
real Mamba3.forward(), instead of re-deriving forward()'s math by hand (which drifted).

Follows the module-global-override pattern already used by mamba_ssm/utils/determinism.py.
"""

from contextlib import contextmanager

# The ONLY line that differs between branches. Each branch is pinned to one machine
# (`nano` -> H200, `rtx6000-adapt` -> RTX6000), so this identifies which implementation
# is loaded. run_side.py asserts it against --tag, which makes it impossible to compare
# a branch against itself after a stray checkout or a bad PYTHONPATH.
IMPL = "nano"

_ACTIVE = None


def is_active():
    return _ACTIVE is not None


def expect_impl(name):
    """Raise unless the loaded mamba_ssm is the implementation the caller expects."""
    if IMPL != name:
        raise RuntimeError(
            f"expected the {name!r} implementation but imported {IMPL!r} "
            f"(check the checked-out branch and PYTHONPATH)"
        )


def tap(name, t):
    """Record `t` under `name` when capture is active; always return `t` unchanged
    so this can be inserted into forward() without altering the computation."""
    if _ACTIVE is None:          # no `global`: this only reads the name and
        return t                 # mutates the dict, it never rebinds _ACTIVE
    if name in _ACTIVE:
        raise KeyError(f"tap {name!r} recorded twice in one capture")
    _ACTIVE[name] = t.detach().float().cpu()
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
