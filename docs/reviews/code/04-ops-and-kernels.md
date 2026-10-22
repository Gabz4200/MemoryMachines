# 04 — The ops layer and the kernel boundary

## 1. Inventory

```
ops/__init__.py                33    re-exports (dispatch_*, reference_*)
ops/dispatch.py               137    thin per-op adapters over dispatch_op
ops/registry.py               217    OpSpec, backend normalisation, loader/call, auto fallback
ops/attention.py              114    custom_op + fake (body raises NotImplementedError)
ops/causal_conv1d.py          110    custom_op + fake (body raises), naive wrapper, 3 OpSpecs
ops/references/attention*     166    reference_qk_circuit, build_sdpa_bias, SDPA reference
ops/references/causal_conv1d* 246    reference fwd / update / state recompute
kernels/taichi/                 —    loader; returns None when taichi or module missing
kernels/cpu_loader.py           —    loads native/cpu/*.so; native/cpu/ is empty
kernels/native/torch/*.cpp      —    registration.cpp fully commented out, autograd.cpp 3 lines
```

Nothing under `kernels/native/` is compiled, `native/cpu/` contains no sources, and both
loaders therefore always return `None`. `tests/test_ops_registry.py:203-207` asserts
exactly that, so the empty state is intentional — but it means the entire dispatch stack
has never had a backend to dispatch to.

## 2. Backend semantics as implemented

| `backend` | resolves to | failure mode |
| --- | --- | --- |
| `naive` | `spec.naive` (pure torch) | none |
| `auto` | taichi → cpu → naive | any exception ⇒ warning + naive |
| `gpu`, `taichi` | taichi loader, arch from `torch.cuda.is_available()` | `NotImplementedError` |
| `cpu`, `compiled` | CPU extension (`compiled` is a *CPU* alias) | `NotImplementedError` |

Two naming defects worth fixing on their own: `gpu` is served by the Taichi loader (so
"gpu" is not a device choice), and `compiled` means "CPU C++ extension", which will read
as `torch.compile` to every new contributor. Rename to `taichi` and `cpp`, and drop the
aliases.

## 3. The layer is orphaned

- No call site in `src/` (`grep -rn "dispatch_op(" src/` → `ops/registry.py` only).
- The model imports the reference directly: `blocks/attention.py:824-829, 977-983`,
  `blocks/sparse.py:249-255`, `blocks/causal_conv1d.py:15-20`.
- Both custom-op bodies raise (`ops/attention.py:37`, `ops/causal_conv1d.py:32`).

The reference docstrings assert the opposite of the code:
`ops/references/attention_reference.py:4` says "reached ONLY via explicit import — never
as a silent fallback", and the model reaches it via a function-local import on every
attention call. `ops/references/causal_conv1d_reference.py:4-5` says "reached ONLY via
`dispatch_causal_conv1d(backend='naive')`", and `ShortConvolution` imports it at module
scope.

Consequences beyond dead code:

- **A second implementation cannot be validated.** There is only the reference, so the
  "parity between reference and optimised path" rule has nothing to compare.
- **The dispatcher cannot be tested against the model.** `dispatch_universal_pe_attention`
  builds its own dense causal bias (`build_sdpa_bias`), so its output does not equal the
  model's position-derived masking even for the same inputs (`§4`).
- **Function-local imports** in the hottest function in the repo add a `sys.modules` hit
  per layer and a `torch.compile` graph break.

## 4. The reference is not semantically equivalent to the model path

Four differences, all observable:

1. **Mask polarity.** Reference: bool `True` = participate
   (`attention_reference.py:44-50`). Model: bool `True` = blocked
   (`blocks/attention.py:111-113`). The same tensor means opposite things.
2. **Causal shape.** Reference builds a dense upper-left `tril` bias
   (`attention_reference.py:43-44`). The model compares `q_pos`/`k_pos`
   (`blocks/attention.py:31-51`). Equal only when `Lq == Lk` and positions are `arange`;
   different for every cached or replayed call.
3. **Fully-masked rows.** Reference returns NaN (no sanitisation,
   `attention_reference.py:160-163`); the model zeros the row
   (`blocks/attention.py:119-140, 902`).
4. **Dropout.** Reference hardcodes `train=True` (`attention_reference.py:162`), so it is
   stochastic in `eval()`. The model uses `nn.Dropout`.

Until these converge, "auto falls back to the reference" would change numerics in three
of the four cases. That is a correctness bug waiting for the first kernel.

## 5. Failure handling

`_try_load_and_call` (`registry.py:101-116`) wraps load and call in one
`except Exception`; `_dispatch_auto` (`155-169`) converts it into a warning plus the
reference. Explicit backends re-raise as `NotImplementedError` (`171-185`), losing the
original type. `kernels/taichi/__init__.py:60-65` catches `ImportError` around
`importlib.import_module` and returns `None`, so an `ImportError` raised *inside* an
existing kernel module is reported as "loader returned None" rather than as a missing
dependency.

The dispatcher is also not traceable: dict lookup, `f-string` warning, exception
handling, and dynamic tuple returns (`138-207`). Under `torch.compile` it will graph-break
before it ever reaches a kernel.

**Fix.** Resolve once per `(op, backend)` and cache the outcome; catch `ImportError` and
`loader is None` only; let call-time exceptions propagate; emit the warning once per
process, not per call; and keep `auto`'s fallback but record it in a module-level
`BACKEND_IN_USE` that tests can assert.

## 6. Custom-op contract defects

- **Return type mismatch.** `causal_conv1d_op` is annotated `-> torch.Tensor`
  (`ops/causal_conv1d.py:31`), the naive implementation returns
  `tuple[Tensor, Tensor | None]` (`66-86`), and `dispatch.py:44` annotates a tuple. A
  compiled kernel's unpack contract is undefined, and `register_fake` returns
  `empty_like(x)` (`54`), so it cannot represent `output_final_state=True` at all.
- **`mutates_args=()` while the reference mutates `cache` in place**
  (`causal_conv1d_reference.py:232-235`). The update variant has no `custom_op`, so its
  in-place semantics cannot be expressed.
- **No `register_autograd` anywhere in `src/`** (`grep -rn register_autograd src/` → no
  Python hits; one comment in `native/torch/autograd.cpp`). A landed kernel would be
  non-differentiable, and `vmap`/`grad` over the op unsupported. The module docstrings
  call this "future" wiring; a completed kernel needs it in the same change.
- **Fake kernels validate less than the reference.** The attention fake checks only
  `dim() >= 3` (`ops/attention.py:58-60`), so tracing accepts head counts and GQA
  configurations that the real op would reject; it raises `RuntimeError` where the
  reference raises `ValueError`.

## 7. Import direction

`ops/` never imports `models/` — correct. `models/` imports `ops.references` and a
private helper (`_check_activation`, `blocks/causal_conv1d.py:16`) — incorrect if `ops/`
is meant to sit below `models/`. `ops/references` is the *lowest* layer semantically
(pure functions over tensors), but it lives under the layer that is supposed to depend on
it.

**Fix (with S1-1).** If the dispatcher is kept: move references to
`models/blocks/reference/` or `ops/reference/` with an explicit public surface, and have
`ops/dispatch` depend downward on them. If the dispatcher is deleted: move the reference
functions next to their only consumer (`blocks/attention.py`, `blocks/causal_conv1d.py`)
and delete `ops/` above them.

## 8. Duplication across the boundary

| Concept | Model copy | Ops copy |
| --- | --- | --- |
| QK + CVPE + scale | `attention.py:383-416` `_score_qk` | `attention_reference.py:57-64` `reference_qk_circuit` |
| Mask + softmax + reduce | `attention.py:143-178`, and again `856-918` | `attention_reference.py:29-54` |
| Causal conv | `blocks/causal_conv1d.py` (delegates) | `causal_conv1d_reference.py` (owner) |

The conv case is the pattern to copy: one owner, one thin wrapper, no second math.
Attention is the pattern to fix.

## 9. Verdict

The ops layer is a well-shaped *skeleton* for a kernel boundary — `OpSpec`, explicit
backend names, `custom_op` + `register_fake` + `mutates_args`, a reference implementation
per op — attached to nothing, with a fallback policy that would hide the first real bug
in whatever gets attached. Either finish it (one kernel, one autograd registration, one
parity test) in the same change that routes the model through it, or delete it. The
current state costs maintenance and teaches the wrong lesson about how the model runs.
