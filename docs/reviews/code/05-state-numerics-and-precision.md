# 05 — State, numerics, and precision

## 1. State and caching

**What is modelled.** `CompressedCache` (`shared.py:58-72`) carries a compressed key/value
store, its entry positions, the selection pool, and the SWA cache (`k/v` plus their
absolute positions). `CSA2SharedState` (`shared.py:75-88`) carries the per-forward
compressed store plus the seed flag. Both are passed in and returned, never stored.

**What is not returned.** All three backbones end with `ModelOutput(..., new_state=None)`
(`backbone.py:131, 389, 647`); `ModelState` is never constructed; `CSA2.forward_shared`
documents `use_cache` as a no-op (`sparse.py:703-706`). The HF adapter forwards
`past_key_values=output.new_state` (`modeling_project.py:105-111`), so the external API
advertises a cache that is always `None`.

**Consequence.** `generate()` is `O(n²)`; every incremental-decode path in the repo is
untested because there is nothing to increment. The scientific review already flags the
consequence for BLT (one patch behind) and CED (support re-read per layer); the code
consequence is simpler: the pieces exist (`_update_swa_pos`, `CompressedCache`) and are
never assembled.

**Fix.** Either (a) implement `decode_step` on `CausalDecoder` that takes
`CompressedCache`, appends one position, and returns the updated cache as `new_state`
(+ one test comparing the last logit of stepwise decode against a full forward), or
(b) raise on `use_cache=True` / `past_key_values is not None` at the backbone and adapter.
(a) is the only version that makes the 32k design point usable; (b) is one hour of work
and removes the lie.

**One piece of module state does exist.** `EngramMemory._multiplier_cache`
(`memory.py:750, 803-807`) is a CPU-pinned dict mutated inside `forward`. It is not a
buffer, not in `state_dict`, grows without bound with distinct `layer_id`, and is a data
race under threaded calls. It is also the mechanism of the CUDA device-mismatch defect
in §5.

## 2. GDN-2: recurrent vs chunk parity

The two reference paths claim equivalence (`gdn2.py:34-35, 495-497`), and the test asserts
it in fp64 with `atol=rtol=1e-10` (`tests/test_gdn2.py:27-32`). Within the tested regime
they agree: measured max difference `8.9e-16` at `T = C = 8`, `g = -8`.

**Outside that regime the chunk path breaks.** `_chunk_single` divides by cumulative
decay:

```python
gamma  = torch.exp(torch.cumsum(gc, dim=0))   # [C, K]
k_bar  = kc / gamma                            # gdn2.py:245
k_tail = (gamma_c / gamma) * kc                # :254
```

When the per-chunk sum of `|g|` exceeds ~87 (fp32) or ~709 (fp64), `gamma` reaches `0`,
`k_bar` becomes `inf`, and `q_gamma @ k_barᵀ` is `0 · inf = NaN`. Measured:

| `T` | `C` | `g`/step | dtype | recurrent NaN | chunk NaN |
| --- | --- | --- | --- | --- | --- |
| 8 | 8 | -8 | fp64 | 0 | 0 (max diff 8.9e-16) |
| 64 | 64 | -8 | fp32 | 0 | **512** |
| 64 | 64 | -20 | fp32 | 0 | **512** |
| 64 | 64 | -20 | fp64 | 0 | **512** |
| 8 | 8 | -100 | fp64 | 0 | **8** |

The recurrent path never divides by the accumulated decay, so it decays to zero
gracefully. `_compute_dtype` upcasts fp32 inputs to fp32 (`85-88, 369-384`), so bf16
training does **not** protect the chunk path.

**Also:** `ced_gdn2_backend` defaults to `"naive"`, which dispatches to the *recurrent*
function (`gdn2.py:583-584`), which passes `chunk_size=None` (`474`). The
`ced_gdn2_chunk_size` knob therefore has no effect at the default backend — a config field
that changes nothing until another field is changed.

**Fix.** Stabilise by subtracting the per-chunk max of `cumsum(g)` before the exponential,
folding the removed scale into `gamma_c`; add the `g=-8, T=C=64, fp32` case to
`tests/test_gdn2.py`; document (or honour) `chunk_size` under `naive`.

**Cost of the recurrent path** (what actually runs): an `O(4·T·HV·K·V)` scan with one
Python iteration and one graph node per token per head (`207-211`), no chunking or
checkpointing, `torch.zeros(B, T, HV, V)` allocated up front (`387`), and a full
fp32 upcast/downcast round trip of six tensors (`369-384, 429`). The output projection is
the only place a kernel would pay off; the scan itself is the reason the memory branch is
the most expensive path in the model.

## 3. Masking and positions

The position-derived mask is the strongest part of the numerics (`02` §3). Two things to
know when changing it:

- **`reuse` mode broadcasts the previous step's pool** when `q_len` differs
  (`sparse.py:820-830`). This is documented as an approximation, and it is what makes the
  looped decoder cheap; do not "fix" it into a recompute without measuring.
- **The cache + `is_causal` combination raises** (`attention.py:737-740`), and a windowed
  cache without explicit `past_positions` raises (`705-707`). Any fused-attention rewrite
  must preserve both — they are the guards that keep SWA replay correct.

`_resolve_positions` falls back to `_past_total_len`, which calls `.item()`
(`shared.py:115-130`), once per layer in the cached path. That is the sync in §6.

## 4. Precision policy

| Site | Policy | Risk |
| --- | --- | --- |
| Attention softmax | fp32, cast back (`attention.py:777`) | correct, costs a full-size fp32 copy |
| GDN-2 core | fp32 always, fp64 if any input is fp64 (`gdn2.py:85-88`) | no bf16 path; 2× memory |
| GDN-2 decay | `A_log`/`dt_bias` created fp32 and re-forced `.float()` (`818-820, 906`) | `model.double()` silently keeps an fp32 decay |
| CVPE offset MLP | runs in the module dtype after `.to(bf16)` (`positioning.py:188-215`) | positional feature degrades first |
| `AttentionSink` | fp32, no max subtraction (`compression.py:551-555`) | NaN for logits > 88 |
| fast-weight cache | snapshotted in activation dtype (`fast_weights.py:389, 1019`) | bf16 forward then fp32 forward downcasts history |
| `causal_conv1d_naive` | inherits input dtype | fine |

Only two rows are defects today (`AttentionSink`, and the fp32 decay under
`model.double()`); the rest is a policy that should be stated in one place instead of
inferred per module.

## 5. Verified defects in the unwired mechanisms

These are S2 by reachability (`03` S2-1) but S1 if anyone wires the knob:

1. **`associative_losses` crashes for `B > 1`.** `_step` returns a 0-dim
   `loss.detach()` (`fast_weights.py:308`); `342` reshapes it to `(B, 1)` →
   `RuntimeError: shape [B, 1] is invalid for input of size 1`. The docstring promises
   `(B, T)`.
2. **Engram hashing is CPU-only.** `_odd_multipliers` builds a CPU tensor
   (`memory.py:103-106`) and is cached on CPU (`806`); `850-852` multiplies it by
   on-device ids → device mismatch on CUDA. `self.offsets` *is* a buffer and moves
   correctly, so the bug is isolated to the multipliers.
3. **`_SsdTable` aliases an mmap as an `nn.Parameter`** (`memory.py:217`). Any `.to()`
   rebinds `weight` to a copy while `_mmap` keeps writing the file; `normal_init_` then
   updates the file and not the live weights. There is no `flush()` after optimizer
   steps, and the file name is per-seed, not per-layer (`742`), so two Engram layers in
   one model share one table.
4. **`LivPerceiver` sparse mode applies `o_proj` twice** — CSA2's `grouped_out` already
   projects (`liv.py:855-861`), then `attn.project_heads` runs again (`864, 876`).
5. **`create_graph = torch.is_grad_enabled()`** (`fast_weights.py:373, 1003`): under
   reentrant checkpointing the recompute runs with grad disabled, so the inner trajectory
   is silently detached; under `torch.compile` the value is baked at trace time.
6. **`_SsdTable.__init__` reuses any file with a matching byte size** (`205-209`), so a
   shape change silently reinterprets stale weights.
7. **`ContinuumMemorySystem.should_update` is never called** (`1339` vs `1365`), so
   `update_intervals` is inert. `HopeAttentionBlock` additionally calls SDPA with
   `is_causal=True` and no mask or window (`1479`).

## 6. Sync ledger

One device→host sync per layer per forward:

| Site | What it waits for | Removable? |
| --- | --- | --- |
| `attention.py:696` | implied sequence length | yes — derive from positions |
| `shared.py:122` | past total length | yes — carry it in the cache |
| `compression.py:140` | last compressed end | yes — same |
| `blt.py:100` | max patch id | yes — `patch_ids` is validated anyway |
| `fast_weights.py:1243-1244` | `cu_seqlens` bounds | yes — keep as a tensor |
| `memory.py:233-236, 891` | bounds checks on ids | partially — check once at the boundary |

Each is `~µs` alone and a pipeline bubble in aggregate; at 40 layers the position ones
serialise every step. They are irrelevant on CPU and the first thing to remove when a GPU
run is profiled.

## 7. Initialisation and `reset_parameters`

- `GatedDeltaNet2.reset_parameters` is **not idempotent**: `A_log` and `dt_bias` are
  drawn only in `__init__` (`gdn2.py:818-820`), so a second call keeps the first draw
  while re-initialising everything else.
- `ParallelAttentionGDN2.__init__` calls `reset_parameters()` immediately after the
  sub-modules already initialised themselves (`parallel_hybrid.py:191-196, 229-245`) —
  harmless but doubles the RNG consumption, which matters for "same seed, same init"
  comparisons across commits.
- `LivDoubleGatedBase.__init__` explicitly calls the base `reset_parameters` to bypass the
  subclass override (`liv.py:66`), and the subclasses call it again (`332, 508`); a new
  subclass that forgets the second call gets an uninitialised mixer.
- `DenseFFN.reset_parameters` and three BLT sites resolve the method with `getattr` and
  skip silently (`ffn.py:324-332`, `blt.py:181-186, 364-369, 443-445`). The comment calls
  it a contract violation; the code does not.
- **Factored projections change the RNG stream.** `factorize_linear`
  (`linear_init.py`) draws fresh factors, so the same seed no longer reproduces the same
  tensors as a pre-budget commit. Scope: init only; nothing depends on checkpoint bytes.
- **A factored pair must be initialised as a pair.** `init_linear` matches variance by
  splitting the target between the factors (`Var(A)=1/in`, `Var(B)=std²·in/rank`) and
  clips only the input factor, because clipping both compounds the two truncation
  shrinkages and lands the product 12% below the dense draw it replaces (measured
  `0.007746` vs `0.008781`; after the fix `0.008843`). Any `module.apply(...)` initialiser
  must skip a `FactoredLinear`'s own factors or it will re-init them individually and
  destroy the match — `gdn2._initialize_weights` (`gdn2.py:846`) is the one such site and
  now marks both factors done in the same visit.
- **Per-column scales belong in the input factor.** For a projection initialised with
  different scales per input block (`ParallelAttentionGDN2.out_proj`, `1e-3` on the memory
  half), `init_linear_columns` puts the whole column scale on `A`, so `Var(W[:,j]) = s_j²`
  exactly; the SVD-truncation alternative lost the split's tail (measured memory half
  `6.3e-4` against a `1e-3` target).
