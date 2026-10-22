# 07 — Remediation plan

Ordered by *risk removed per unit of change*, not by tidiness. Each phase is independently
shippable and has an acceptance check that fails before and passes after. Do not start a
phase before the previous one is green: several later fixes assume the earlier contract.

## Phase 0 — Stop lying to callers (½ day)

1. Backbones raise on `use_cache=True`, `past_key_values is not None`, and any
   `attention_mask` containing `False`/non-zero (unless the mask is all-keep).
2. HF adapter: same, with a message naming the missing capability.
3. `SwiGLU` even-width rule moved into `__post_init__`; `ced_attention_variant`,
   `hidden % mhc_num_streams`, and `beta_guard ⇒ use_qk_l2norm` added to boot validation.
4. Delete `ModelState` or wire it (Phase 2).

**Acceptance:** `pytest tests/test_config_validation.py tests/test_hf_transformers.py
tests/test_backbone.py -q` green; a manual `model.generate(..., use_cache=True)` raises
with the intended message.

## Phase 1 — Make the config honest (1–2 days)

1. Decide §1 of `06` (wire / isolate / delete) for `liv`, `memory`, `rkan`, `linears`,
   and for the Titans Blocks.
2. Delete the config fields of anything isolated or deleted; add a `__post_init__` rule
   for anything kept but conditional.
3. Remove `blocks/__init__.py`'s private exports; make the barrel lazy.

**Acceptance:** the config-field audit test from `06` §6 passes (every field read by at
least one `src/` module); import of `blocks.attention` no longer imports `fast_weights`
(`python -X importtime` shows < 300 modules).

## Phase 2 — Cache or no cache (3–5 days)

1. Add `CausalDecoder.decode_step(hidden_states, cache, position_ids)` returning
   `(logits, new_cache)` using `CompressedCache` + `_update_swa_pos`. Thread the GDN-2
   recurrent/conv state through `CSA2SharedState` (fixes `03` S2-8 as a side effect).
2. Return it as `ModelOutput.new_state`; wire the HF adapter both ways.
3. Test: stepwise decode of 32 tokens equals the last 32 logits of a full forward
   (`atol=1e-5` fp32), for `ced_attention_gdn2_parallel=True` and `False`.

**If the estimate slips:** ship only step 1 of Phase 0 (raise loudly) and move on. The
honest-refusal version is one hour; a wrong cache is worse than none.

## Phase 3 — Attention core (1 week, highest payoff)

1. Build the additive bias from the existing `q_pos`/`k_pos` masks (reuse
   `_window_mask`, `_causal_key_mask`; do not re-derive).
2. Call `F.scaled_dot_product_attention(..., attn_mask=bias, enable_gqa=True)`.
3. Keep `_score_qk` + explicit softmax as `attention_backend="naive"`;
   default `"auto"`.
4. Remove `repeat_interleave` on the fused path; keep it on the reference path.
5. Delete the function-local reference imports (S1-1) and hoist anything still needed.

**Acceptance:** `tests/test_attention_position_guards.py`,
`tests/test_attention_windows.py`, `tests/test_attention_mask.py`,
`tests/test_ced_loop.py` unchanged and green; new test asserting fused-vs-reference
`atol=1e-5` for four cases (no cache, cache, window, fully-masked row); peak memory at
`L=4096, H=16` reduced by ≥ 4× (record the number in `05`).

## Phase 4 — Ops: finish or delete (1–3 days)

**Finish** (only if a kernel is actually being written):

1. Route the model through `dispatch_*` with `backend` from config.
2. Fix the four semantic differences in `04` §4 *before* enabling `auto`.
3. Resolve loaders once per `(op, backend)`; catch only import/`None`; propagate call-time
   exceptions with their original type.
4. `register_autograd` in the same change as the first kernel; fix the
   `Tensor` vs `(Tensor, Tensor)` return contract; give the update op a `custom_op` with
   `mutates_args=("cache",)`.

**Delete** (default choice):

1. Remove `ops/dispatch.py`, `ops/registry.py`, `OpImpl`, and the empty kernels tree.
2. Move the two reference modules next to their consumers
   (`blocks/attention.py`, `blocks/causal_conv1d.py`) and drop the `ops/` package.

Either way, one mask convention and one NaN policy (`04` §4).

**Acceptance:** no `except Exception` remains around a kernel call; `grep -rn
"dispatch_op(" src/` returns either a real model call site or nothing.

## Phase 5 — Deduplicate the model path (2–3 days)

1. Extract `_build_decoder_stack(...)` and `masked_decode(...)` from the two backbones
   into `ced.py`; both call them.
2. Allocate exactly one `CEDGlobalKVProjection`; drop the per-layer list unless injected.
3. One `forward_shared` signature in `protocols.py`, implemented by CSA2 and both hybrid
   blocks; `_SharedStateAdapter` raises on unsupported kwargs.
4. `CausalDecoder.forward` either returns the full sequence or is renamed to
   `prefill`-only.

**Acceptance:** `tests/test_backbone.py`, `test_blt.py`, `test_ced_*` unchanged and
green; `git diff --stat` shows fewer net lines; checkpoint size for a 20-layer decoder
drops by the 19 projections (record the number in `05`).

## Phase 6 — GDN-2 numerical safety (2 days)

1. Stabilise the chunk path (max-shifted `cumsum(g)`), or gate
   `backend in ("chunk","wy_fast")` behind a warning until it is stable.
2. Honest `chunk_size`: honour it under `naive` or reject a non-default value.
3. Add the `g=-8, T=C=64, fp32` regression and an `associative_losses(B=2)` test.

**Acceptance:** `pytest tests/test_gdn2.py tests/test_gdn2_paper.py -q` green including
the new strong-decay case; the chunk/recurrent max difference at moderate decay stays
`< 1e-12` (fp64).

## Phase 7 — Config split (3–5 days, after Phases 1 and 5)

Split into `AttentionConfig` / `MemoryConfig` / `CEDConfig` / `BLTConfig`, compose, keep
flat read-only properties for one release. Type block constructors against the narrow
config.

**Acceptance:** `GroupedQueryAttention` no longer imports `MemoryMachineConfig`; no block
reads a field outside its own group (audit test); all configs in `configs/` still compose.

## What I would *not* do now

- **Rewrite `memory.py`/`liv.py` before deciding their fate.** Fixing defects in code
  that may be deleted is negative work.
- **`torch.compile` the model.** It will break on the sync points, the harness dict
  lookups, and the function-local imports; Phase 3 removes most of those, and compiling
  before Phase 3 optimises the wrong graph.
- **Chase line coverage.** The gaps are structural (§6 of `06`); three meta-tests are
  worth more than 200 unit tests over dead paths.
- **Rename modules for clarity** before the consolidation in Phase 5 — the rename diff
  will hide the semantic changes.

## Ordering rationale

Phase 0 removes the failure modes that produce *wrong results* for external callers
(mask, cache) at near-zero cost. Phase 1 shrinks the surface that Phases 5–7 must carry.
Phase 2 makes the 32k context real. Phase 3 is the only change that alters the cost class
of the model. Phase 4 is a decision, not a refactor, and should follow Phase 3 so the
kernel boundary is defined by the path that actually runs. Phase 5 is pure
deduplication and becomes trivial once the fused path exists. Phase 6 is an isolated
numerical fix on the memory branch. Phase 7 is the cleanup that makes the rest stick.
