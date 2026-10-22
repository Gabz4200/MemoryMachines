# 02 — What is genuinely good

Ranked by how much work it saves and how hard it is to retrofit. Each entry says what
the code does, why it matters, and what not to break while fixing the defects in `03`.

## 1. One typed forward contract, one composition root

`ModelInput` / `ModelOutput` (`io.py:14-51`) cross every boundary: backbones, the
Lightning shell, the HF adapter, scripts. The backbone is a pure `nn.Module` that does
not know that Lightning, Hydra, `wandb`, or a training loop exist; `LitResearchSystem`
owns loss, step metrics, and optimizer configuration.

**Why it matters.** The entire model can be built from a 12-line config in a doctest
(`backbone.py:166-172`) and exercised in one call. That is why there are 1,547 tests and
why a reviewer can run any block in isolation. Most research repos of this size cannot
be instantiated without a trainer.

**Protect.** Keep `ModelOutput` additive. Do not put the loss or the optimizer step in
the backbone to "simplify" the Lightning module.

## 2. State is threaded through arguments, never stored on the module

`CompressedCache` and `CSA2SharedState` are dataclasses passed in and returned
(`shared.py:58-88`). CSA2's `mode` is a per-call override, not a mutated `self.mode`
(`sparse.py:718-722`), which is exactly what lets the looped decoder cycle
`full/reindex/reuse` over one shared layer without touching module state
(`ced.py:1320`). Per-block `reset_parameters()` completes the picture.

**Why it matters.** Re-entrancy, DDP safety, and checkpoint round-trips come for free.
The alternative (`self.cache = ...` inside `forward`) is the usual source of
"works with batch size 1, breaks with gradient accumulation" bugs. This is the single
strongest engineering decision in the repository.

**Protect.** Do not add mutable module state to make caching "easier" (see
`05-state-numerics-and-precision.md` §1 for the one place the opposite happened:
`EngramMemory._multiplier_cache`).

## 3. Masks are computed from absolute positions, and the fragile cases raise

`_window_mask` / `_causal_key_mask` compare `q_pos` against `k_pos`
(`shared.py:163-195`, `attention.py:31-51`) instead of assuming `arange`. Replaying a
truncated SWA cache therefore stays correct. When correctness cannot be guaranteed, the
code raises with a cause: cache + `is_causal` (`attention.py:737-740`), windowed cache
without `past_positions` (`705-707`), `seeded_store` / `reindex` / `reuse` preconditions
(`sparse.py:748-820`).

**Why it matters.** Position-relative masking is the difference between a windowed model
that works at prefill and silently corrupts at decode. It is also the piece most often
missing from sliding-window reimplementations.

**Protect.** This is the invariant to preserve when the attention core is rewritten
(`03-defects.md` S1-5). A fused SDPA path must take the same `q_pos`/`k_pos` and produce
the same mask, or must be rejected behind a capability check.

## 4. Initialisation is a deliberate, centralised policy

Truncated-normal at `initializer_range`; residual output projections zeroed or
depth-scaled at `std / sqrt(2N)` (`shared.py:20-55`); the fused parallel block keeps the
memory half of its `2d -> d` projection at `1e-3` so the block *starts* as attention and
grows the memory branch (`parallel_hybrid.py:242-261`, via `init_linear_columns` so the
split survives factoring); `init_residual_linears` is shared by attention, FFN, and BLT
local layers and accepts dense or factored projections (`blocks/linear_init.py`), so one
init policy covers both layer classes.

**Why it matters.** Init is the difference between a 40-layer stack that trains and one
that does not, and the memory-branch schedule is a real architectural decision encoded
where it acts. The scientific review argues the *value* of the default scale is wrong
(S3 in `../03-weaknesses.md`); that is a tuning dispute, not a design defect. The design
— one function, one place — is right.

**Protect.** When changing the scale, change `residual_init_scale`/`init_residual_linears`,
not the call sites. Do not let a new block invent its own `nn.Linear` defaults.

## 5. Guards are loud where the algebra is fragile

Cross-field length checks (`ced.py:950-971`), `kv_states` shape validation
(`ced.py:1131-1132`), mask-layout validation with an explicit `(Lq, Lk)` requirement
(`attention.py:99-115`), position bounds (`shared.py:425`), config whitelists
(`config_schema.py:592-601, 630-641`). Error messages name the offending value.

**Why it matters.** In a research codebase the expensive failures are silent, not loud.
These guards convert a week of "why is the loss flat" into a traceback.

**Protect.** The fix for the gaps in `03` is to *extend* this pattern (validate
`ced_attention_variant`, `beta_guard ⇒ use_qk_l2norm`, cache contracts), never to
replace it with defaults.

## 6. Numerical hygiene in the attention path

Softmax runs in fp32 and casts back (`attention.py:777`); `_sanitize_attention_weights`
zeros fully-masked rows instead of emitting NaN (`attention.py:119-140`); the exit step
is clamped and the clamp is documented as drift protection rather than pretending to be
logic (`ced.py:780-789`); `GroupedOutputProjection` uses per-group RMSNorm plus a
depth-scaled residual (`compression.py:558-649`).

**Why it matters.** Fully-masked rows are guaranteed to appear once padding, causal
windows, and sparse selection compose. The sanitizer is the reason a padded batch does
not poison the whole loss.

**Protect.** Any fused kernel must reproduce the zero-row behaviour, not just the shape.
The reference does *not* (`ops/references/attention_reference.py:160-163` returns NaN) —
see `04-ops-and-kernels.md` §4.

## 7. Registry plus injectable sublayers

`@register_model("cheapllm")` and friends (`registry.py:9-22`) make model selection a
string; every backbone accepts `encoder_layers`, `decoder_layers`, `global_projs`,
`decoder_ffns`. The seam is used by tests to swap in probe layers.

**Why it matters.** It is a real test seam, not decoration: a test can assert what the
decoder does with a known encoder output without building a 40-layer model.

**Protect.** Keep the injection points when the backbones are de-duplicated
(`03` S3-1). Do not replace them with `monkeypatch`-only tests.

## 8. Block reuse over copies

`ParallelAttentionGDN2` + `_build_branch` gives the Titans variant without a second
forward (`parallel_hybrid.py:198-218, 346-361`); `DenseFFN` parameterises kind, norm,
width, and update interval; `CSA2` parameterises the sharing mode; the local byte
encoder and decoder share `_LocalLayer` pieces; `CausalEncoderDecoderStack` is an
explicit facade. When a second implementation exists, it is usually a *parameter*, not a
fork.

**Why it matters.** Every fork doubles the maintenance cost of the hardest code in the
repo. The two places where a fork did happen (backbone masked-decode path, GDN-2
recurrent vs chunk) are exactly where defects accumulated.

**Protect.** Use this as the argument for extracting the duplicated generation path: the
repo's own convention already says parameterise, do not fork.

## 9. Review findings are encoded as tests

`test_*_review_guards.py`, `test_attention_position_guards.py`, `test_gdn2_paper.py`,
`test_hope_selfref_parity.py`, `test_ced_*`, 1,547 collected. The tests assert *contracts*
(causality probes, position guards, parity between two implementations), not
implementation details.

**Why it matters.** The chunk/recurrent GDN-2 parity test and the BLT causality probes
are the kind of test that catches a rewrite. They are also the reason the defects in
`03` are credible: each one is a place the test suite cannot reach (unwired module,
default-off path, or a contract nobody asserted).

**Protect.** When fixing S1-5 (attention core), keep the position-guard tests as the
acceptance criterion and add the same for the fused path.

## Caveat that belongs with the strengths

The import graph does not honour the layering: `blocks/__init__.py` eagerly imports all
45 exports, including private helpers, so importing `blocks.attention` pulls
`fast_weights` and `ops` (1,313 modules, 3.22 s cold). The separation is real at the
level of *dependencies* and fake at the level of *import cost*. Fixing that is
mechanical (`06` §5) and does not weaken anything above.
