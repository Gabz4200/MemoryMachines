# 03 — The bad points, ranked by severity

Format per item: **claim → why it is a science problem → minimal fix.**
"Minimal fix" means the cheapest change that makes the claim testable, not the cheapest
change that makes the symptom disappear.

---

## S1 — The default configuration is nine simultaneous hypotheses

**Symptom.** `cvpe`, `frvm`, `csa_use_sink`, `residual_init_scale`, `dyT` in both norm
slots, `swa_bounded_replay`, parallel hybrid, alternating Titans, and sparse selection are
all on by default, with `tie_word_embeddings=True` as a tenth. None has a recorded ablation.

**Why it is a science problem.** A perplexity number from this configuration supports no
claim. Worse, the mechanisms interact in ways that make attribution impossible even
post hoc: FRVM changes the value path that CVPE's gradients flow through; DyT changes the
scale that `residual_init_scale` was tuned against; the memory branch's contribution depends
on the window size, which depends on sequence length. Any single-mechanism story told from
this configuration is unfalsifiable.

**Minimal fix.** Declare a *reference ablation configuration* in `configs/` (a named Hydra
group) in which every contested mechanism is off, then one config per mechanism turning
exactly one on. Twelve configs, one harness, one table. Until that exists, the repository
is a hypothesis generator.

---

## S2 — The global memory branch is probably not being trained

**Symptom.** The memory branch (GDN-2 / Titans) is the only path for information beyond the
attention window, but its output half is initialised at `1e-3` and the attention window is
256 tokens. On sequences shorter than roughly 2× the window, tokens beyond the window are
rare, so the gradient that would justify the memory branch is rare too.

**Why it is a science problem.** If the branch is dead, the architecture is "SWA attention
with expensive decoration," and every claim about hybrid memory is unsupported — while the
code still looks correct and the loss still goes down. This is the most dangerous failure
mode in the repository because *nothing in the code or logs reveals it*.

**Minimal fix.** Two measurements, both cheap: (a) at test time, zero the memory half of
`out_proj` and report the perplexity delta; (b) report the L2 norm of the memory half
relative to the attention half during training. If the delta is ~0 and the ratio stays near
`1e-3`, the branch is decorative. Then either train at ≥4× window or drop the branch.

---

## S3 — The default initialisation puts the model in a near-dead regime

**Symptom (measured).** With `initializer_range=0.006` (the project default), CheapLLM
logits have standard deviation:

| hidden | encoder + decoder layers | logit std |
| --- | --- | --- |
| 32 | 4 + 3 | 5.2e-10 |
| 128 | 4 + 4 | 8.6e-08 |
| 256 | 8 + 8 | 8.2e-07 |

The same model with `initializer_range=1.0` produces logits of order `1e17` on the same
input. So output scale is a steep function of `init_range` and depth, in both directions.

**Why it is a science problem.** At 1e-10 the softmax is uniform to numerical precision:
cross-entropy is pinned at `log V`, gradients through the head are ~1e-10, and the first
thousands of steps learn only the unigram bias. Combined with the zero-init philosophy
(§7 of `02-strengths.md`), the model starts with *every* structural mechanism at exactly
zero and *also* with no signal scale. That is not "vanilla attention at init"; it is a dead
start, and any hyperparameter conclusion drawn from it (learning rate, warmup, depth
scaling) is a conclusion about the dead regime.

At 1e17 the opposite failure shows the other half of the same defect: **there is no
variance-preserving normalization anywhere in the residual path.** DyT is elementwise
`tanh(x·α + β)·γ`; it bounds the *input to a mixer* but never rescales the residual stream,
so the residual magnitude is a free function of depth and init. GPT-2's `0.02/√(2N)`
convention exists precisely to keep that scale at O(1); a single global `0.006` cannot.

**Minimal fix.** Adopt the standard split: embedding/head at `initializer_range`, residual
output projections at `initializer_range/√(2N)`, and verify that a freshly initialised model
has logits of order 0.1–1 at the target width. This is a one-line change with a measurable
acceptance test, and it invalidates nothing else.

---

## S4 — CVPE is not a positional encoding, and it is weakest exactly where position matters most

**Symptom.** `b_qk = γ(q)·f(r/q) + β(q)` with `r = q − k`, `f` bounded in `[0,1)`
(optionally `log1p(9t)/log 10`), `γ, β` per-head linear maps of the query only. No RoPE, no
absolute embedding anywhere in the model.

**Why it is a science problem.** Four distinct issues, in increasing order of severity.

1. *It is a prior, not a representation.* The bias enters logits before the softmax, so
   position influences *which* tokens are read but never *what is stored*. It cannot enter
   the residual stream, cannot be transformed by an FFN, and cannot be used multiplicatively
   by a later layer. A model with no positional representation in the residual stream must
   re-derive position at every layer from attention patterns alone.
2. *It is a per-head scalar.* The bias has no head-dimension structure, so it cannot express
   the phase/rotation geometry that makes RoPE compose across layers and generalize to
   longer contexts. Relative position here is a lookup table over `r/q`, not a group action.
3. *It saturates.* `f ∈ [0,1)` with a log warp: as `r/q` grows, `f → 1` and `∂f/∂r → 0`.
   At long range the bias is flat and carries no distance information; only `β(q)` remains,
   which is position-independent. So the model's positional resolution is concentrated at
   *small* distances.
4. *Saturation lands exactly where the architecture needs it least.* Local distance is
   handled by the SWA mask, which is exact and does not need a learned bias. Long-range
   selection is handled by CSA2's indexer and by the decoder's global store — and those are
   the reads with no positional signal at all. The one place position must be informative
   is the only place CVPE is uninformative.
5. *Non-stationarity.* Because the argument is `r/q` rather than `r`, the same distance
   means different biases at different absolute positions. This is a deliberate
   context-relative parameterization, but it means the bias function is not
   translation-invariant, so there is no principled length extrapolation and the decoder's
   queries (positioned after the encoder stream) live in a `q`-range the encoder never saw.

**Minimal fix.** Two ablations and one addition. Ablations: CVPE vs no positional signal vs
ALiBi-style `−c·r` per head. Addition, if you want to keep the "content-versus-position"
name honest: make the bias a *bilinear* form — a low-rank `γ(q)ᵀ M f(k)` or a per-head
rotation applied to q and k — so that content and position interact multiplicatively rather
than additively, and so the bias can express distance without a saturating scalar.

---

## S5 — FRVM removes the linear structure of the OV circuit and cannot copy

**Symptom.** `o_q = Σ_k w_qk [ v_k ⊙ SiLU( v_k + φ_h(v_q − v_k) ) ]`, `φ` zero-init, so at
step 0 the reduction is `Σ w_qk (v_k ⊙ SiLU(v_k))`.

**Why it is a science problem.**

1. *The "vanilla at init" claim is false in the only way that matters.* With `φ = 0` the
   pairwise term vanishes but the base term does not: `v ⊙ SiLU(v)` rectifies and
   approximately squares positive values, suppresses negative ones. A copying head needs an
   OV circuit close to identity; `v ⊙ SiLU(v)` cannot copy a negative channel and distorts
   large positive ones. Induction and copy behaviour are exactly what small LMs rely on
   early in training.
2. *The reduction is no longer separable.* Because `φ(v_q − v_k)` depends on the query, the
   value transform cannot be precomputed per key. `attention = softmax(QKᵀ)V` becomes
   `softmax(QKᵀ) · G(Q, K, V)` where `G` is a per-pair nonlinearity: same asymptotics,
   but no matmul structure, no value-side cache reuse, and `O(chunk · L · d)` intermediate
   memory for the chunked path you already have to use. This is the property that makes
   attention cheap and cacheable, and it is being traded for a second-order interaction
   that has not been shown to help.
3. *Asymmetry by accident.* The gate reads `v_q − v_k`, injecting a query-key distinction
   into the value path. That is not "value gating"; it is a per-pair feature, i.e. a
   low-rank bilinear attention. Interesting, but it should be named as such, and compared
   against the cheaper alternatives (gated linear attention, second-order pooling).

**Minimal fix.** Wrap the whole OV in a residual: `o = Σ w v + λ · Σ w [v ⊙ SiLU(v + φ(Δv))]`
with `λ` zero-init. Then the block *is* vanilla attention at step 0, copying is free, and
the added term is a measurable perturbation. This one change converts FRVM from a
confound into an experiment.

---

## S6 — Support is computed by the shallowest layer, then frozen; the decoder re-reads one vector set twenty times

**Symptom.** Encoder: `full` at layer 1, `reuse` for layers 2–19. Decoder: store seeded once
from `global_projs[0](encoder_final)`; the other `dec_n − 1` projections are instantiated and
never called.

**Why it is a science problem.** Two separate claims are being made implicitly and neither
is stated. (a) *Support stability*: layer 1's top-k is a good support for layer 19. Layer 1
has the least contextual information in the stack, so this is the hardest possible version
of the claim — and if it fails, it fails for a reason that a mid-stack `full` would fix.
(b) *Store sufficiency*: a single summary, produced before any decoder computation, is
sufficient for twenty layers of decoding. The decoder cannot compress its own reasoning;
all decoder-internal long-range structure must live in the residual stream.

**Minimal fix.** Move the encoder's `full` to the middle of the stack (or allow two: one at
layer 1, one at layer `n//2`) and ablate. Separately, delete or wire the unused
`global_projs[1:]`; dead parameters that appear live are a review hazard.

---

## S7 — The attention "sink" has no value vector and is therefore a scalar output gate

**Symptom.** `a = e^z / (Σ e^z + e^{s})`. Row sums are `< 1`; the sink mass is discarded.

**Why it is a science problem.** This is exactly `(1 − p_sink) · Σ w v`. It is a learned
per-query decay on the attention output — a gate, not a sink. The published sink
formulations add a virtual token *with a learned value* precisely so the convex combination
stays intact and the model can represent "attend to a fixed learned vector." Yours cannot.
The mechanism may still be useful (a head that wants to output nothing can shrink its
output), but the name asserts a property the math does not have, and the missing value
vector removes the thing sinks are for.

**Minimal fix.** Add the value vector (`s_value`, `head_dim`), include its contribution
`p_sink · s_value` in the output, and re-run whatever motivated the sink in the first place.

---

## S8 — The compressed store is auto-associative, and this is unstated

**Symptom.** `BlockCompressor` emits one vector per block; CSA2 uses it as both key and
value.

**Why it is a science problem.** The store can only represent "retrieve X by X." For the
encoder's self-attention that is a defensible restriction (it is a soft nearest-neighbour
memory). For the decoder, which queries the store with its own representations, it means
decoder queries must live in the encoder's key subspace; any decoder-specific "I need the
thing that answers this question, not the thing that looks like this question" is
unrepresentable. This is precisely the regime where a compressed memory is supposed to earn
its keep.

**Minimal fix.** Add a separate value projection in `BlockCompressor` (or a low-rank one,
`D_h → r → D_h`), doubling nothing but one matmul, and ablate. It is the cheapest possible
increase in memory expressivity.

---

## S9 — BLT decodes each byte from exactly one patch vector, one patch behind

**Symptom (verified by intervention).** In `VocabularyDecoder`, a byte in patch `p` attends
to exactly one key: `patch_reps[p−1]`. The masked path gives each slot its own latent. So
per-byte global conditioning is a single vector, and in the full path that vector belongs to
the *previous* patch.

**Why it is a science problem.** A byte cannot see the latent of the patch it belongs to,
only the one before it. Global context therefore arrives with a latency of one patch
(`blt_patch_size` bytes) and with the bandwidth of one vector. Everything else must come
from the 512-byte local causal transformer. This is a much harder bottleneck than the BLT
paper's cross-attention over all patch latents, and it is a design choice, not a bug — but
it means "BLT" here is a different model with a different claim: *patch-summary-conditioned
byte decoding*, not *latent-conditioned byte decoding*. Claimed as the latter, it will not
reproduce published BLT results.

**Minimal fix.** Allow the byte to see `patch_reps[p−1:p+1]` (two keys: previous and own) and
measure. That single change makes the comparison to BLT meaningful.

---

## S10 — `AdaptiveExitGate` has a correct distribution and no training signal

**Symptom.** `p(t) = λ_t Π_{j<t}(1 − λ_j)`, residual mass on the final step, exit at the
first step whose CDF exceeds `q`. Selection is a hard `torch.where`, so only the chosen
step receives gradient. The repository's only auxiliary loss is the BLT latent KL.

**Why it is a science problem.** A hazard model with no compute penalty has no incentive to
exit early: exiting later is weakly better for the loss and costs nothing in the objective,
so the learned `λ` encodes "which step was used," not "when it is safe to stop." At
inference, `q` therefore selects from a distribution that was never trained to be
calibrated. The hard selection also makes the gradient a biased estimator of the expected
loss over exit times.

**Minimal fix.** Add `τ · E[t]` with `E[t] = Σ_t t · p(t)`, which you already compute, and
compare against the hard-selection-only baseline. That is the ACT/ponder-cost term, it is
three lines, and it turns early exit from a hope into a trade-off curve.

---

## S11 — Titans on alternating layers is a cost landmine, on by default

Second-order differentiation through the inner loop, on odd encoder layers, with
`ced_encoder_alternating=True` by default. There is no measurement of (a) whether the
meta-learned branch beats GDN-2 at matched FLOPs, or (b) what it costs. Heterogeneous
memory *types* per layer is a strong architectural claim (that different depths want
different associative rules); it needs a controlled comparison against homogeneous GDN-2
and homogeneous Titans. Until then the default is an unmeasured 2–4× training cost.

---

## S12 — `DenseFFN` interval gating is static depth pruning, not a schedule

`DenseFFN.forward` applies the FFN only when `step % update_interval == 0`, and CED passes
`step = layer index`. With `update_interval > 1` a layer's FFN is simply never applied — for
every forward pass, forever. The docstring describes a per-forward-call counter; the call
sites pass depth. Likewise CMS's `should_update(step)` is never used in `forward`, which
applies all levels unconditionally. So "continuum" / "time cycle" is, in the wired path,
depth-indexed gating.

This matters because it makes the CMS idea untestable: the mechanism that would justify the
name (a *time-varying* update schedule) does not exist in any executed path.

---

## S13 — Naming drift is a scientific problem, not a cosmetic one

- "Sink" without a value vector (S7).
- "Content-versus-position" that is a scalar prior, not a positional representation (S4).
- "Membership" mask whose `True` means *excluded* — cost one verification cycle already.
- "Hierarchical" sparse selection whose prefilter is inert below 65k tokens
  (`pool_blocks × block_size = 16384` candidates; the store is `L/m ≈ L/4`).
- "Continuum" memory with no schedule in the executed path (S12).
- "KL" that is not a bound (`GaussianLatentSampler`: a variance regularizer toward a fixed
  target; the docstring is admirably honest, but the symbol `D_KL` will invite the wrong
  question).
- "Cheap" in a codebase whose default runs second-order Titans on half the encoder.

Each of these is a *claim embedded in a name that the mathematics does not support*. In a
research artifact that is the most expensive kind of debt, because the name is what everyone
downstream reads.

---

## S14 — Roughly four thousand lines are unwired, and some duplicate the thesis

Not wired into any registered backbone: `SinglePassMHC`, `EngramMemory`, `DSparkDrafter`,
`LivConvBlock`, `LivPerceiver`, `HopeBlock`, `HopeAttentionBlock`, `ContinuumMemorySystem`,
`TitansFastWeightsMemory`, `NativeSparseAttention`, `JacobiRKAN`/`PadeRKAN`,
`ChunkedBottleneckLinear`. (`FactoredLinear` left this list: every registered backbone now
builds its projections through it, see `code/03` S2-13.)

Science-level assessment of each:

- **mHC** — the single-pass form `X' = B X + C F(A_{l−1} X)` (input mixing uses the
  *previous* block's `A`) is a Jacobi-style one-step-delayed coupling, which is genuinely
  interesting. But a Sinkhorn-constrained multi-stream residual is very close to
  Hyper-Connections with a different constraint; the novel part is the *shared-`B`
  two-branch* variant. Either wire that variant as an ablation against HC, or cut it.
- **Engram** — hashed n-gram lookup tables are classical, and BLT already has hashed n-gram
  embeddings in the byte encoder. Two hashed n-gram memories in one model is redundancy,
  and n-gram tables are known to saturate.
- **LIV** — gated, head-factorized linear mixing without decay. Little new; the perceiver
  variant (iterative latent bottleneck with sparse routing through CSA2) is the only part
  worth keeping, and it overlaps with the masked Perceiver-AR path you already have.
- **rKAN** — orthogonal to the architecture's thesis. Park or remove.
- **DSparkDrafter** — an inference optimization, not a science claim.
- **NSA** — a reimplementation of a mechanism your own CSA2 supersedes. Keeping both invites
  "which one is the contribution?"
- **linears** — `ChunkedBottleneckLinear` (scalar pool per chunk → `G×G` bottleneck →
  broadcast, `O(D_in + D_out + R²)`) is the one genuinely interesting item: cross-chunk
  interaction only through a low-rank bottleneck is a real efficiency/expressivity
  trade-off, and it has no test. `FactoredLinear` is now wired (per-role rank budget), so
  the plain rank constraint is no longer surplus.

Surplus code is not neutral: it dilutes the thesis, and a reviewer reads it as "the author
has not decided what the contribution is."

---

## S15 — Baseline correctness: `ExampleBackbone` ignores `attention_mask`

Padding is not masked, so the baseline is wrong on any padded batch. Any comparison against
it is unfair to the baseline, which makes the comparison worthless in the direction that
matters (showing the CED wins). Fix before the first benchmark table exists.

---

## S16 — Two mask conventions in one codebase

The module path documents bool `True = masked out`; `ops/references/attention_reference.py`
uses SDPA semantics where `True = participate`. Both are individually defensible; together
they are a footgun that has already produced one false alarm in this review. Pick SDPA
semantics at the op boundary (it is the convention every kernel will use) and convert once,
at the module entry.

---

## S17 — There is no measurement apparatus at all

No baseline harness, no ablation group, no scaling comparison, no probe suite, no recorded
perplexity anywhere in the repository. Every claim in AGENTS.md and every docstring is a
hypothesis. This is the single largest gap between the code and a paper, and it is
independent of every item above.

The next file states exactly which measurements decide which claims, in what order.
