# 02 — The great points, ranked by scientific value

Ranking criterion: how much *new, checkable knowledge* the idea could produce if it were
tested. Not elegance, not effort.

---

## 1. Compression and selection as separate operators (CSA2)

**The idea.** `BlockCompressor` learns a softmax-weighted m:1 pool; `SparseIndexer` is a
separate module with its own low-rank query, its own index keys, and its own scoring
function; the top-k support it produces is a first-class object that other layers consume.

**Why it is good.** Every compressed-attention scheme I know fuses these. Native Sparse
Attention pools and selects with one parameter set and one representation. HCA compresses
by local pooling and attends densely to the result. MoBA routes blocks but never compresses
them. Gist/memory-token methods compress without selecting. Fusing them creates a real
optimization conflict: the same weights must produce a summary that is *representative*
(a low-variance average) and a score that is *discriminative* (a high-variance argmax). Your
split removes that conflict by construction, and it is the kind of claim that survives
contact with an ablation: freeze the indexer and train the compressor, and vice versa.

**What it costs to test.** One ablation at ~100M parameters: shared-projection CSA2 vs
split CSA2, same FLOPs. If split wins, that is a clean, publishable mechanism result.

---

## 2. Cross-depth support inheritance (`reuse` mode)

**The idea.** The top-k support is computed once (encoder layer 1) and inherited by
eighteen layers; the decoder recomputes it only every fourth layer, and even then only by
rescoring a cached candidate pool rather than rescanning the store.

**Why it is good.** This is a genuinely new axis. The sparse-attention literature treats
the support as a per-layer, per-call quantity, and pays for it every time. You have turned
it into a *budget allocated across depth*, with three levels of fidelity (`full`,
`reindex`, `reuse`) and a cheap intermediate (`reindex` only rescores ≤16k cached
candidates). If support stability holds, this is a large constant-factor win at equal
quality, and it implies something non-obvious about transformer internals: that layers
mostly agree about *where* the relevant context is, even when they disagree about *what to
do with it*.

**Why it is the best experiment in the repository.** It is falsifiable in a day, it has a
trivial control (`reindex` everywhere), and it is the repository's main cost claim. If it
fails, CSA2 is still viable but no longer cheap; if it holds, it is the headline.

**The honest caveat.** As configured, the support is computed by the *shallowest*
contextual layer (encoder layer 1), which is the layer with the least information. Support
inheritance would be far more defensible if the support were computed once at the *middle*
of the encoder (say layer 10) and inherited outward in both directions, or if the first
`full` were at the layer where representations stabilize.

---

## 3. Parallel hybrid with an asymmetry that makes the new branch optional at init

**The idea.** `h = norm(x)`; `Attn_swa(h) ∥ Mem_global(h)`; concatenate; project 2d→d; one
residual. The attention half of the output projection is initialised at `init_range`, the
memory half at `1e-3`.

**Why it is good.** Two separate virtues.

*Scientific:* parallel rather than sequential composition forces the model to treat the two
mixers as functionals of one representation, not as stages. That is a different inductive
bias from Jamba/Griffin, and it is the right one if the goal is "local syntax and global
retrieval are different views of the same token," which is what the encoder is for.

*Methodological:* the asymmetric init is a **warm start with an off-switch**. At step 0 the
block is a standard attention block, so the memory branch is a perturbation that can be
zeroed at test time and measured. Very few hybrid papers provide that affordance; most
force you to retrain from scratch to ablate the mixer. Keep this property deliberately, and
protect it: any change that makes the memory branch non-zero at init destroys the cleanest
ablation you have.

---

## 4. The bottleneck is one inspectable tensor

**The idea.** Eq. 1 is literally `global_projs[0](encoder_final)` — one projection, one
tensor, shared by every decoder layer.

**Why it is good.** Compression-based architectures usually hide their bottleneck inside a
stack, which makes "what did the summary lose?" unanswerable. Here the bottleneck is a
named tensor of shape `(B, L/m, H_kv, D_h)`. You can probe it linearly, measure its mutual
information with the prefix, replace it with an oracle summary to bound the loss, and ablate
the projection. That is an unusually good *experimental affordance* for an architectural
idea, and it is free — it falls out of the design.

---

## 5. GDN-2 as a family, not a point

**The idea.** `S_t = (I − k_t (b_t ⊙ k_t)^T) Diag(exp g_t) S_{t−1} + k_t (w_t ⊙ v_t)^T`,
with channel-wise erase `b`, write `w`, decay `g`. Setting `b = w = β·1` recovers KDA;
also tying decay recovers Gated DeltaNet. Resting state is an exact centered
reparameterization; `apply_beta_guard` derives a certified non-expansion condition from the
key-weighted variance of the erase gate.

**Why it is good.** This is the only place in the repository where a mechanism is presented
as a *family with proven special cases* rather than a single architecture. That is how
sequence-mixer theory should be written: name the general object, show the published models
as slices, then add a constraint with a proof. The beta guard is a real stability
certificate (non-expansion of the key-weighted variance), and the resting-state algebra is
verifiably exact because it is pure linearity.

**Honest limit.** These are *notes*, not yet contributions: both are opt-in, off by default,
and address failure modes that may not occur at your scale. Their value is that they make
the family *analyzable*, which matters if you ever want a theory section.

---

## 6. A genuinely second-order inner loop (Titans), gated on autograd

**The idea.** The fast-weight memory's inner update is differentiated through
(`create_graph = torch.is_grad_enabled()`), so training meta-learns the update rule and
evaluation detaches it. Chunked evaluation uses chunk-start weights plus accumulated deltas
(TNT-style associativity).

**Why it is good.** Most "learned memory" implementations silently detach and are therefore
just another linear attention with extra steps. Differentiating the inner loop is the actual
claim of the Titans/HOPE line, and you implement it. Gating on `is_grad_enabled()` rather
than a config flag is also the correct semantics: the inner loop *is* part of the function
during training and *is* a fixed recurrence at inference.

**Honest limit.** This is also the repository's largest hidden cost: second-order through a
20-layer stack with Titans on alternating layers, on by default.

---

## 7. Zero-initialisation as a design principle

Nearly every new mechanism starts as an exact no-op: `block_bias` and `score_proj` zero
(uniform pooling), indexer `weight_proj` zero (no routing preference), CVPE `gamma/beta`
zero (vanilla attention), FRVM `phi` zero (no pairwise term), sink logit zero (no sink
mass), `value_proj` zero in Engram, latent `mu` and `log_sigma` zero (σ = target),
depth-scaled or zero output projections.

**Why it is good.** It makes each mechanism a *perturbation of a known-good model*, which
is exactly the condition under which ablations are interpretable. This is a real
methodological asset and it is rare to see it applied this consistently.

**The catch** (see `03-weaknesses.md` §1): the principle is applied *jointly* to nine
mechanisms and then combined with `initializer_range=0.006`, which produces a model whose
logits are ~1e-10 at small width. Zero-init plus tiny init is not "start vanilla"; it is
"start dead."

---

## 8. Oracle reference implementations and a dispatcher that refuses to lie

`ops/references/` contains pure-PyTorch ground truth for the QK circuit, the FRVM reduction,
and a full SDPA oracle; `ops/dispatch.py` downgrades from Taichi to C++ to reference *with a
warning*, but raises if you explicitly ask for a compiled backend that is unavailable.

**Why it is good.** This is the best scientific hygiene in the repository, and it is the
thing that makes kernel work citable: a fused kernel is only a contribution if it is
bit-comparable to a readable oracle on adversarial shapes. Refusing silent downgrade is the
difference between a benchmark and a marketing claim.

---

## 9. Patch latents as learned readouts of their own bytes (BLT)

`patch_init_proj(max-pool)` only *initialises* the patch representation; the actual latent
is a cross-attention readout restricted to the patch's own bytes, so each patch summarises
itself with a learned query rather than a fixed pooling function. Membership masking is
exact, and the causality probe confirms no leakage.

**Why it is good.** Max-pool init plus learned readout is strictly more expressive than
mean/max pooling and costs one cross-attention pass. It also makes the local encoder's
output a *queryable* object, which is what the latent stack needs.

---

## 10. Structural causality actually holds

Probe (command in `05-evidence-and-experiments.md`): perturb the last token of a 16-token
sequence in the non-masked CED path; per-position `max|Δlogit|` is zero for positions 0–14
and non-zero only at 15, in both the plain and the looped decoder. So `reuse`/`reindex`
caching, the seeded store, and the Perceiver-AR position arithmetic are all causally sound.
A caching scheme that leaks future information is a very common failure; you do not have it.

---

## 11. Mask-count interpolation in `MaskedTokenEmbedding`

`L < C` uses the last `L` rows in order; `L = C` is exact; `L > C` linearly interpolates
between neighbours. One table serves any mask count, with the endpoints anchored.

**Why it is good.** It turns masked-token count into a *continuous* hyperparameter with no
retraining and no extra parameters, and it degrades gracefully in both directions. Small,
but genuinely clever, and the kind of thing that shows up in a systems paper as "why this
scales to variable compute."

---

## What the strengths have in common

Seven of the eleven are *affordances*: they make a claim measurable (off-switch init,
inspectable bottleneck, oracle ops, causality probe) rather than asserting it. That is the
right instinct for a research repository, and it is why the weaknesses in the next file are
all fixable with experiments rather than rewrites.
