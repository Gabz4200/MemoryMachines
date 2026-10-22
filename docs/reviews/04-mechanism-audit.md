# 04 — Mechanism audit, algebra first

Each mechanism is given as: the operator as written, what it is algebraically, and the
consequence that follows. No code review; only the mathematics and what it implies.

---

## A. The CED stack: `replace` mixers inside a residual network

### The operator

Encoder and decoder layers perform, per layer,

```
x ← M_i(x)                 # mixer: attention / CSA2 / parallel hybrid
x ← x + F_i(norm(x))       # DenseFFN, pre-norm residual
```

The mixer is a **replacement**, not a residual increment. Only the FFN is residual.

### Why this is the most consequential line in the repository

In a residual network, `out_proj` initialised at zero or at `init_range/√(2N)` is *safe*,
because the identity path carries the signal and the branch is a perturbation. Under
`x ← M(x)` the same initialisation is a **multiplier near zero**. The two design choices —
zero-ish output init and no residual — are individually defensible and jointly destructive.

Measured at the project's default `initializer_range=0.006`, tiny config, eval mode:

| quantity | value | ratio |
| --- | --- | --- |
| `‖embed(x)‖` | 2.29e+01 | — |
| `‖M_0(x)‖` (encoder layer 0) | 4.59e−03 | **2.0e−04** |
| `‖F_0(norm(x)) − x‖` (FFN delta) | 5.46e−12 | **1.2e−09** |
| final logit std | ~5e−10 | — |

So the stack begins by contracting the residual stream by four orders of magnitude, and the
FFN — the only residual path — contributes a factor of `1e−9` at initialisation. Every
subsequent layer is a near-identity map on a signal that is already `1e−4` of the
embedding. This is the algebraic cause of the dead-init finding in `03-weaknesses.md` §S3;
it is not a hyperparameter tuning issue and it is not fixed by choosing a different
`initializer_range` (a larger range just converts the contraction into the `1e17`
expansion observed at `init_range=1.0`).

### Why the diagnostic is *attenuation*, not just scale

The contraction is multiplicative at the first mixer and additive afterwards, so the
network's effective depth at initialisation is one. Gradient flow to the embedding is
correspondingly tiny. A model can recover, but it must first learn to undo a `1e−4`
attenuation using gradients that are themselves `1e−4`-scaled — the classic
vanishing-activation trap that pre-norm residuals exist to prevent.

**The fix is structural, not numerical.** Either (a) make the mixer residual
(`x ← x + M(x)`), in which case all the zero-inits become the correct "off-switch"
they were designed to be, or (b) keep replacement mixers and initialise `out_proj` at
`init_range` (no depth scaling), accepting that each layer starts as a random projection of
the same magnitude as its input. Option (a) is the one that preserves the architecture's
own ablation philosophy.

### Eq. 1 as cross-attention, stated precisely

The decoder computes

```
Q = W_q · decoder_state            (width d_dec)
K = V = C_enc                      (encoder-compressed entries, width D_h, H_kv heads)
A = softmax(QKᵀ/√D_h + b_CVPE − ∞·[comp_end > pos])
```

with `V = K` (auto-associative, see §B) and `pos` starting at `N` (after the encoder
stream). Formally this is *cross-attention to a fixed memory*, i.e. a single-step
memory-augmented decoder. The information-theoretic content is `I(prefix ; C_enc)`, capped
by `L/m · H_kv · D_h` floats regardless of `L`. Whether the cap binds is measurable: replace
`C_enc` with the full encoder stream at the same compute and compare.

---

## B. CSA2: pooling, selection, and what `reuse` actually assumes

### Pooling

For block `j` covering tokens `mj … mj+m−1`:

```
s_jt = softmax_t( score_proj(h_t) + block_bias_j )       # over t in block
C_j  = Σ_t s_jt · kv_proj(h_t)                            # one vector, used as K and V
```

`score_proj` and `block_bias` are zero, so at init `s = 1/m` — uniform mean pooling, which
is the correct neutral element. The store is `L/m × H_kv × D_h`.

**Algebraic consequence: auto-associativity.** With `K = V = C`, the memory computes
`softmax(q·Cᵀ)·C`, a nearest-neighbour reconstruction. It cannot represent retrieval of a
value different from the key. For a decoder that queries with its own states, this forces
`q_dec ∈ span(C)` to retrieve anything useful — a rank constraint that the `decoder_in_proj`
does not remove.

### Selection

```
z_h = ReLU( q_idx · k_idx )              # H_idx index heads, per-entry
score = Σ_h w_h z_h                      # w zero-init
top = argtop_k(score over {j : comp_end_j ≤ pos})
```

Three algebraic properties worth stating:

1. **`ReLU` makes the score a non-negative combination.** No head can veto an entry; the
   indexer can only add evidence. Routing is monotone-aggregative, which is a strictly
   weaker hypothesis class than signed scoring. A signed score (or a `tanh`) would allow
   "this entry looks relevant to head 3 but is poison for head 7."
2. **Zero-init weights give tied scores.** All `score = 0` at init, so `argtop_k` falls back
   to the tie-breaking order — in practice the lowest indices. The model therefore starts
   with a *positional* prior (attend to the earliest blocks), not a neutral one. Harmless
   for training, but it means "the indexer learned to select" must be established, not
   assumed: at step 0 it is an ALiBi-like prefix preference.
3. **The hierarchical prefilter is inert at realistic lengths.** `pool_blocks × block_size =
   2048 × 8 = 16384` candidate entries; the store is `L/m ≈ L/4`. The prefilter only engages
   above `L ≈ 65k`. Below that, the "hierarchical" path is a no-op, so the mechanism is
   untested by construction at any length you can currently train.

### What `reuse` assumes, and the error it can make

`reuse` inherits `top` and computes `softmax(q·C[top])`. The approximation error relative to
`full` is

```
‖Δo‖ ≤ ‖V[top]‖ · ‖Δw‖ + ‖w[missing]‖ · ‖V[missing]‖
```

The second term is the one that matters: mass that a *fresh* score would have assigned to
entries outside the inherited support is simply unreachable. Support inheritance is
therefore exact iff the top-k set is stable under the change of query distribution between
layers. Layer 1 (where the support is fixed) has the least contextual information in the
stack, which is the worst possible place to make that bet — see `03-weaknesses.md` §S6.

`reindex` is the interesting middle: it re-scores only the cached pool. That is a
*recall-preserving* approximation (the candidate set is fixed, the ranking is fresh), and it
is the right compromise if support drift is real but small. Its cost is `O(L/m · H_idx · D_i)`
— no compression, no store scan.

### Sink

```
a = e^{z} / (Σ e^{z} + e^{s})  ⇒  o = (1 − p_s) · Σ w v
```

A per-query scalar decay on the OV output. Algebraically a gate, not a sink: there is no
value associated with the sink mass, so the map is not a convex combination and the output
norm is not preserved. See `03-weaknesses.md` §S7.

### Grouped output

Heads are partitioned into `G` groups, each group projected to `group_dim`, concatenated,
then projected to `hidden`. This is a **low-rank factorisation across heads**: the mixer
cannot emit an arbitrary `d`-dimensional update per head, only a combination drawn from a
`G · group_dim`-dimensional space. With `G = 8`, `group_dim = 1024`, `hidden = 1024`, the
factorisation is rank-`8192 → 1024`, which is not restrictive; at the default it is
effectively a full projection with a group-sparse parameterisation. It only becomes a real
constraint if `group_dim · G < hidden`.

---

## C. GDN-2: the one place with real algebra

### Recurrence

```
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) Diag(exp g_t) S_{t−1} + k_t (w_t ⊙ v_t)ᵀ ,   o_t = S_tᵀ q_t
```

This is an affine recurrence in `S`. The transition is a rank-one Householder-like erase
followed by a channel-wise decay; the input is rank-one. Special cases:

- `b = w = β·1` → KDA (gated delta rule with scalar per-token write strength).
- additionally tied decay → Gated DeltaNet.
- `b = 0` → pure decayed outer-product memory (linear attention with forget gate).

So the family is correctly stated: GDN-2 contains the published mixers as sections, and the
free parameters are *per-channel* erase and write rather than per-token scalars. That is a
genuine generalisation — the erase direction `b ⊙ k` is a *different direction* from the
write direction `k`, which is what allows selective overwriting of a subspace.

### Resting state is exact, and it is a reparameterisation, not new dynamics

Write the recurrence as `S_t = A_t S_{t−1} + Z_t`. Substituting `S = S₀ + D`:

```
D_t = A_t D_{t−1} + (Z_t − (I − A_t) S₀)
```

The implementation subtracts exactly `(I − A_t)S₀` from the write target and adds `S₀ᵀq` at
the readout, so `D` follows the canonical recurrence and `S_t = S₀ + D_t` identically. Two
consequences: (a) the identity is exact, so it cannot change expressivity — it only changes
*conditioning*, making `S₀` the fixed point of the input-free flow; (b) it is therefore a
regulariser/initialisation statement, not an architectural one, and should be described that
way.

### The beta guard proves a one-step property, not the operator's stability

The guard computes `r² = n·ν − a²` with `a = Σ k_i b_i`, `ν = Σ k_i b_i²`, i.e. the
key-weighted variance of the erase gate, and shrinks `b` toward its key-weighted mean when
`r²` exceeds a threshold derived from `exp(−2 g_max)`. On the rank-one invariant subspace
`S = k vᵀ`, the update multiplies the `k`-component by `1 − Σ_i b_i k_i²`, so non-expansion
requires that key-weighted mean to stay below 2. The guard enforces a *sufficient* condition
for that, and it is a legitimate certificate **for one step along the key direction**.

It does not certify the composed chunk operator `Π_t (I − k_t b_tᵀ) Diag(exp g_t)`, which is
a product of non-commuting rank-one updates. That product is what actually runs over a
chunk, and product stability is not implied by one-step non-expansion. If the guard is to be
presented as a stability guarantee, the missing step is a bound on the spectral radius of
the chunk product — or an empirical check that the guard's presence changes divergence
frequency.

---

## D. Titans / HOPE: a two-timescale leaky integrator with a learned update rule

`S_t = η S_{t−1} − θ ∇ℓ(k_t, v_t; W_t)`, `M_t = (1−α)M_{t−1} + S_t`, `W_{t+1} = W_t + M_t`.

Algebraically this is **momentum**: `W` is updated by a leaky integral of gradients, where
the gradient is taken through a deep MLP with an L2 associative loss `ℓ = ½‖f(k;W) − v‖²`.
With `α → 0` it is plain gradient descent on the associative loss; with `α → 1` it is
SGD-with-momentum. Making `η, θ, α` functions of the input (`σ(mlp(x))`) makes the *step
size schedule itself* content-dependent, which is the actual novelty: not "a memory," but
"a per-token optimizer."

Two honest limitations.

1. **`η → 0` recovering GDN** is true only in the linear (`depth = 1`) case, where the
   gradient is `k(kᵀW − vᵀ)` and the update is exactly the delta rule. At `depth > 1` the
   gradient is nonlinear in `W` and the correspondence is only qualitative. The code
   implements the depth-1 closed form separately, which is the right call, but the claim
   should be stated with the same restriction.
2. **Chunked evaluation is an approximation of a different object.** Evaluating gradients
   relative to chunk-start weights and accumulating deltas is the standard linearisation
   used for chunked delta rules; it is exact for the linear case and approximate otherwise.
   Since `create_graph=True` in training, the *training* path is exact-per-token and the
   *eval* path is chunked, so train/eval differ in the update rule, not just in
   differentiation. That is a real train/test mismatch to state explicitly.

---

## E. Position and the OV circuit

### CVPE

`b_qk = γ(q) f(r/q) + β(q)`, `f` bounded, per-head scalar. As an operator on the attention
matrix this is `B = diag(γ(q)) F diag(1) + 1 β(q)ᵀ`, i.e. a rank-structured additive prior:
a per-query amplitude times a Toeplitz-like distance function, plus a per-query row
constant. Two structural facts:

- It is **not a group action**: the bias is not invariant to translation in `(q,k)` because
  its argument is `r/q`. Consequently there is no compositional law across layers, which is
  precisely what makes RoPE generalise.
- It is **not in the residual stream**, so it cannot be transformed. Any computation that
  needs "where am I" in the value space must reconstruct it from attention weights.

### FRVM

`o_q = Σ_k w_qk [ v_k ⊙ σ(v_k + φ(v_q − v_k)) ]`. Expanding to first order in `φ`:

```
o_q ≈ Σ_k w_qk [ v_k ⊙ σ(v_k) ] + Σ_k w_qk (v_k ⊙ σ'(v_k)) ⊙ φ(v_q − v_k)
```

The zeroth-order term is a fixed nonlinearity on values, not the identity: the OV circuit at
init is `diag(σ(v)+vσ'(v))`, which is near-zero for negative `v` and ≈`2v` for large
positive `v`. Copying is impossible at init, and copying is the OV circuit that induction
heads depend on. The first-order term is a genuinely new bilinear interaction — but it is a
*value-space* interaction, so it does not commute with the `QKᵀV` factorisation, which is why
the reference implementation must chunk over queries.

The two mechanisms interact badly: CVPE modulates `w_qk` while FRVM gates `v_k` by a function
of `v_q`, so the two "content" paths multiply. Any ablation that removes one changes the
gradient scale of the other. This is the concrete reason S1 (confounded defaults) is not a
bookkeeping complaint.

---

## F. DyT as the only normalisation

`DyT(x) = tanh(α ⊙ x + β) ⊙ γ`, elementwise, no reduction over the feature axis.

A norm does two things: it bounds the input scale and it makes the layer's output
*invariant* to the input's scale. DyT does the first (bounded by `|γ|`) and not the second
(no division by a statistic). Consequences:

- The residual stream has **no scale control**. Residual magnitude is a free function of
  depth and initialisation — which is exactly what the measurements in §A and
  `03-weaknesses.md` §S3 show, in both directions.
- `tanh` saturation is input-scale dependent, so as the residual grows the layer becomes
  progressively more linear-then-saturated, changing the effective nonlinearity with depth.
- Attention logit scale is controlled only by the q/k projection initialisation. With
  `initializer_range` applied globally, logit temperature is an emergent function of width
  and depth rather than a controlled constant.

DyT is a defensible choice on cost grounds; it is not a drop-in replacement for a norm in a
network whose residual stream is unnormalised *and* whose mixers are replacements. The
combination is the problem, not DyT alone.

---

## G. BLT algebra

Patch latent: `P_j = CrossAttn(Q = init_j, K = V = bytes of patch j)`, with
`init_j = patch_init_proj(max-pool)`. So `P_j ∈ span(h_B)` for the patch's own bytes only —
a learned readout, correctly restricted.

Byte decoder: `h_i ← h_i + CrossAttn(q = h_i, K = V = {P_{p(i)−1}})` — **one key**. Then a
causal byte-level self-attention with window `blt_local_window_size`, then an FFN.

As an operator the byte LM is

```
p(b_i | b_<i) = softmax( W_o · [ h_i^{(local)}(b_<i) + Attn(h_i, P_{p(i)−1}) ] )
```

so the global path contributes a *single* vector per byte, delayed by one patch. The
information rate through the global path is `D_h` floats per patch, i.e. `D_h /
blt_patch_size` per byte, versus `d` per byte through the (exact) local window. The intended
reading — "bytes are conditioned on the latent that summarises their context" — requires the
current patch's latent; the implemented reading is "conditioned on the previous patch's
summary."

`GaussianLatentSampler`: `z = μ(p) + ε·T·σ`, `μ` zero-init, `log σ` zero-init so `σ = σ_target`,
and `L_reg = log(σ_target/σ) + σ²/(2σ_target²) − ½`, which is `D_KL(N(μ,σ²) ‖ N(μ,σ_target²))`
and is **zero at init** and increase-only away from the target in both directions. It is a
variance anchor, not a variational bound: there is no `E_q[log p(x|z)]` term, so
"ELBO" language would be wrong (the docstring is careful about this) and the symbol `D_KL`
invites the wrong question. As a regulariser it is well-posed: CE alone is convex in the
logits and would drive `σ → 0`; the anchor prevents collapse.

---

## H. Unwired mechanisms — algebra verdicts

- **mHC (single-pass)**: `X_{l+1} = B_l X_l + C_l F_l(A_{l−1} X_l)`. Using the *previous*
  block's `A` makes this a Jacobi sweep on a coupled fixed point; the coupling is one step
  of a Gauss–Jacobi iteration, so the composition is not the fixed point but a truncation of
  it. Interesting, and almost exactly Hyper-Connections with a Sinkhorn constraint on `B`.
  The novel piece is the shared-`B` two-branch variant (same residual geometry, different
  branch maps), which is a real hypothesis: residual mixing should be branch-agnostic while
  input/output mixing is branch-specific.
- **Engram**: multiplicative-XOR hashing of n-grams into prime-sized tables inside one flat
  embedding. This is feature hashing with collision probability `≈ 1/table_size` per order;
  the algebraic content is a sparse, fixed, non-learned key map — a random feature map for
  n-grams. Zero-init value projection makes it correct at step 0, but the capacity/scaling
  story is that of count tables: it saturates, and it duplicates BLT's n-gram embeddings.
- **LIV**: `y = post_gate ⊙ mix(pre_gate ⊙ x_proj(x))` per head. With elementwise gates and
  no decay, this is gated linear mixing; the head-factorised form means no cross-head
  interaction, so it is `H` independent scalar-gated channel mixes. Low expressivity per
  parameter; not obviously better than a depthwise conv + linear.
- **rKAN**: rational activations with Padé denominators and constrained parameters. The
  approximation-theory claim (rational > polynomial per parameter) is plausible; it is
  orthogonal to the architecture's thesis and should be a separate paper or nothing.
- **linears**: `FactoredLinear` is a rank constraint `W = W_out W_in` (no expressivity gain
  past `min(d_in,d_out)`, correctly warned). It is no longer a wiring candidate: every
  registered backbone replaces its rank-deficient projections with it at the composition
  root, with the rank per role set by `config.rank_ratios` (`code/03` S2-13).
  `ChunkedBottleneckLinear` computes `O(D_in +
  D_out + R²)` with cross-chunk interaction only through a `G×G` (or `G→R→G`) bottleneck
  applied to *scalar* chunk pools — the information bottleneck is one scalar per chunk, which
  is a very strong constraint and the interesting thing about it.

---

## I. Ops

`ops/references/` is an oracle layer: readable ground truth for the QK circuit, the CVPE
bias, the FRVM reduction, and a full SDPA path, reached only by explicit import. `dispatch`
resolves `naive → taichi → cpp → reference` with a warning, and *raises* on an explicit
request for an unavailable compiled backend. That is the correct contract: silent downgrade
would make every benchmark uninterpretable.

Two algebra-level notes: the reference uses SDPA mask semantics (bool `True = participate`)
while the module path uses `True = masked`; and the "universal PE" oracle is a stub that
raises, so the claimed oracle coverage is incomplete. Both are cheap to fix and both matter
for the credibility of the kernel work, which is the part of the repository most likely to be
cited by someone else.
