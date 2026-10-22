# 06 — Potential: what to keep, what to cut, what is publishable

## The honest ranking of the ideas by expected value

Expected value = (novelty) × (probability the claim survives a fair test) × (cost of the test
being low). The probabilities are my subjective estimates, stated so they can be argued with.

| Idea | Novelty | P(survives fair test) | Test cost | Verdict |
| --- | --- | --- | --- | --- |
| Compression ≠ selection (split operators) | high | 0.6 | 1 run | **Keep, headline** |
| Support inheritance across depth (`reuse`) | high | 0.5 | 1 measurement + 1 run | **Keep, headline** |
| Explicit single bottleneck (Eq. 1) | medium | 0.5 | 1 run (oracle gap) | **Keep, second paper** |
| Parallel hybrid, additive superposition | medium | 0.4 | 2 runs | Keep as ablation |
| GDN-2 family + resting state + beta guard | medium (analysis) | n/a | 0 | Keep as theory note |
| Titans meta-learned inner loop | medium | 0.5 (works) / 0.15 (worth its cost) | 2 runs | Keep, measure cost |
| BLT byte decoding | low (reimplementation) | n/a | — | Fix the lag, then keep |
| DyT | low (published) | 0.5 | 1 run | Ablate, do not defend |
| CVPE | low-medium | 0.25 | 1 run | Ablate; expect to lose to RoPE |
| FRVM | medium | 0.3 | 1 run | Reformulate with residual, then ablate |
| Attention sink (as implemented) | none | 0.3 | 1 run | Fix the value vector or drop |
| mHC / Engram / LIV / CMS / rKAN / DPA / NSA / linears | 0 (w.r.t. published work) | — | — | **Cut or park** |
| Oracle ops + kernel parity | medium (methodology) | 1.0 | already done | **Keep, publish as artifact** |

## The two papers that are actually here

### Paper A (recommended): *Sparse support is a property of the network, not of the layer*

The claim: **the top-k support of a compressed memory can be computed once and inherited
across depth**, turning support selection from a per-layer cost into a budget allocated over
the stack, with three fidelity levels (`full`, `reindex`, `reuse`).

Why it is a paper and not a blog post:

- It is a *mechanism* claim, so it needs one table (E1, E2, E14) and no new architecture.
- It has a clean control (`reindex` everywhere) and a clean null (support is unstable).
- It produces a new measurable quantity — support drift with depth — that is interesting even
  if the mechanism loses. A negative result ("support drifts by layer 4, so inheritance must
  be re-anchored") is publishable and useful to everyone building sparse attention.
- It is cheap: 100M–300M parameters, tens of billions of tokens, three seeds.

The paper writes itself around one figure: Jaccard overlap and mass-outside-support as a
function of depth, for several models (yours, a plain sparse-attention model, and a trained
GPT-style baseline with the same indexer bolted on). If the curves are flat, inheritance is
sound and the cost claim follows. If they are not, you have measured something nobody has
measured.

### Paper B (riskier, higher ceiling): *How much context fits in one vector set?*

The claim: an asymmetric encoder–decoder with a single explicit compressed bottleneck (Eq. 1)
matches a full-attention decoder at a fraction of decode cost.

This needs E10 (oracle gap) as its core experiment, because the interesting quantity is not
perplexity but the **compression frontier**: quality as a function of store size
`L/m · H_kv · D_h`. Sweep `m` and `H_kv`, plot quality vs bytes of state. That curve is a
result whether or not your architecture wins, and it directly answers "how cheap can memory
be?" — the question the project is named after.

The risk: the one-patch-lag BLT result (S9) and the auto-associative store (S8) both suggest
the current bottleneck is weaker than it needs to be. Fix both before running the frontier,
or the curve will measure the wrong thing.

## What to cut, and why cutting is a scientific act

Cut: `EngramMemory`, `LivConvBlock`, `LivPerceiver`, `HopeBlock`, `HopeAttentionBlock`,
`ContinuumMemorySystem`, `TitansFastWeightsMemory` (the duplicate standalone),
`NativeSparseAttention`, `DSparkDrafter`, `JacobiRKAN`/`PadeRKAN`.

Park with a one-line note: `SinglePassMHC` (only the shared-`B` two-branch variant is novel
enough to test), `ChunkedBottleneckLinear` (interesting bottleneck, no test).

The reason this is science and not housekeeping: a reader cannot tell which mechanism the
paper is about when four thousand lines are unwired. Every cut increases the signal-to-noise
ratio of the claim. The repository already contains one and a half papers; the surplus is
diluting them.

## The structural fix that must happen first

`x ← M(x)` with a small `out_proj` is a multiplicative contraction (`04` §A, measured: `2e−4`
at encoder layer 0). Fixing this is not optional and it is not tuning: it is the difference
between a network whose depth does something and a network whose depth is one layer followed
by near-identity maps. Either restore the residual or remove the depth scaling. Then re-run
every number in the repository, because none of them were produced by a network that was
actually using its depth at initialisation.

This is also the most likely explanation for any "our model trains slowly" observation, and
it should be checked before any architecture conclusion is drawn.

## What the architecture could become

Three directions that are genuinely enabled by what exists, in increasing order of ambition:

1. **Support as a program.** If support is stable across depth, it is also plausibly stable
   across *tokens* — which would make it cacheable across a decode step, and eventually
   predictable. Predicting tomorrow's support from today's is a small, well-posed learning
   problem with an obvious payoff (zero indexer cost at inference). Nothing in the repository
   does this, and everything needed to try it is already there.

2. **The bottleneck as an interface, not an artifact.** Eq. 1 already makes the compressed
   store external to the decoder. That is one step away from treating it as an *interface*:
   a fixed-size summary that can be written by one model and read by another, cached across
   requests, or produced by a cheaper model than the one that reads it. The masked
   Perceiver-AR path is the natural testbed (write once, decode L latents in parallel), and
   it is the closest thing in the repository to a genuinely new capability rather than a
   cheaper operator.

3. **The parallel hybrid as a general composition rule.** If exact-local ∥ approximate-global
   works at one scale, the interesting question is whether the *ratio* of the two branches
   should vary with depth (local early, global late) or with position. That is a
   testable refinement of the hybrid literature, and your asymmetric init makes it
   measurable without retraining from scratch.

## The ceiling, honestly

The realistic best outcome of this repository, executed well, is:

- one mechanism paper on support inheritance, with a new measurable quantity and a clean cost
  model;
- one methodology artifact — the oracle/dispatch/parity discipline plus the mechanism
  telemetry panel — that other people reuse;
- a byte-level long-context model whose compression frontier is documented.

That is a good outcome. It is not a new architecture family, and nothing in the current code
supports the larger claim. The grand-architecture framing ("a cheap LLM built from CSA2 +
GDN-2 + Titans + mHC + Engram + CMS + BLT") is the weakest thing in the repository precisely
because it is the sum of nine unmeasured parts; the strongest thing is one measured part.

## Ninety days, concretely

1. **Week 1** — fix the contraction (E4/E5); add telemetry; fix `ExampleBackbone` padding;
   delete or park the unwired modules.
2. **Week 2** — E1 on whatever checkpoint exists; E3.1 zero-ablation. Write the two figures.
3. **Weeks 3–6** — E2 at 100M/10B, three seeds: `full` vs `reindex` vs `reuse` at equal FLOPs.
   This is the paper.
4. **Weeks 7–10** — E6, E7, E10 in parallel; each is one run and each closes a named
   criticism from `03-weaknesses.md`.
5. **Weeks 11–12** — the compression frontier (E10 sweep) and the write-up.

If only one thing gets done, do E1 and E4. E1 decides whether the architecture's central cost
claim is true; E4 decides whether the network is using its depth at all.

## Closing assessment

This is a serious research repository built by someone with real architectural taste: the
separation of compression from selection, the decision to make state explicit and shared, and
the discipline of zero-initialising every new mechanism are all choices a good researcher
makes. What it lacks is not ideas but *adjudication* — the willingness to let a small number
of experiments kill most of them. Nine mechanisms on by default is not ambition; it is
deferred judgement. The single highest-value change is not architectural at all: pick three
claims, write down what would falsify each, and run those experiments before writing another
module.
