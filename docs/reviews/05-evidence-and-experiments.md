# 05 — Evidence: what is already measured, and the experiments that decide the claims

Everything below is either already run (with the command) or specified so that it can be run
without further design work. No experiment here needs a new mechanism.

---

## Part 1 — Probes already run

### P1. Structural causality of the CED path — PASS

```bash
# /tmp/opencode/leak_probe.py: tiny CheapLLM, 16 tokens, perturb the last token
uv run --no-sync python /tmp/opencode/leak_probe.py
```

Result: per-position `max|Δlogit| = 0` for positions 0–14 and non-zero only at position 15,
both with and without `ced_decoder_looping`. So the seeded store, the `reuse`/`reindex`
cache, and the Perceiver-AR position arithmetic do not leak future information. (The same
probe on the BLT decoder, with a patch-level perturbation, gave the same verdict: byte in
patch `p` reads only `patch_reps[p−1]`.)

**Scope of the claim:** causality of the *implemented* configuration at one tiny size. It is
not a proof for every schedule, and it depends on `start_pos`/`position_ids` defaults rather
than on an explicit mask invariant — worth a regression test.

### P2. Initialisation scale — FAIL

```bash
# same script family: vary hidden/layers, measure logits.std()
```

| config | logit std |
| --- | --- |
| hidden 32, 4+3 layers, `init_range=0.006` | 5.2e−10 |
| hidden 128, 4+4, `init_range=0.006` | 8.6e−08 |
| hidden 256, 8+8, `init_range=0.006` | 8.2e−07 |
| hidden 32, 4+3, `init_range=1.0` | ~1e+17 |

### P3. Where the attenuation happens — encoder layer 0, multiplicatively

```bash
# per-layer ‖x‖ and per-layer FFN delta, default init
```

| stage | ‖x‖ | ratio |
| --- | --- | --- |
| embed | 2.29e+01 | — |
| encoder mixer 0 | 4.59e−03 | 2.0e−04 |
| encoder FFN 0 (delta) | 5.46e−12 | 1.2e−09 |
| encoder mixers 1–5 | ≈ 3e−03 | ≈ 0.92–1.16 per layer |

So: **one multiplicative contraction at the first replacement mixer, then near-identity
propagation, with the FFN contributing nothing at init.** The cause is structural
(`x ← M(x)` with a small `out_proj`), not a hyperparameter, and this is the single most
important measured fact in the review.

---

## Part 2 — The claims, and the experiment that decides each

Format: **claim → experiment → metric → decision rule.** "Equal compute" means matched
training FLOPs, not matched steps; every comparison is at 3 seeds with mean ± std and a
decision at the 95% CI.

### E1 — Support stability (decides the `reuse` claim)

Claim: *the top-k support is stable across depth, so it can be computed once.*

No training needed. Take a partially trained model (even 1B tokens), run forward on held-out
data, and for each layer compute `Jaccard(top_k(layer 1), top_k(layer i))` for `i = 2…19`;
also compute the *regret*: with a freshly computed score at layer `i`, what fraction of
attention mass falls on entries **outside** layer 1's support.

Decision rule: if mean `Jaccard < 0.5` or regret `> 10%` of mass, `reuse`-forever is
unsound and the `full` point must move to mid-stack (or be repeated).

This is a 20-minute measurement on any checkpoint and it directly prices the architecture's
main cost claim. Run it before anything else.

### E2 — `reuse` vs `reindex` vs `full` (decides the cost claim)

Three configs, identical parameters, equal *compute* budget (so the sparse configs get more
tokens), 100M parameters, ≥10B tokens. Metric: held-out perplexity at equal FLOPs, plus
wall-clock throughput.

Decision rule: `reuse` must beat `full` at equal FLOPs by a margin larger than seed noise;
if it only ties, the store-scan savings are real but the accuracy cost cancels them, and the
honest claim becomes "equal quality at lower cost," not "better."

### E3 — Is the memory branch alive? (decides C2)

Two measurements, no retraining:

1. Test-time ablation: zero the memory half of `out_proj`, report `Δ` perplexity.
2. Training telemetry: `‖W_mem‖ / ‖W_attn‖` per layer over training.

Decision rule: if `Δ ppl < 1%` and the ratio stays near `1e-3`, the branch is decorative and
the parallel-hybrid claim is void at the current window/length. Then either train at
≥4× window or drop the branch. This experiment is cheap and it is the one most likely to
change the paper's story.

### E4 — Residual vs replacement mixer (decides the CED stack)

Add `x ← x + M(x)` in the encoder and decoder, everything else fixed, equal tokens.
Three arms: (a) current replacement + small init, (b) replacement + `init_range` init,
(c) residual + small init.

Metric: loss at fixed token budget, plus logit std at init.

Prediction from the algebra: (c) is fastest, (b) trains but noisily, (a) is the slowest by a
wide margin. If (a) is not clearly worst, the algebra in `04` §A is wrong and the review's
main structural criticism should be retracted.

### E5 — Initialisation scheme (decides S3)

Standard split (`initializer_range` for embed/head, `initializer_range/√(2N)` for residual
outputs, `init_range` for replacement mixers) vs the current single global value. Acceptance
test at init: logits std in `[0.1, 1]`, no layer with `‖Δ‖/‖x‖ < 1e−6`.

### E6 — Position: CVPE vs the alternatives (decides S4)

Four arms at equal parameters: none, CVPE, ALiBi (`−c·r` per head), RoPE on q/k (keeping
everything else). Three metrics:

1. Perplexity at train length 512.
2. Length extrapolation: evaluate at 2048 and 4096 without retraining.
3. **Position-decodability probe**: train a linear classifier from each layer's residual
   stream to the token's position (bucketed). This directly tests the claim "position is a
   prior, not a representation": if the probe is near chance for CVPE but near-perfect for
   RoPE, the criticism is confirmed independently of perplexity.

Decision rule: CVPE must beat ALiBi on (1) or (2) to be worth its cost; if it loses
extrapolation by a wide margin, the missing in-stream representation is the reason.

### E7 — FRVM: does it help, and does it break copying? (decides S5)

Three arms: vanilla `Σwv`, FRVM as written, residual-FRVM (`Σwv + λ·Σw[v⊙σ(v+φ(Δv))]`,
`λ` zero-init). Two metrics: held-out perplexity, and a **copy/induction probe** (sequences
with a repeated random token; measure the model's ability to predict the token following a
previous occurrence, at matched loss).

Decision rule: if vanilla wins the copy probe, the base-term nonlinearity is a real
expressivity loss and residual-FRVM is the only defensible form.

### E8 — The sink: value vector or gate? (decides S7)

Three arms: no sink, current sink, sink + learned value vector. Metric: perplexity, plus
`p_sink` distribution per head (is it used at all?).

### E9 — Auto-associative vs split K/V store (decides S8)

Add a low-rank value projection to `BlockCompressor` (`D_h → r → D_h`, `r = D_h/4`). Two
arms, equal parameters. Metric: perplexity, plus a **retrieval probe**: can the model answer
a query whose key and value are different tokens (associative recall)?

### E10 — Is one store enough for twenty decoder layers? (decides C1/S6)

Three arms: (a) current single store, (b) per-layer store projections (already instantiated —
just wire them), (c) oracle: full encoder stream instead of the compressed store (upper
bound). Metric: perplexity; plus per-layer **store knockout** (zero the store's contribution
at layer `i` only) to see which layers actually use it.

Decision rule: if (c) ≈ (a), the bottleneck is not binding and the compression ratio can be
increased — a strong result for the "cheap" thesis. If (a) ≪ (c), the gap prices the
bottleneck and tells you what to fix (wider store, split K/V, per-layer projection).

### E11 — BLT: one-patch lag vs two keys vs all patches (decides S9)

Three arms: `{P_{p−1}}`, `{P_{p−1}, P_p}`, dense cross-attention over all patch latents.
Metric: bits-per-byte at equal compute, plus per-position bpb *within* a patch (the lag
hypothesis predicts the first byte of each patch is worst).

### E12 — Titans vs GDN-2, homogeneous (decides S11)

Four arms: all-GDN-2, all-Titans, alternating (current), alternating with `create_graph`
disabled. Equal FLOPs, same token budget. Metrics: perplexity, wall-clock, peak memory.

### E13 — Exit gate with a compute penalty (decides S10)

Two arms: hard-selection only (current), `+ τ·E[t]`. Metric: quality at **matched average
compute** (sweep `τ` and `q`), which is the only fair early-exit comparison. A Pareto curve,
not a single number.

### E14 — Routing hypothesis class (cheap, decides the indexer's ceiling)

Arms: `ReLU` (current) vs signed scoring; zero-init vs random-init `weight_proj`. Metric:
perplexity, plus the E1 regret measure. If signed scoring reduces regret materially, the
monotone-aggregative restriction is real.

---

## Part 3 — Telemetry worth building once

A single logging panel, emitted every `n` steps, would have caught both P2/P3 and S2 without
any experiment:

| quantity | why |
| --- | --- |
| `logit.std()` | detects the dead/exploding init regimes |
| per-layer `‖Δmixer‖ / ‖x‖`, `‖Δffn‖ / ‖x‖` | detects replacement-mixer contraction and dead FFNs |
| `‖W_mem‖ / ‖W_attn‖` per parallel block | detects a decorative memory branch |
| support `Jaccard(layer 1, layer i)` + mass-outside-support | prices `reuse` continuously |
| mean `p_sink` per head | detects an unused sink |
| `‖φ‖`, `‖γ_cvpe‖`, `‖β_cvpe‖` | detects unused CVPE/FRVM |
| latent `σ` mean | detects KL collapse in BLT sampling |
| mean exit step, exit-step histogram | detects an uncalibrated exit gate |

This panel is the difference between "the loss went down" and "the mechanism did what the
paper says."

---

## Part 4 — Order of operations

**P0 — no training, days.** Fix `ExampleBackbone` padding; fix the init split; add telemetry;
run E1 (support stability) and E3.1 (zero-ablate the memory branch) on the first checkpoint
that exists. Two of the review's three most severe criticisms are decided here.

**P1 — one training run each, ~100M / ≥10B tokens.** E4 (residual vs replacement) and E5
(init). These are the two changes most likely to alter the loss curve, and E4 is a
falsification test of this review's main structural claim.

**P2 — mechanism ablations.** E2 (`reuse` cost), E6 (position), E7 (FRVM + copy probe), E10
(store sufficiency). These produce the paper's main table.

**P3 — secondary.** E8, E9, E11, E12, E13, E14.

**Reporting contract.** Every number: 3 seeds, mean ± std, equal-FLOPs and equal-parameters
columns, and the exact config hash. A mechanism that is not in a row of the table is not part
of the architecture, whatever the code contains.
