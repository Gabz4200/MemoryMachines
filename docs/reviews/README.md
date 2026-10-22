# Architecture review — `src/memory_machines/models/` and `src/memory_machines/ops/`

Scope: the **science**, not the engineering. Ideas, concepts, algebra, and potential.
Code details are cited only when they change what the architecture *means*.

Author of the review: external read of the repository at commit state of 2026-10-22.
Every factual claim below is either derived from the source or from a probe whose
command is given in `05-evidence-and-experiments.md`.

## Files

| File | Contents |
| --- | --- |
| `01-architecture-thesis.md` | What the design claims, as an argument rather than a module list |
| `02-strengths.md` | The ideas that are genuinely good, ranked, with the published baseline each beats |
| `03-weaknesses.md` | The ideas that are weak, misnamed, or unvalidated, ranked by severity |
| `04-mechanism-audit.md` | Algebra-level audit, mechanism by mechanism, including the unwired modules |
| `05-evidence-and-experiments.md` | Falsifiable claims, the experiments that decide them, and the two probes already run |
| `06-potential.md` | What to keep, what to cut, what is publishable, and in what order |

**Companion engineering review:** `code/` covers the same surface as code — contracts,
state, kernel dispatch, numerics, cost, dead code, remediation order. Read it when the
question is "how is this implemented and what will break", not "is this science right".
`code/README.md` maps its findings to the `S<n>` numbers above.

## One-paragraph verdict

The repository contains **one publishable architectural idea**, **one strong and
underexplored one**, **one interesting but currently unjustified one**, and roughly
**four thousand lines of unwired surplus**. The publishable idea is the separation of
*compression* from *selection* in CSA2, combined with cross-depth *support inheritance*
(`reuse` mode) through a single typed, shared store. The strong idea is the parallel
hybrid — exact local attention ∥ approximate global memory, superposed additively on the
same normalized input. The unjustified idea is the CED encoder→decoder handoff through a
single shared compressed store (Eq. 1). The surplus is mHC, Engram, LIV, rKAN, DPA, NSA,
`ChunkedBottleneckLinear`, `TitansFastWeightsMemory`, HopeBlock/CMS. (`FactoredLinear`
is wired since the rank budget landed: every backbone builds its projections through it
with per-role ratios; see `code/06` §2b.)

The engineering is far ahead of the science. There is no ablation harness, no baseline
comparison, and several mechanisms that are on by default with no measurement of whether
they do anything — including two whose mathematics does not match their names (CVPE is a
learned relative-logit bias, not a positional encoding; the attention "sink" has no value
vector and is therefore a scalar output gate). No result this repository can currently
produce is attributable to any single idea.

That is a fixable problem, and it is cheaper to fix than the code that already exists.
