# Code review — `src/memory_machines/models/` and `src/memory_machines/ops/`

Scope: the **engineering**, not the science. Contracts, state, layering, kernel
dispatch, numerics, cost, dead code. The scientific review lives one directory up
(`docs/reviews/README.md`); where a finding is the code-side face of a scientific
defect, the `S<n>` number from `03-weaknesses.md` is cited instead of restating it.

Every claim below is either read directly from the source (file:line) or produced by
a probe whose command is in the ledger at the end of this file. Nothing is inferred
from a docstring that the code contradicts; where the two disagree, the code wins and
the disagreement is itself a finding.

## One-paragraph verdict

The top of this stack is better designed than most research code: one typed forward
contract, state threaded through arguments, masks derived from absolute positions,
deliberate initialisation, and loud guards where the algebra is fragile. The bottom is
not finished: the ops layer is architecturally orphaned (nothing on the model path
calls it, and it would silently downgrade if it did), attention still materialises
`O(L²)` scores and expands KV by `H/Hkv`, the recurrent memory branch has no decode
state, and roughly 2.4k lines of mechanisms (`liv`, `memory`, `rkan`,
`ChunkedBottleneckLinear`) are unreachable from any model while their config knobs
remain live. The config seam — 139 flat fields read directly by blocks — is what
makes all of that possible. None of it is hard to fix; the ordering matters more than
the effort.

## Files

| File | Contents |
| --- | --- |
| `01-contracts-and-layering.md` | `ModelInput`/`ModelOutput`, `protocols.py`, registry, composition roots, extension seams |
| `02-strengths.md` | What is genuinely good, ranked, with the reason it matters and what to protect |
| `03-defects.md` | Ranked code defects S1–S4, each with evidence, consequence, minimal fix (S2-13: the rank budget is declared per module path) |
| `04-ops-and-kernels.md` | Dispatch semantics, reference duplication, mask polarity, custom-op contract, autograd |
| `05-state-numerics-and-precision.md` | Cache/state semantics, chunk-vs-recurrent parity, dtype policy, sync points |
| `06-dead-code-config-and-cost.md` | Unreachable modules, the 139-field config, per-call allocations, import cost |
| `07-remediation-plan.md` | Ordered phases with acceptance checks |

## Severity ladder

- **S1** — the code is wrong or the capability is absent *today*: wrong numbers, a
  silently ignored API argument, a crash on a supported platform.
- **S2** — the code will break or silently mislead the first time it is used as
  advertised: a cache contract that returns `None`, a knob with no effect, an
  exception path that hides real failures.
- **S3** — maintainability tax with a named future cost: duplicated generation path,
  inheritance used for helper reuse, per-layer device syncs, ambiguous units.
- **S4** — sweep: dead locals, stale docstrings, exported privates, assertions used as
  runtime validation.

## The five things I would do first

1. **Attention core**: `F.scaled_dot_product_attention(..., enable_gqa=True)` with the
   window as an additive bias; drop `repeat_interleave`. Until then the 32k design
   point in the config is aspirational.
2. **Make the config honest**: wire or delete `engram_*`, `mhc_*`, `dspark_*`,
   `liv`/`rkan`/`ChunkedBottleneckLinear`, then split the 139 fields per subsystem.
3. **Wire or delete `ops/`**; stop catching exceptions around the kernel *call*.
4. **One** `global_proj`, **one** masked-decode helper, **one** `forward_shared`
   signature.
5. **Return the cache you already model**, or reject `use_cache=True` and non-trivial
   `attention_mask` loudly at the backbone boundary.

## Mapping to the scientific review

| Scientific finding | Code-side counterpart here |
| --- | --- |
| S13 naming drift | `03-defects.md` S4 (docstring/code disagreements), `05` (unit ambiguity) |
| S14 four thousand unwired lines | `06-dead-code-config-and-cost.md` §1–2 |
| S15/S16 mask handling | `03-defects.md` S2-4 and S2-5, `04` §4 |
| S10 `AdaptiveExitGate` untrained | `03-defects.md` S3 (gate clamp as drift protection) |
| S9 BLT one-patch-behind decode | `03-defects.md` S3-6 (dead prefix keys), `06` §4 (mask cost) |
| S8 auto-associative store | `05` §3 (reuse-mode broadcast is an approximation, documented) |

## Verification ledger

Executed for this review:

| Probe | Command | Result |
| --- | --- | --- |
| Test count | `uv run --no-sync pytest --collect-only -q` | 1555 collected |
| Import cost | `python -X importtime -c "import memory_machines.models.blocks.attention"` | 1313 modules, 3.22 s |
| GDN-2 chunk vs recurrent, strong decay | see `05` §2 | chunk NaN at `T=C=64, g=-8` (fp32); recurrent finite |
| `AttentionSink` overflow | read `compression.py:551-555` | no max-subtraction; fp32 `exp` overflows at logit > 88 |
| `ced_attention_variant` boot validation | `grep -n ced_attention_variant config_schema.py` | declared only, never validated |
| `register_autograd` presence | `grep -rn register_autograd src/` | absent (one comment in `native/torch/autograd.cpp`) |
| `dispatch_*` call sites in `src/` | `grep -rn "dispatch_op(" src/` | none outside `ops/` |

Not verified: CUDA behaviour of any finding, `torch.compile`/export of the custom ops
(no real kernel exists), and runtime behaviour of the unwired mechanisms (no model
constructs them). Claims about those are marked as code-read, not measured.
