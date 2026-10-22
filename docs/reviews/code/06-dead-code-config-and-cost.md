# 06 — Dead code, config debt, and cost

## 1. Unreachable mechanisms

Imported only by `blocks/__init__.py` and tests; no backbone constructs them, no config
field selects them.

| Module | Lines | Contents | Status |
| --- | --- | --- | --- |
| `liv.py` | 889 | `LivDoubleGatedBase`, `LivConvBlock`, `LivPerceiver` (two-pass compressed cross-attention) | correct only in non-sparse mode; sparse mode double-projects (`05` §5.4) |
| `memory.py` | 999 | Engram (hashed n-grams, SSD offload), `SinglePassMHC`, `DSparkDrafter`, `to_streams` | 4 verified defects (`05` §5.1-3, 6) |
| `rkan.py` | 473 | rational-KAN FFN variant | unused, unvalidated at config level |
| `linears.py` | 258 | `FactoredLinear`, `ChunkedBottleneckLinear`; `linear_init.py` (322) and `low_rank.py` (363) hold the init helpers and the rank policy | **wired**: every backbone builds through `FactoredLinear` (§2b); `ChunkedBottleneckLinear` still has no consumer |
| `fast_weights.py` | 1480 | Titans, SRT, CMS, HopeBlock, `HopeAttentionBlock` | partly reachable (`SelfReferentialTitanMixer` via the hybrid branch); the Blocks are not |

Cost is not the line count; it is:

1. **Schema surface.** `engram_*` (16 fields), `mhc_*` (5), `dspark_*` (4), `cms_*` (3),
   `rkan_*` are validated at boot. A user who sets them gets no effect and no warning.
2. **Maintenance.** `blocks/__init__.py` re-exports 77 symbols including these, so every
   rename is a public-API change and every import pays the compile/import cost.
3. **False coverage.** Tests exercise the modules directly, so line coverage looks high
   while no model path is proven. The defects in `05` §5 are exactly the ones a
   model-level test would have caught.
4. **Duplication of the thesis.** `SelfReferentialTitanMemory` and `LivPerceiver` both
   re-implement pieces of the CSA2 store; `ContinuumMemorySystem` re-implements a
   multi-rate FFN. The scientific review counts these as surplus (`../03-weaknesses.md`
   S14) because they compete with the one idea the repo can defend.

**Decision required (not a refactor):** pick one of

- **(a) Wire them.** Add `ced_encoder_memory: gdn2|titan|engram|liv|none` and one shape
  test per value; keep the knobs, fix the defects in `05` §5 first.
- **(b) Isolate them.** Move to `src/memory_machines/experimental/`, delete the config
  fields, stop exporting from `blocks/__init__.py`, keep the tests as unit tests of the
  experimental package.
- **(c) Delete.** `rkan` and `ChunkedBottleneckLinear` have no consumer and no published
  claim; they are the two I would delete outright (`FactoredLinear` is now on every path).

Do not leave the current state: a knob that does nothing is a claim that the feature
works.

## 2. Knobs that do nothing today

| Knob | Why it is inert |
| --- | --- |
| `ced_gdn2_chunk_size` | backend defaults to `naive` ⇒ recurrent path, `chunk_size=None` (`gdn2.py:474, 583-584`) |
| `csm_*`/`cms_update_intervals` | `should_update` never called (`05` §5.7) |
| `engram_*`, `mhc_*`, `dspark_*` | no construction path (§1) |
| `csa_use_sink` | used, but the sink is numerically unsafe (§2 in `03`) |
| `ExampleBackbone.backend` | stored, never read (`backbone.py:79, 92`) |
| `attn_weights` in `ModelOutput` | never populated |
| `pixel_values` in `ModelInput` | no consumer |
| `model.name` for `liv`/`rkan` | no registration |
| `titan_use_qk_projection` | silently ignored on `naive`/`recurrent` (`fast_weights.py:817-827`) |
| `blt_latent_temperature` | only read when `blt_latent_sampling=True`; documented as such |

Each is small. Together they are the reason a new user cannot tell which half of the
config is real. The fix is not to delete the fields but to make the model *reject* a
non-default value it cannot honour — a `__post_init__` rule per subsystem ("`engram_*`
requires `model.name='blt'` and `blt_use_engram=True`") is enough.

## 2b. Rank budget

`apply_low_rank` runs once per backbone (`backbone.py`) and replaces every projection
whose role budget is both requested and strictly cheaper to factor
(`low_rank.py:284-363`). At the default 32-wide test config that is −19% parameters for
`CheapLLM` (`tests/test_low_rank.py`). The saving is bounded by the
parameter algebra: factoring an `in x out` projection costs `rank * (in + out)` against
`in * out`, so a square projection only gets cheaper below 50% rank. That is why the
Value/Output/FFN-Up/Down entries (0.75) do *not* fire on square or expanding matrices:
0.75 of `d` costs 1.5x dense. Cutting those honestly means cutting them below 0.5, which
the spectral evidence says is the information bottleneck, so the budget is refused and
the projection stays dense.

## 3. Config structure

`MemoryMachineConfig` is 139 fields in one dataclass (`config_schema.py`, 804 lines).
Distribution: attention/CSA ~35, CED ~30, BLT ~25, GDN-2 ~14, Titans ~10, Engram ~16,
CMS ~3, mHC ~5, DSpark ~4, training/model ~10 (the last now includes `low_rank_mode`
plus `rank_ratios`, which is one field carrying 31 per-role ratios).

Three concrete costs:

- **A block cannot be built from its own config.** `GroupedQueryAttention.__init__` reads
  15 fields (`attention.py:230-293`); `EngramMemory` reads 16 (`memory.py:690-704`).
  Tests fabricate the full object to test one projection.
- **Validation is duplicated and inconsistent.** `hidden % mhc_num_streams` is enforced
  only inside `EngramMemory` (`memory.py:707`) — nowhere in the schema — so a config that
  builds fine everywhere else crashes when Engram is constructed. Conversely
  `ced_attention_variant` is enforced only at model build (`03` S2-12), `SwiGLU`'s
  even-width rule only in the block (`ffn.py:52`), and `beta_guard ⇒ use_qk_l2norm` only
  at first forward (`gdn2.py:365-366`).
- **Two table caps.** Config caps `len(orders)*hash_heads*entries*dim <= 268M`
  (`config_schema.py:642-644`); `_SsdTable` caps `sum(primes)*dim` (`memory.py:194-195`);
  the RAM path (`nn.Embedding`, `747`) has no cap at all.

Target shape:

```python
@dataclass(frozen=True)
class AttentionConfig: ...      # heads, kv heads, window blocks, variant, sink, indexer
@dataclass(frozen=True)
class MemoryConfig: ...         # gdn2/titan/engram/liv selection + their knobs
@dataclass(frozen=True)
class CEDConfig: ...            # encoder/decoder depth, widths, loops, FFN kinds
@dataclass(frozen=True)
class BLTConfig: ...            # byte models, patch sizes, latent sampling

@dataclass
class MemoryMachineConfig:
    attention: AttentionConfig
    memory: MemoryConfig
    ced: CEDConfig
    blt: BLTConfig
```

Do this *after* §1 (otherwise you namespace dead fields), and keep a flat read-only
compatibility property for one release so `scripts/` and configs keep working.

## 4. Per-call cost

Measured line counts, not profiled times; the arithmetic is what matters.

| Site | Allocation | At `B=1, L=4096, H=16` |
| --- | --- | --- |
| Attention scores (`attention.py:771-777`) | `(B,H,Lq,Lk)` fp32 | 1.07 GB per layer |
| KV expansion (`attention.py:377-381`) | `(B,H,Lk,Dh)` × `H/Hkv` | 4× the K/V bytes |
| Indexer scores (`compression.py:258`) | `(B,Lq,Hidx,Lc)` fp32 | 537 MB per layer with `Hidx=32, m=4` |
| CVPE bias (`positioning.py:206-215`) | `(B,Lq,Lk)` fp32 + `(B,Lq,Lk,mlp)` | 1.07 GB + 2.1 GB |
| BLT mempool mask (`blt.py:331, 468`) | `(B,P,N)` bool × layers | 268 MB per layer at `N=32k, P=8k` |
| `GroupedOutputProjection` (`compression.py:648`) | `(B,Lq,num_groups*group_dim)` | 8192-wide intermediate at defaults |

No buffers are used for `arange`/`eye`/mask constants: `torch.eye(chunk_size)` is
allocated per head per segment (`gdn2.py:234`), `arange` per forward at eight sites,
`F.pad` per call in the compressor (`compression.py:124`). On CPU this is noise; on an
accelerator it is allocator churn and, for `torch.compile`, recompilation.

The CVPE row is the one to fix first: the offset MLP materialises a
`(B, Lq, Lk, mlp_hidden)` tensor, which is 2.1 GB *per layer* at 4k context, and there is
no query chunking. `gated_reduction` chunks its reduction (`attention_reference.py:85-103`)
but the spatial grid is materialised outside the chunk loop (`attention.py:451`), so the
bound it buys is defeated.

## 5. Import graph

`blocks/__init__.py` imports all 45 exports eagerly. Measured: importing
`memory_machines.models.blocks.attention` loads **1,313 modules in 3.22 s** and pulls
`fast_weights` (1,480 lines) and `ops` into the process.

Consequences: a doctest of `attention.py` pays the whole import cost; a CLI that only
needs the reference conv pays for Titans; and circular-import risk grows with every new
block. Fix: keep `blocks/__init__.py` for the public surface but make it lazy
(`__getattr__` module hook), and stop exporting privates. This is mechanical and does not
change behaviour.

## 6. Test-suite shape

1,547 tests, high line coverage — and the defects above are invisible to it. The reason
is structural, not effort:

- No test constructs a model with `engram_*`/`liv`/`rkan` selected, because no model can.
- No test asserts a cache round-trip, because no backbone returns one.
- No test asserts mask semantics across the reference boundary, because nothing calls it.
- No test asserts that every config field is read (`139` fields, `grep` audit absent).

Three cheap meta-tests would close most of the gap:

1. **Config-field audit**: every field in `MemoryMachineConfig` is read by at least one
   `src/` module (`ast` walk over attribute accesses). Any field read by none fails.
2. **Construct-and-step matrix**: for each registered model × each reachable mechanism
   selector, build the smallest config and run one forward + one backward.
3. **Backend parity**: for each `OpSpec`, assert
   `dispatch_op(name, "naive") == dispatch_op(name, "auto")` and that both agree with the
   model path on a shared mask, once a kernel exists.

Each is a test that fails when a knob becomes dead, which is the class of defect this
review keeps finding.
