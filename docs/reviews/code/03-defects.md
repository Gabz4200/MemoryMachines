# 03 — Code defects, ranked

| # | Defect | Severity | Where |
| --- | --- | --- | --- |
| S1-1 | The ops layer is not on the model path | S1 | `ops/*`, `blocks/attention.py` |
| S1-2 | `except Exception` around the kernel *call* | S1 | `ops/registry.py:101-169` |
| S1-3 | 19 of 20 decoder global-KV projections are dead parameters | S1 | `ced.py:975-977` |
| S1-4 | No KV cache, but an API that returns and forwards one | S1 | `backbone.py`, HF adapter |
| S1-5 | `O(L²)` score materialisation and KV expansion | S1 | `attention.py:377-381, 771-777` |
| S2-1 | ~2.5k lines unreachable from any model, knobs still live | S2 | `06` §1 |
| S2-2 | 139-field config read directly by blocks | S2 | `config_schema.py` |
| S2-3 | `attention_mask` accepted and dropped at the top level | S2 | `backbone.py:131, 389, 647` |
| S2-4 | Two opposite bool-mask conventions | S2 | `attention.py:111` vs `attention_reference.py:44` |
| S2-5 | Inheritance + `delattr`/`Identity` + `strict=False` migration | S2 | `sparse.py:96-99, 135-136` |
| S2-6 | Protocol drift: three `forward_shared` signatures | S2 | `protocols.py`, `sparse.py`, `parallel_hybrid.py` |
| S2-7 | GDN-2 chunk path NaNs under strong decay; `chunk_size` inert by default | S2 | `gdn2.py:245-259, 583-584` |
| S2-8 | Parallel-hybrid memory branch is unmasked and stateless across calls | S2 | `parallel_hybrid.py:302` |
| S2-9 | BLT masked decode projects prefix keys it then masks away | S2 | `blt.py:466-468` |
| S2-10 | `AttentionSink` exponentiates without subtracting a row max | S2 | `compression.py:551-555` |
| S2-11 | `FRVMSpatialOV` bias ships in checkpoints and is never used | S2 | `positioning.py:249-261`, `attention.py:452` |
| S2-12 | `ced_attention_variant` is not validated at boot | S2 | `config_schema.py:386` |
| S2-13 | A projection's rank budget is decided by its module path | S2 | `low_rank.py:131-185` |
| S3-1 | Duplicated masked-decode path across backbones | S3 | `backbone.py:284-307, 566-600` |
| S3-2 | `CausalDecoder.forward` silently truncates its output | S3 | `ced.py:1435-1479` |
| S3-3 | Window units are ambiguous, one window is scaled twice | S3 | `sparse.py:167-196` |
| S3-4 | Device→host sync inside every layer | S3 | 5 sites, `05` §6 |
| S3-5 | `_SharedStateAdapter` silently drops cache and seeding flags | S3 | `parallel_hybrid.py:99-112` |
| S3-6 | `out_proj` init halves depend on `torch.cat` order | S3 | `parallel_hybrid.py:241-245, 303` |
| S3-7 | `ContinuumMemorySystem.should_update` is never called | S3 | `fast_weights.py:1339, 1365` |
| S3-8 | `TitansMAGBlock` and `HopeBlock` discard the memory state | S3 | `fast_weights.py:526, 1423` |
| S3-9 | `reset_parameters` silently skips modules without the method | S3 | `ffn.py:324-332`, `blt.py:181-186` |
| S4 | Sweep list | S4 | end of file |

---

## S1 — wrong today

### S1-1 The ops layer is not on the model path at all

**Evidence.** Nothing in `src/` calls `dispatch_op`, `dispatch_universal_pe_attention`,
or `dispatch_causal_conv1d` (`grep -rn "dispatch_op(" src/` → only `ops/registry.py`).
The model imports the reference directly with a function-local import
(`blocks/attention.py:824-829, 977-983`; `blocks/sparse.py:249-255`) and
`ShortConvolution` imports `ops.references.causal_conv1d_reference` at module scope
(`blocks/causal_conv1d.py:15-20`). Both registered custom ops raise
`NotImplementedError` (`ops/attention.py:37`, `ops/causal_conv1d.py:32`).

**Consequence.** The documented architecture — "reference path everywhere; optimized
kernels behind the same contract" (`ops/__init__.py`, `ml-implementation-preferences`) —
does not exist. Production *is* the naive oracle, so there is no second implementation
to validate against and no way to switch. Two abstractions (`dispatch.py` + `registry.py`,
~350 lines) are maintained for a call graph of zero. Worse, the reference is imported
*inside* `_core_attend` and `_mask_softmax_reduce`, which is a per-call import lookup and
a `torch.compile` graph break.

**Minimal fix.** One of:
(a) route the model through `dispatch_*(backend=config.*_backend)` and implement the
first real kernel; or
(b) delete `ops/dispatch.py`, `ops/registry.py`, `models/protocols.OpImpl`, and move the
references under `blocks/` until a kernel exists.

Do not leave both half-alive. If (a), hoist the imports to module scope.

### S1-2 `except Exception` around the kernel call masks real bugs

**Evidence.** `_try_load_and_call` wraps load *and* call in `except Exception`
(`ops/registry.py:101-116`); `_dispatch_auto` turns any exception into a `UserWarning`
plus the reference (`155-169`). The loader is re-resolved and the warning re-emitted on
every call — no memoised "this backend is unavailable" state.

**Consequence.** A future kernel with a shape bug, a dtype bug, or an OOM is
indistinguishable from "not installed": training silently continues on the reference with
different numerics and different memory behaviour. Explicit backends re-raise as
`NotImplementedError` (`171-185`), destroying the original exception type. This is the
one place in the repo that violates its own fail-fast rule in the hot path.

**Minimal fix.** Resolve once (cache per `(op, backend)`), catch only `ImportError` /
loader-returns-`None`, and let call-time exceptions propagate with their original type.
Keep the reference fallback only for `auto`.

### S1-3 Nineteen of twenty decoder global-KV projections are dead parameters

**Evidence.** `ced.py:975-977` builds `global_projs = nn.ModuleList([CEDGlobalKVProjection(...) for _ in range(dec_n)])`;
only `global_projs[0]` is executed (`1151`, `1286`). The docstring says the list exists
"for the length-zip guard".

**Consequence.** For the default 20-layer decoder, 19 × (`W_K`, `W_V`, score projection)
parameters are created, shipped in every checkpoint, synchronised by DDP, and given
optimizer state — and receive `None` gradient. DDP with `find_unused_parameters=False`
would raise; with `True` it silently pays the broadcast cost. Checkpoints are larger than
the model they describe, and a `strict=True` load from a future single-projection model
fails.

**Minimal fix.** Allocate exactly one projection; keep the `ModuleList` only when a caller
injects `global_projs`. Update the docstring to explain the Eq. 1 sharing.

### S1-4 No KV cache, but an API that returns and forwards one

**Evidence.** `ModelOutput.new_state` is `None` from all three backbones
(`backbone.py:131, 389, 647`); `ModelState` is never constructed; `CSA2.forward_shared`'s
`use_cache` is a documented no-op (`sparse.py:703-706`). The HF adapter accepts
`past_key_values` and `use_cache`, passes `state=past_key_values`
(`modeling_project.py:84`), and returns `past_key_values=output.new_state`
(`105-111`).

**Consequence.** `model.generate()` recomputes the full prefix at every step — `O(n²)`
time, and the 32k design point becomes unusable in practice. A user passing
`use_cache=True` gets no cache *and no error*. The HF contract is silently violated in
the one place external users will hit it.

**Minimal fix.** Short term: raise `NotImplementedError` when `use_cache=True` or
`past_key_values is not None`. Real fix: thread `CompressedCache` (already modelled in
`shared.py:58-72`) through `decode_full_context`, return it as `new_state`, and add one
incremental-decode test asserting `logits[:, -1]` matches a full forward.

### S1-5 `O(L²)` scores and `H/Hkv` KV expansion on every call

**Evidence.** `repeat_interleave` expands K/V by `num_heads // num_kv_heads`
(`attention.py:377-381`, `sparse.py:247-248, 364`); then `masked_fill(-inf)` + fp32
softmax materialises the full `(B, H, Lq, Lk)` tensor (`attention.py:771-777`). No
`F.scaled_dot_product_attention` anywhere on the model path (the only call is in
`fast_weights.py:1479`, an unwired block), although `dispatch.py:100-114` already exposes
`enable_gqa`.

**Consequence.** At `L = 8192`, `H = 16`, fp32: 4 GiB per layer per call just for the
score matrix, plus the expanded K/V. This is simultaneously the memory ceiling and the
bandwidth bill, and it is why the configured 32k context has never been exercised. It
also makes every other optimisation (kernel dispatch, compiled backends) pointless,
because the fused path would have to be written from scratch anyway.

**Minimal fix.** Add a fused core: build an additive bias from the existing
position-derived masks, call
`F.scaled_dot_product_attention(q, k, v, attn_mask=bias, enable_gqa=True)`, and keep
`_score_qk` + explicit softmax as the reference behind `attention_backend="naive"`.
Acceptance: the position-guard tests (`tests/test_attention_position_guards.py`) and the
SWA-replay tests must pass unchanged; add a peak-memory probe at `L=4096`.

---

## S2 — will break or silently mislead when used as advertised

### S2-1 Unreachable mechanisms with live config knobs

`liv.py` (889), `memory.py` (999: Engram, mHC, DSpark), `rkan.py` (473), `low_rank.py` (363), `linear_init.py` (322), `linears.py` (258)
(246) are imported only by `blocks/__init__.py` and tests. No backbone constructs them;
no config switch selects them. Yet `config_schema.py` advertises `engram_*`, `mhc_*`,
`dspark_*` (16 + 5 + 4 fields) and validates them — setting them changes nothing and
emits no warning.

That is worse than dead code: the schema promises a capability. It is also where the
verified defects live (`05` §5): `associative_losses` crashes for `B > 1`, Engram hashing
is CPU-only, LIV double-projects in sparse mode. Anyone who wires a knob inherits them.

**Fix.** Add one selector (`ced_encoder_memory: gdn2|titan|engram|liv`) and one shape
test per path, or move the modules to `experimental/`, delete the knobs, and stop
re-exporting them. See `06` §1-2.

### S2-2 The config is a 139-field god-object

One flat dataclass, namespaced only by prefix, serving three model families and five
memory mechanisms (`config_schema.py`, 804 lines, 139 fields). `GroupedQueryAttention`
alone reads 15 (`attention.py:230-293`); `EngramMemory` reads 16 (`memory.py:690-704`).
Cross-field invariants are enforced in the schema *and* re-checked (or not) in blocks:
`hidden % mhc_num_streams` exists only in `EngramMemory` (`memory.py:707`), so
`hidden_size=10, mhc_num_streams=4` builds fine everywhere else.

**Consequence.** Every test must fabricate a full config; every block is coupled to
every unrelated decision; "interface only where implementations vary" is unachievable.
This is the root cause of S2-1 and S2-12.

**Fix.** `AttentionConfig` / `MemoryConfig` / `CEDConfig` / `BLTConfig` composed into
`MemoryMachineConfig`; block constructors typed against the narrow one. Mechanical but
wide; do it after the dead knobs are removed, not before.

### S2-3 `attention_mask` is accepted and dropped

All three backbones document the drop (`backbone.py:131, 389, 647`). `ExampleBackbone`
even says padded tokens "attend to each other" (`backbone.py:50-56`). HF callers pass
masks routinely; `modeling_project.py:84` forwards it into a contract that ignores it.

**Fix.** Minimum: raise when the mask has any `False`/non-zero entry. Better: pass it to
the SWA branch and use it to exclude pad positions from compressed entries. The
scientific review already ranks this (`../03-weaknesses.md` S15).

### S2-4 Two opposite bool-mask conventions

`_prepare_attention_mask` treats bool `True` as **blocked** (`attention.py:111-113`,
docstring `574`); `build_sdpa_bias` treats `True` as **participate**
(`ops/references/attention_reference.py:44-50`). Both are locally defensible (MHA vs
SDPA). Together they are a trap: the same mask tensor means opposite things depending on
which layer you hand it to, and the reference additionally returns NaN on fully-masked
rows where the model zeros them.

**Fix.** Adopt SDPA semantics (`True` = participate, matching `torch`), convert once at
the boundary, and add a shared test that runs both paths on the same mask.

### S2-5 Inheritance used for helper reuse, plus a checkpoint hack

`CompressedSparseAttention` subclasses `GroupedQueryAttention`, then
`delattr`s `q_proj/k_proj/o_proj` and installs `nn.Identity` (`sparse.py:96-99`),
overriding `_project_qkv` to raise (`135-136`). The parent is used for CVPE/FRVM and head
helpers only.

**Consequence.** LSP is violated by construction: any parent method that touches the
projections now silently multiplies by identity instead of failing. `state_dict` keys
change between versions, so the docstring instructs users to load with `strict=False` —
a migration note instead of a migration, and the mechanism by which a missing tensor
becomes a silent zero.

**Fix.** Extract `_AttentionPrimitives` (CVPE, FRVM, head reshape, mask helpers) as a
mixin or free functions; compose instead of subclassing. Then `strict=True` stays
meaningful.

### S2-6 Protocols are decorative; the block contract is three signatures

`protocols.Decoder` omits `kv_states`, `kv_position_ids`, and `exit_threshold`
(`protocols.py:68-91`). `_validate_encoder_layers` is `isinstance`-based
(`ced.py:75-85`). `CSA2.forward_shared` takes `mode`/`kv_states`/`kv_position_ids`
(`sparse.py:670-682`); `ParallelAttentionGDN2.forward_shared` takes neither
(`parallel_hybrid.py:247-256`); `_SharedStateAdapter` accepts `compress_from` /
`seeded_store` and ignores them (`99-112`).

**Fix.** One optional-kwarg signature defined once in `protocols.py` and imported by all
three; make the adapter raise on unsupported kwargs instead of dropping them.

### S2-7 GDN-2 chunk path NaNs under strong decay, and `chunk_size` is inert by default

**Evidence (measured).** `_chunk_single` divides by the cumulative decay:
`gamma = exp(cumsum(gc))`, `k_bar = kc / gamma` (`gdn2.py:244-245`), and
`k_tail = (gamma_c / gamma) * kc` (`254`). Once the per-chunk sum of `|g|` exceeds
~87 (fp32) or ~709 (fp64), `gamma` reaches 0, `k_bar` becomes `inf`, and `q_gamma @ k_barᵀ`
is `0 · inf = NaN`. Measured with `T = C = 64`, `g = -8` per step, fp32: 512 NaN in the
chunk output, recurrent output finite. At `T = C = 8` the two paths agree to `8.9e-16`.
The recurrent path has no division and never NaNs.

`ced_gdn2_backend` defaults to `"naive"`, which dispatches to
`gated_delta_rule_2_recurrent_forward` (`gdn2.py:583-584`), which passes `chunk_size=None`
(`474`) — so `ced_gdn2_chunk_size` (default 64) has no effect unless the backend is
switched to `"chunk"`/`"wy_fast"`.

**Consequence.** A user switching to the parallel chunkwise backend for speed gets NaN
training exactly when the memory decays fast, which is when the mechanism is supposed to
matter. The parity test cannot catch it because it runs fp64 with moderate decay.

**Fix.** Stabilise the chunk form: subtract the per-chunk max of `cumsum(g)` before
`exp`, or compute `k_bar = kc * exp(-cumsum(g) + cumsum(g).max)` and carry the scale into
`gamma_c`; add a regression at `g=-8, T=C=64, fp32` asserting finiteness and parity.
Document that `chunk_size` is inert under `naive`, or honour it.

### S2-8 Parallel-hybrid memory branch: unmasked, stateless across calls

`forward_shared` calls `memory_out, _, _ = self.gdn2(h)` (`parallel_hybrid.py:302`): no
mask, no `cu_seqlens`, no `past_key_values`, and the returned cache is discarded. The
masked implementation exists but is unreachable (`gdn2.py:952-986`).

**Consequence.** Padded tokens feed the recurrent state and leak into real rows; and at
decode the memory branch starts from a zero state on a single new token, so train-time
full-sequence mixing is not reproduced. The block is the default encoder path
(`ced_attention_gdn2_parallel=True`), so this is the default model.

**Fix.** Pass the mask through (`GatedDeltaNet2` already supports it) and thread the
recurrent/conv cache in the returned shared state. Both are precondition for any
incremental decode work (S1-4).

### S2-9 BLT masked decode projects prefix keys it then masks away

`VocabularyDecoder.decode_masked_patches` builds
`keys = patch_proj(cat([begin_patch, patch_reps]))` (`blt.py:466`) and then restricts
every masked slot to its own latent with `patch_ids != patch_index` (`468`). With
`first_patch_index > 0` the begin patch and every prefix latent are unreachable.

**Consequence.** Wasted projection (`P+1` patch projections per call), and `begin_patch`
is unusable in masked mode despite being constructed for it.

**Fix.** Build keys from `patch_reps[first:]` only and shift the id space accordingly; or
drop `begin_patch` from the masked path.

### S2-10 `AttentionSink` exponentiates without subtracting a row max

`weights = exp(z) / (sum exp(z) + exp(sink))` (`compression.py:551-555`). The comment
reasons only about fp16/bf16 underflow; in fp32 `exp` overflows above ~88, producing
`inf/inf = NaN`. The sink is on by default when `csa_use_sink=True`.

**Fix.** Subtract a per-row max from both numerator and sink logit, or use
`logsumexp` with the sink concatenated as an extra logit. One test with `logits = 100`.

### S2-11 `FRVMSpatialOV` bias is dead in the model path

`FRVMSpatialOV` registers a bias (`positioning.py:249`) and applies it in `forward`
(`261`), but the model extracts only `.weight` (`attention.py:452`) and multiplies it via
`F.linear(grid, weight)`. The bias is in every checkpoint, receives no gradient, and
`module.forward(x) != model(x)`.

**Fix.** Either delete the bias or use the module in `attention.py`. Today only
`tests/test_liv_positioning.py` exercises `forward`, which is why the divergence survives.

### S2-12 `ced_attention_variant` is never validated at boot

Declared at `config_schema.py:386` and described at `142-154`; the only validation is
`_variant_window_attr` at model construction (`parallel_hybrid.py:51-60`). A typo
survives config validation and fails later with `AttributeError` from `getattr` (`190`);
`_build_attention`'s fallthrough would silently map an unknown variant to
`HeavilyCompressedAttention` (`71-77`) if it were reached.

**Fix.** Add the enum to the config's `__post_init__` alongside `ced_encoder_norm` and
`cms_ffn`, and make `_build_attention` raise on the fallthrough.

---

### S2-13 A projection's rank budget is decided by its module path

`linear_role` (`low_rank.py:131-185`) maps a `named_modules()` path to a role, so renaming
an attribute changes its budget: `self.k_proj` → `self.key_proj` no longer matches a leaf,
and `apply_low_rank` now **raises** `ValueError("no rank role for projection ...")` rather
than leaving it dense. The role table is therefore the one declaration site for a projection;
a new block must add its leaf (`None` for a dense invariant) or its path fragment.

**Mitigation today.** The raise makes a rename loud at build time, and
`apply_low_rank` (`low_rank.py:284-363`) warns once per process per role when a ratio in
`(0, 1)` is rejected by the cost guard, so a budget that quietly does nothing is reported
instead of inferred. `tests/test_low_rank.py` additionally asserts that no
reachable `nn.Linear` in `CheapLLM` or `BLTBackbone` still has a role the policy would
factor. A block not built by either backbone (mHC, Engram, LIV perceiver, D-Spark, HOPE,
rKAN) gets no such guard, but its projections must still be declared before they can be
budgeted.

**Fix if the roles keep growing.** Classify from the owning module type instead of the
path string; the table form (`_LEAF_ROLES`, `_PATH_ROLES`) is already a data structure, so
this is a table swap, not a rewrite.

---

## S3 — named future cost

**S3-1 Duplicated masked-decode path.** `CheapLLM.forward` and `BLTBackbone.forward`
contain the same ~40 lines (`backbone.py:284-307` vs `566-600`), and the same five
sub-module constructions with the same `reset_parameters` ritual (`230-263` vs
`507-543`). Extract one `masked_decode(...)` helper plus one `_build_decoder_stack(...)`
into `ced.py`.

**S3-2 `CausalDecoder.forward` silently truncates.** It delegates to `prefill`
(`ced.py:1435-1479`), which returns only the last `window_size` steps when
`swa_bounded_replay` is on. A `forward` returning fewer timesteps than it received is
shape-hostile for `torch.compile` and a footgun for any caller that is not a backbone.
Rename it `prefill`-only or return the full sequence.

**S3-3 Window units are ambiguous and one window is double-scaled.**
`window_size` means compressed blocks for CSA variants but raw tokens for `dense`
(`parallel_hybrid.py:159-160` vs `ced.py:175-179`). `token_window_size = window_size *
compression_ratio` (`sparse.py:167-196`) while the config docstring already describes the
default as `256 * m` (`config_schema.py:108-116`, defaults `365-367`): the configured
1024 becomes 4096 tokens. `config_schema.py:148-154` documents
`ced_encoder_first_window_size` as `window_size_sparse * ratio`, but `_build_encoder`
uses `or config.window_size`. Encode units in the names (`_blocks` / `_tokens`),
convert once at build, assert at boot.

**S3-4 Device→host sync inside every layer.** `.item()` at `attention.py:696`,
`shared.py:122`, `compression.py:140`, `blt.py:100`, `fast_weights.py:1243-1244`. Each
serialises the GPU pipeline once per layer; the position one is derivable from the
cached length plus the window.

**S3-5 `_SharedStateAdapter` drops cache and seeding flags.** `99-112` discards the
inner cache and ignores `compress_from`/`compress_start_pos`/`seeded_store`. Silent
no-ops are worse than `TypeError`.

**S3-6 `out_proj` init halves depend on `torch.cat` order.**
`parallel_hybrid.py:241-245` initialises columns `[:d]` with trunc-normal and `[d:]` with
`memory_init_scale`; the concatenation is `cat((attn_out, memory_out))` at `303`.
Reordering the `cat` silently swaps the init schedule. Name the halves and slice by
name, or initialise each branch's projection separately.

**S3-7 `ContinuumMemorySystem.should_update` is never called.** Defined at
`fast_weights.py:1339`, `forward` at `1365-1372` never consults it; `HopeBlock` only
copies intervals into the cache (`1427`). `update_intervals` is inert metadata.

**S3-8 `TitansMAGBlock` and `HopeBlock` discard the memory state**
(`fast_weights.py:526`, `1423`), so neither can be used incrementally; re-running on a
tail re-memorises from scratch.

**S3-9 `reset_parameters` silently skips modules lacking the method**
(`ffn.py:324-332`, `blt.py:181-186, 364-369`). The docstring calls it a contract
violation, then the code stays silent. Raise instead.

---

## S4 — sweep

- `ced.py:1146-1147`: `if not layers: return hidden_states` makes a 0-layer decoder an
  identity. Should fail in config validation.
- `ced.py:516-521`: `shared = None` is assigned and immediately overwritten.
- `sparse.py:223`: `_core_attend` docstring documents a `bidirectional` argument it does
  not have.
- `assert` used as runtime validation: `blt.py:313`, `backbone.py:605`, `ced.py:1330`.
  Vanishes under `python -O`. Use `raise`.
- `blocks/__init__.py`: exports privates (`_sinkhorn`, `_window_mask`,
  `_causal_comp_mask`, `_past_total_len`, `_resolve_positions`, `_check_position_bounds`,
  `_update_swa_pos`) and eagerly imports all 77 symbols (1,313 modules / 3.22 s before the low-rank split).
- `ffn.py:52-53`: `SwiGLU` rejects odd `intermediate_size`, but the schema only requires
  positive (`config_schema.py:325`), so the failure lands at model construction. Validate
  where the value is defined, or drop the constraint (`ClassicMLP` does not have it).
- `ffn.py:343-349`: docstring says `step` is a forward-call counter; call sites pass the
  layer index (`ced.py:256, 1172`) or a constant `0` (`1324, 1376`). Fix the docstring.
- `models/registry.py`: registration depends on import order; add an explicit
  `register_all()`.
- `ExampleBackbone.backend` is stored and never read (`backbone.py:79-92`).
- `ShortConvolution.state_size`, `OpSpec.torch_op`, `ensure_taichi`, `DyT.hidden_dim`:
  defined, never read.
- `fast_weights.py`: `_fast_shapes` implemented three times (102-109, 204-205, 923-924);
  the gate-update body transcribed in three places (289-308, 408-425, 613-651 vs
  683-796).
- `liv.py:880-885`: `masked_hidden` is computed and immediately superseded (`_masked_mix`
  dead work).
- `compression.py:139-140`, `blt.py:89-100`: host syncs inside validation, per call.
