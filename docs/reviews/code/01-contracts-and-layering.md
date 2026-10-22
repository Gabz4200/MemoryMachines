# 01 — Contracts and layering

## What the layering claims

```
scripts/ (Hydra) ──> modules/lightning_module.py ──> models/backbone.py
                                                        │
                                     ┌──────────────────┼──────────────────┐
                                blocks/ced.py      blocks/blt.py       blocks/sparse.py
                                     │                                     │
                                     └──────────> ops/ (dispatch) <────────┘
```

The claim, stated in `models/backbone.py:1-10`, `models/io.py:1-10` and
`models/protocols.py:1-8`: backbones are pure `nn.Module` composition roots speaking
`ModelInput -> ModelOutput`; the framework layer (Lightning, Hydra, callbacks) never
enters; extension is structural, via PEP 544 protocols.

## What the code actually enforces

**Holds.** No import from `torch.nn.functional`-level framework code into `models/`;
no Lightning or Hydra import under `models/` or `ops/`; `LitResearchSystem` is the only
place that owns loss and optimizer (`modules/lightning_module.py`). Every backbone
implements `__call__`, `get_input_embeddings`, `set_input_embeddings`, which is what
`integrations/transformers` needs.

**Does not hold.** Three separate claims are decorative:

1. **`protocols.Decoder` is not the interface the model uses.** It declares
   `decode_full_context(hidden_states, encoder_final, position_ids, *, start_pos,
   seeded_store, exit_threshold, num_masked_tokens)` (`protocols.py:68-91`). The real
   method also takes `kv_states` and `kv_position_ids` (`ced.py:1075-1087`), and
   `CausalDecoder.forward` is a *different* method that truncates (`ced.py:1435`).
   A conforming implementer of the protocol cannot be substituted.
2. **`protocols.TokenMixer` is not how layers are validated.** `_validate_encoder_layers`
   is `isinstance(layer, (CompressedSparseAttention2, ParallelAttentionGDN2, GroupedQueryAttention))`
   (`ced.py:75-85`). A structurally-conforming third-party mixer is rejected. The
   protocol file says the opposite in its opening paragraph.
3. **`OpImpl` describes an op nobody calls.** Nothing in `src/` invokes
   `dispatch_op` (`grep -rn "dispatch_op(" src/` → only `ops/registry.py` itself).

So the extension story is: *inheritance plus isinstance*, with a protocol file that
documents an extension story the code does not support. Pick one.

## The typed contract: `ModelInput` / `ModelOutput`

`io.py:14-51` is small and good. Fields are optional and additive
(`input_ids`, `attention_mask`, `patch_ids`, `state`, `inputs_embeds`, `pixel_values`,
`num_masked_tokens`, `temperature`), so a new modality does not break callers.

Problems:

- **`ModelState` is dead.** Declared at `io.py:14-20`, never constructed outside tests.
  Its existence is why `new_state` looks implemented. Either use it for the KV cache
  (`03-defects.md` S1-4) or delete it.
- **`attention_mask` is accepted and dropped** by all three backbones
  (`backbone.py:131, 389, 647`; documented as "API parity" in each docstring). This is
  the single most user-visible contract lie in the repo: `transformers` callers pass
  masks as a matter of course.
- **`inputs_embeds` is a fork in the road, not a variant.** Each backbone re-implements
  "use embeds if given, else embed ids, else raise" (`backbone.py:113-119, 271-275,
  553-558`). Three copies today; the BLT one additionally refuses `patch_ids=None`
  (`backbone.py:556-557`), so the two-family contract already diverges.

## Registry

`registry.py:1-22` is 22 lines and does the right thing: a dict, duplicate registration
raises, `get_model` lists what exists. Two notes:

- Registration happens as an import side effect of `models/__init__.py`. That is fine
  by design, but it means the available model set depends on import order and on whether
  `memory_machines.models` was imported at all. An explicit `register_all()` would make
  the registry deterministic and testable.
- `blocks/__init__.py` eagerly re-exports 77 symbols, including privates
  (`_sinkhorn`, `_window_mask`, `_causal_comp_mask`, `_past_total_len`,
  `_resolve_positions`, `_check_position_bounds`, `_update_swa_pos`). Importing one leaf
  module (`blocks.attention`) loads 1,313 modules and pulls `fast_weights` and `ops`
  (measured, `README.md` ledger). That is the cost of the convenience barrel; split
  public exports from internal ones, or make `blocks/__init__.py` lazy.

## The rank budget is a second pass over the tree

`_apply_rank_budget` (`backbone.py:43`) runs once at the end of each backbone
`__init__`, after `reset_parameters`, and replaces every projection whose role has a
profitable ratio with a `FactoredLinear` (`blocks/low_rank.py:284`). The backbone is
therefore not only a composition root but a rewriter of its own children, which is the
smallest change that covers ~60 construction sites without threading a policy through
every block constructor. The cost: a block built directly (as every block-level test
does) is dense, so block tests and model tests exercise different layer classes; the
guard test (`tests/test_low_rank.py`) is what keeps the two consistent.
The pass is idempotent and never runs from `reset_parameters`, so a reset on an
already-factored model keeps the budget.

## Composition roots: duplicated, not shared

`CheapLLM` and `BLTBackbone` construct the same five pieces (`decoder_in_proj`,
`masked_token_embed`, `up`, `decoder`, `encoder_layers`/`encoder_ffns`) with the same
conditional logic and the same `reset_parameters` ritual
(`backbone.py:230-263` vs `507-543`), then run the same ~40-line masked Perceiver-AR
block (`backbone.py:284-307` vs `566-600`). `CausalEncoderDecoderStack` — the class
that exists precisely to own this composition — delegates to `CausalDecoder` and adds
nothing but `encode` (`ced.py:463-530`).

Consequence: a change to the masked-decode contract (new kwarg, new position rule, new
`seeded_store` semantics) must be made in three places, and only two of them have
tests. `03-defects.md` S3-1 proposes the extraction.

## What is right and should not be "cleaned up"

- `ModelInput`/`ModelOutput` stay dataclasses, not `TypedDict` or `argparse`-style
  bags: they carry defaults and are constructible in doctests.
- Backbones stay ignorant of the loss and of the trainer.
- `blocks/` keeps returning tensors and taking explicit state, rather than growing a
  `configure(...)` or a global context object.

The defects above are about *unenforced* claims, not about the shape of the design.
