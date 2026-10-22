# 01 — The architecture as a scientific claim

## The argument, not the inventory

Stripped of module names, `models/` makes one argument:

> **A cheap language model should not be one deep stack of one mixer. It should be a
> two-stage compressor: an encoder that turns tokens into a compressed global memory, and
> a decoder that reads that memory. Inside the encoder, every layer is a *parallel* pair of
> (local exact attention) ∥ (global associative memory). The global store is *selected
> sparsely* and *shared across depth* rather than rebuilt per layer.**

Everything in the repository is an instance of four commitments.

### C1 — Asymmetric encode/decode with an explicit information bottleneck (Eq. 1)

The encoder writes a compressed KV store (`CEDGlobalKVProjection`). The decoder never
rebuilds it: every decoder layer reads the same tensor produced by a single projection of
the encoder's final states, and is itself narrow (`ced_decoder_hidden_size`), then widened
again by `VocabularyUpProjection` in front of the softmax.

Scientific content: this is the only decision in the repository that is a genuine
architectural *thesis* rather than a mechanism. It asserts that long-range context is a
fixed-size artifact produced once, so decode-time compute is decoupled from context length
in the strongest available sense — not "sparse attention over the past" but "read a summary
someone else wrote." The correct neighbours are retrieval/compression work (gist tokens,
ICAE, MemGPT-style memory), not the efficient-attention literature. The failure modes are
therefore the compression literature's failure modes: the summary is a bottleneck, it is
lossy in ways training does not penalize directly, and it is read by a query distribution
the writer never saw.

### C2 — Parallel hybrid: exact-local ∥ approximate-global, superposed

`ParallelAttentionGDN2` computes `h = norm(x)`, then
`Attn_swa(h) ∥ Mem_global(h)`, concatenates the two d-wide results, projects 2d→d, and adds
one residual. The attention branch is windowed; the memory branch is never windowed.

Scientific content: this is the Jamba/Griffin/Zamba hybrid idea with the crucial twist that
the two mixers run **in parallel on the same normalized input**. Sequential hybrids divide
labour through depth (this layer local, that layer global). Parallel hybrids divide labour
through *additive superposition at the same depth*, which is the stronger claim: that the
two operators are complementary functionals of the same representation rather than stages
of a pipeline. The asymmetric initialisation (attention half at `init_range`, memory half at
`1e-3`) is the honest admission that the superposition has to be discovered from an
attention-only start.

### C3 — Compression as *selection*, not just pooling

CSA2 does two things that most compressed-attention schemes conflate:

1. **Pooling** — `BlockCompressor` learns a softmax over each block of `m` tokens
   (`score_proj`, `block_bias`), so the summary is a learned weighted mean, not an average.
2. **Selection** — `SparseIndexer` holds its own low-rank query, its own index keys, a
   hierarchical block-max prefilter, and a top-k support that is *cached* and *reused*.

Scientific content: the claim is that pooling and retrieval should be separate operators
with separate parameters, because pooling is a lossy summary (must preserve everything a
little) while selection is a routing decision (must be sharp about a few things). Schemes
that fuse them force one set of weights to be simultaneously blurry and sharp. This
separation is the most defensible novel idea in the repository.

### C4 — State as a first-class shared object

`CSA2SharedState(compressed_kv, comp_end_pos, indexer_k, topk_indices, pool_indices,
total_len)` is threaded through encoder layers and into the decoder. The modes
`full` / `reindex` / `reuse` describe *who is allowed to rewrite the store*:

| Stage | Mode |
| --- | --- |
| Encoder layer 0 | dense SWA attention, no store |
| Encoder layer 1 | `full` (compress + select) |
| Encoder layers 2–19 | `reuse` (inherit layer 1's support) |
| Decoder layer 0 | `full` (seeded from the encoder store) |
| Decoder layers 1, 5, 9, 13, 17 | `reindex` (rescore the cached pool) |
| Remaining decoder layers | `reuse` |

Scientific content: this is the repository's real theoretical bet, and its cost model is
concrete. Only 1 of 20 encoder layers pays for compression *and* selection; only 5 of 20
decoder layers pay for re-selection. A `reuse` layer costs an attention over at most
`csa_top_k` cached indices and no indexer work. Stated as a claim:

> **The top-k support of a compressed memory is stable enough across depth that it can be
> computed once and inherited many times.**

That is falsifiable, cheap to test, and — as far as the efficient-attention literature
shows — not established anywhere. It is also the claim most likely to be wrong, which is
what makes it worth testing first.

## What the default configuration actually instantiates

- `hidden_size=1024`, `num_layers=40` split 20 encoder + 20 decoder.
- Encoder layer 0: dense GQA with window 256. Layer 1: CSA2 `full`.
  Layers 2–19: CSA2 `reuse`. Odd layers replace GDN-2 with Titans
  (`ced_encoder_alternating=True`).
- Decoder: one shared global store from `global_projs[0]`; 20 layers, 5 reindex points.
- On by default and never ablated: `cvpe_enabled=True`, `frvm_enabled=True`,
  `csa_use_sink=True`, `residual_init_scale=True`, `ced_encoder_norm=dyT`,
  `ced_encoder_ffn_norm=dyT`, `swa_bounded_replay=True`, `tie_word_embeddings=True`.
- Off by default: `ced_decoder_looping` (LoopLM), `blt_latent_sampling`, GDN-2 resting
  state, `apply_beta_guard`.

The default is therefore a *stack of nine simultaneous hypotheses*. Any perplexity number
it produces supports none of them individually. This is the central scientific problem of
the repository, and it is orthogonal to whether the ideas are good.

## What is not claimed but is implied

Three implications follow from C1–C4 that the code does not state anywhere, and that a
reviewer will ask about:

1. **The compressed store is auto-associative.** `BlockCompressor` produces one vector per
   block and uses it as both key and value. The store can therefore only represent
   "look up by content X, return X" — not "look up by X, return Y." For a memory the
   decoder reads with its own queries, this is a strong restriction: decoder queries must
   live in the encoder's key subspace.
2. **Position is not represented, only prior-ed.** There is no RoPE and no absolute
   embedding anywhere; position enters only as a learned scalar bias on attention logits
   (CVPE) and as masks. Consequently positional information cannot enter the residual
   stream directly, cannot be transformed by FFNs, and cannot be used multiplicatively by
   later layers.
3. **The decoder cannot compress its own reasoning.** Because the store is seeded from
   encoder-final states, twenty decoder layers re-read one fixed summary twenty times. All
   decoder-internal long-range structure must live in the residual stream.

Each of these is a legitimate design choice. None is stated, and none is tested.
