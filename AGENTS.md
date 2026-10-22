# AGENTS.md

## What this is

Research codebase for a cheap LLM: a causal encoder-decoder transformer built on **CSA2** with auxiliary memory mechanisms (GDN-2, Titans, mHC, Engram, LIV). Alpha project: rough edges are expected, the default config is a starting point.

Three model families are registered by name (`models/registry.py`): `example` (smoke test), `cheapllm` (CED encoder-decoder), `blt` (byte-level). `MemoryMachineConfig` (`utils/config_schema.py`) is the single config interface; it validates many invariants at boot, but not all — see `docs/reviews/code/03-defects.md` S2-12.

## Reviews — `docs/reviews/` (read first, update last)

Two layers, both source of truth:

- `docs/reviews/*.md` — **science**: claims, strengths, defects S1–S17, mechanism algebra, experiments.
- `docs/reviews/code/*.md` — **engineering**: contracts, state, ops, numerics, cost, remediation order.

Read `docs/reviews/README.md` at the start of a session; it indexes both layers. Before changing anything under `src/`, read `code/03-defects.md`. Before changing block math, read `04-mechanism-audit.md`.

Rules:

- Never contradict a review finding without a measurement that disproves it; then update the review.
- Update reviews **only at the very end of the session**, after the last change and the last test run.
- Any change under `src/` requires a `docs/reviews/code/` update: mark resolved findings `resolved` (never delete), add new ones as the next free `S<n>` with `file:line`, correct stale references.
- Reviews are not changelogs. Change the claims that are now wrong; leave the rest untouched.

## `AGENTS.md` self-updates

This file is operational memory, maintained by the agent at the same end-of-session point. Reviews hold claims and evidence; this file holds only what the next session would otherwise rediscover.

- Add: a command/flag/hook that fails or silently does nothing plus its working alternative; a renamed path or key; a config trap; an ordering constraint; a decision with a reason.
- Fix immediately any line this change makes false.
- Do not add: session narratives, plans, review findings, or anything one `grep`/`ls` answers.

## Commands

- Sync: `uv sync --group dev`. On a GPU host add `--no-sources-package torch --no-sources-package torchvision --no-sources-package torchaudio --index https://download.pytorch.org/whl/cu128` — list all three, and use `--index`, never `--default-index`. `uv lock` without the override restores the marker-split lock; never commit a lock written under it.
- Tests: `uv run --no-sync pytest`. Doctests do **not** run there (`testpaths=["tests"]` overrides `--doctest-modules`); run `pytest --doctest-modules src/memory_machines/models/blocks src/memory_machines/utils`.
- Gates: `ruff check --output-format concise`, `ruff format --check`, `pyrefly check`, `aislop scan .` (all via `uv run --no-sync`). aislop complexity warnings are intentionally left — do not disable rules to pass.
- Pre-commit: `uv run --no-sync pre-commit run --all-files`. There is no `.git/hooks/pre-commit`, and the pyrefly hook env cannot install — run `pyrefly check` manually.
- Train/eval/export: `uv run python scripts/train.py` (add `trainer.fast_dev_run=true` for one step), `scripts/eval.py checkpoint=...`, `scripts/export_hf.py --checkpoint ... --output-dir ...`. Default data is synthetic; `configs/model/default.yaml` selects `ExampleBackbone`.
- Build: `uv build`. Library plus scripts; no Docker, no CI beyond pre-commit.

## Traps (currently true)

- `window_size` means compressed **blocks** for CSA variants but raw tokens for `dense`; `token_window_size = window_size * compression_ratio`, so default `window_size_compressed=1024` covers 4096 tokens (`code/03` S3-12).
- No KV cache: backbones return `new_state=None` while the HF adapter accepts `past_key_values`. `generate()` is not incremental (`code/03` S1-4).
- Bool masks are inverted: `models/blocks/attention.py` treats `True` as *blocked*; `ops/references/*` treats it as *participate* (`code/03` S2-4).
- GDN-2 `naive` (the default) runs recurrent and ignores `ced_gdn2_chunk_size`; `chunk` NaNs when a chunk's cumulative decay underflows (`code/05`).
- The `ops/` dispatcher is orphaned and `kernels/` is empty — read `code/04` before touching either.

## Conventions tooling does not enforce

- Docstrings: PyTorch style (signature line first, then `Args:`/`Returns:`/`Raises:`), one-line imperative summary.
- Tests mirror source layout; `tmp_path` for files; fakes only at true externals; `pytest.raises` and `parametrize(ids=...)`.
- Fixes show red-green; refactors keep tests unchanged; one composition root; no import-time side effects.
- Never hardcode, print, commit, or transmit secrets.

