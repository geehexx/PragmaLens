# Local Codex Bootstrap

PragmaLens keeps its heavy Codex control plane outside the product repo.

Local operator overlays such as `AGENTS.md`, `RTK.md`, `.codex/`, and `.agents/`
are expected to be restored from a private sibling control repo and remain
untracked in this working tree.

## Expected layout

- product repo: `PragmaLens`
- private sibling repo: `pragmalens-codex`

Both repos should live under the same parent directory.

## Install the local overlay

From the private control repo, run:

```bash
uv run python scripts/install_local_overlay.py ../PragmaLens
```

That installer restores the local overlay as untracked files and updates local
Git exclude rules so product history stays free of operator-only artifacts.

## Product-repo policy

- Do not track `AGENTS.md`, `RTK.md`, `.codex/`, or `.agents/` here.
- Do not create tracked symlinks into the private repo.
- If the local overlay is missing, product code, tests, packaging, and CI should
  still work without it.
- RTK is an optional local helper for read-only workflows only; product
  verification commands remain `uv`-driven and RTK-independent.
- The default `uv sync` path is intentionally minimal. Use explicit opt-in
  groups for heavier local lanes:
  - `uv sync --group qa` for impacted/local fast feedback
  - `uv sync --group live` for manual live smoke
  - `uv sync --group live --group qa` when both opt-in lanes are needed
