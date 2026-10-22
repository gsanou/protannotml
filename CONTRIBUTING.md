# Contributing

## Conventions

- Code, identifiers, comments and docstrings in English. Tutorials (`docs/tutorials`) are in French.
- Python >= 3.10, type hints on public functions, `ruff` for linting.
- Every numerical rule (IMGT numbering, masking rates, loss formulas, metrics)
  ships with a test that checks it against a documented value or a closed form.
- Cite the source (paper, IMGT page, tool version) in the docstring of any rule
  taken from the literature, and flag anything not documented by IMGT.

## Workflow

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

1. Branch from `main` (`feat/...`, `fix/...`, `docs/...`).
2. Keep commits small; describe *why* in the message.
3. Open a pull request; CI must pass.

## Data and leakage

- Never commit data, model weights or MMseqs2 working files (see `.gitignore`).
- Never touch a sealed test split. New split -> new `split_id` (see `data/README.md`).
- Any pretraining corpus (DAPT/TAPT) must be filtered against all evaluation
  splits with `protannotml.data.partition.filter_pretraining_corpus`.
