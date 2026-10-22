# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-22

### Added
- `imgt`: IMGT positions with sequence-order keys, loop gap/insertion rule
  (CDR1/2/3-IMGT, C-domain FG loop), V/C/G region maps, framework templates,
  V-domain assembly, position vocabulary, gene-name parser.
- `data.partition`: MMseqs2 all-vs-all wrapper, connected components,
  GraphPart-style bridge removal, biological hold-outs, leakage audit,
  novelty tiers, pretraining-corpus filtering.
- `data.masking`: uniform, region-preferential and span masking; 80/10/10
  corruption; temperature sampling across sources.
- `training.weighting_ref`, `decoding.monotonic`: loss-weighting reference
  formulas and O(L*K) constrained numbering decoder.
- `models.mtl`, `training.dapt`: PyTorch multi-task model and DAPT/TAPT script
  (syntax-checked only, not yet executed).
- `eval`: numbering, boundary and assignment metrics; cluster bootstrap,
  paired bootstrap, exact McNemar, Holm.
- Tutorials 1-6 (French) under `docs/tutorials`.
