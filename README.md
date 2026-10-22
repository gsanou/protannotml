# ProtAnnotML

[![CI](https://github.com/OWNER/protannotml/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/protannotml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reference code for **ProtAnnotML**, a machine-learning complement and successor to
IMGT/DomainGapAlign: domain delimitation, IMGT unique numbering with gaps,
FR/CDR annotation, closest gene/allele and species.

This package accompanies a six-module tutorial series (French, in
[`docs/tutorials`](docs/tutorials/index.html)) and implements the deterministic
and statistical parts of the pipeline. Identifiers, variables and comments are
in English. The phased plan is in [ROADMAP.md](ROADMAP.md).

| Tutorial | Topic |
|---|---|
| 1 | Immunogenetics foundations: IgSF domains, IG/TR/MH, V(D)J, IMGT nomenclature |
| 2 | IMGT unique numbering and its exceptions (V, C, G domains) |
| 3 | Leakage-resistant partitioning with MMseqs2 |
| 4 | Protein language models and MLM: DAPT then TAPT |
| 5 | Multi-task learning: weighted and homoscedastic-uncertainty losses, monotonic decoding |
| 6 | Evaluation: metrics, cluster bootstrap, paired tests, baselines |

Open `docs/tutorials/index.html` locally, or enable GitHub Pages
(`.github/workflows/docs.yml` deploys the folder on every push to `main`).

## Repository layout

```
docs/tutorials/       HTML tutorials 1-6 (French), self-contained, deployable to GitHub Pages
data/                 untracked; data/README.md documents layout, manifest and sealing rules
.github/workflows/    CI (ruff, pytest on 3.10-3.13, torch import smoke test) and Pages
src/protannotml/
  imgt/
    numbering.py      IMGT positions (111.1, 112.1, 61A...), sequence order, loop gap/insertion
                      rule, V/C/G region maps, framework templates, V-domain assembly
    vocab.py          position vocabulary sorted in sequence order (UNNUMBERED + positions)
    nomenclature.py   IMGT gene/allele name parsing into a label hierarchy
  data/
    partition.py      MMseqs2 all-vs-all wrapper, connected components, GraphPart-style
                      bridge removal, biological hold-outs, leakage audit, novelty tiers,
                      DAPT/TAPT corpus filtering
    masking.py        uniform / region-preferential / span masking, 80-10-10 corruption,
                      temperature sampling across sources
  training/
    weighting_ref.py  NumPy reference for weighted and homoscedastic-uncertainty losses
    dapt.py           DAPT/TAPT script (PyTorch + transformers)         [not executed]
  models/
    mtl.py            multi-task model, set-valued loss, uncertainty weighting [not executed]
  decoding/
    monotonic.py      O(L*K) constrained decoding: strictly increasing IMGT positions
  eval/
    metrics.py        position accuracy, exact match, boundary F1, set accuracy, ECE
    stats.py          cluster bootstrap CI, paired bootstrap, exact McNemar, Holm
tests/                pytest test suite (43 tests)
tools/run_tests.py    minimal runner for environments without pytest
```

## Install and test

```bash
pip install -e ".[test]"          # numpy, scipy, networkx, pytest
pytest                            # or: python tools/run_tests.py
pip install -e ".[torch]"         # for models/ and training/dapt.py
pip install -e ".[dev]"           # pytest + ruff
```

`data.partition.mmseqs_all_vs_all` needs the `mmseqs` binary (tested option names
against MMseqs2 release 18, tag `18-8cc5c`).

## Status and caveats

* All NumPy/Python logic is covered by tests. The PyTorch modules
  (`models/mtl.py`, `training/dapt.py`) were only syntax-checked: the authoring
  environment had no access to the Python package index. Run them on a small
  batch before any real experiment.
* IMGT documents the CDR3-IMGT / FG-loop insertion order (112.1, 111.1, 112.2...),
  but not insertions for CDR1-IMGT > 12 or CDR2-IMGT > 10. Those follow the
  ANARCI symmetric convention and are flagged (`LoopNumbering.extrapolated`).
  Two IMGT pages also disagree on CDR1/CDR2 gap placement; validate the rule
  against DomainGapAlign outputs before freezing labels.
* Framework gaps are gene-specific. Build `FrameworkTemplate` objects from
  IMGT-gapped germline V-REGIONs (IMGT/GENE-DB), never from hard-coded lists.
* IMGT data licensing: the IMGT terms-of-use page states CC BY-NC-ND 4.0, while a
  July 2026 report announces CC BY 4.0 from 1 July 2026. Confirm with IMGT before
  redistributing derived data or trained weights.

## Contributing, citation, licence

* Contribution rules (conventions, leakage safeguards): [CONTRIBUTING.md](CONTRIBUTING.md)
* How to cite: [CITATION.cff](CITATION.cff)
* Code and tutorials: MIT ([LICENSE](LICENSE)). IMGT data are **not** covered and
  are not redistributed; see [data/README.md](data/README.md).

## References

Lefranc et al. (2003) Dev Comp Immunol 27:55-77; Lefranc et al. (2005) Dev Comp
Immunol 29:185-203 and 29:917-938; Kendall, Gal & Cipolla (2018) CVPR;
Liebel & Koerner (2018) arXiv:1805.06334; Ng & Briney (2025) Patterns 6:101239;
Steinegger & Soeding (2017) Nat Biotechnol 35:1026; Teufel et al. (2023) GraphPart;
Hermann et al. (2024) PMLR 261; Dunbar & Deane (2016) Bioinformatics 32:298.
