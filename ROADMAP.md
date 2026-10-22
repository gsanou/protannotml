# Roadmap

Phased plan with explicit go/no-go criteria. Each phase should end with a
shareable result (preprint, poster or internal report). Status markers:
`[ ]` to do, `[~]` in progress, `[x]` done.

## Phase 0 - Foundations

- [x] Reference code: IMGT numbering rules, partitioning, masking, loss weighting,
      monotonic decoding, metrics and statistics (`src/protannotml`, 43 tests).
- [x] Tutorial series, modules 1-6 (`docs/tutorials`).
- [ ] Confirm IMGT licensing terms in writing (CC BY-NC-ND 4.0 vs CC BY 4.0).
- [ ] Contact the IMGT team: bulk access to DomainGapAlign, curated ground truth,
      possible collaboration.
- [ ] Validate the CDR1/CDR2 gap rule against a few hundred DomainGapAlign outputs
      (two IMGT pages disagree).
- [ ] Run the PyTorch modules (`models/mtl.py`, `training/dapt.py`) on a toy batch.
- [ ] Data manifest: versioned IMGT extraction with SHA-256 for every file.
- [ ] Pre-registered evaluation protocol: test layers, metrics, statistics;
      seal test splits (publish SHA-256 of the assignment files).

## Phase 1 - IG V domains, numbering and regions

Scope: VH, V-KAPPA, V-LAMBDA; heads = region, IMGT position, domain type.

- [ ] Real MMseqs2 all-vs-all; giant-component curve vs threshold (domain and FR-only identity).
- [ ] Layered splits (temporal, species, gene, components) and leakage audit.
- [ ] Negative-control set: non-IgSF proteins, Ig-like folds (FN3, cadherins), IgSF V-LIKE.
- [ ] Base-model ablation (ESM-2, ESM C, an antibody LM, random init) with a linear probe.
- [ ] DAPT; DAPT x TAPT 2x2 factorial (drop TAPT if no gain).
- [ ] Baselines: DomainGapAlign, ANARCI, AbNumber, ANARCII, IgBLAST (protein).

Go criterion: match or beat ANARCI on exact-match numbering on the component
layer, with cluster-bootstrap CIs; report every test layer.

## Phase 2 - TR, gene and species

- [ ] Add V-ALPHA, V-BETA, V-GAMMA, V-DELTA; gene and species heads.
- [ ] Gene head as embedding + nearest-neighbour search over IMGT references
      (open set: new alleles/species without retraining).
- [ ] Conformal prediction sets for gene/allele with guaranteed coverage.
- [ ] Hybrid allele call: model proposes candidates, exact alignment decides.

Go criterion: calibrated gene assignment on held-out genes and species
(ECE and coverage-accuracy curves).

## Phase 3 - C and G domains, multi-domain chains

- [ ] C-DOMAIN / C-LIKE numbering (45.x, 84.x/85.x, 96.x insertions).
- [ ] G-DOMAIN numbering (letter insertions 7A, 54A, 61A, 61B, 72A, 92A).
- [ ] Domain delimitation on full chains and constructs (scFv, bispecifics).

## Phase 4 - Robustness and independent ground truth

- [ ] Controlled perturbation curves: simulated SHM (hotspot model), truncations, indels.
- [ ] Structural check on IMGT/3Dstructure-DB: same IMGT number -> same 3D location
      after superposition (C-alpha distances).
- [ ] "Disagreement" set: sequences where tools diverge, annotated independently.
- [ ] Interpretability: do representations encode anchors 23, 41, 104, 118?

## Phase 5 - Release and publication

- [ ] Distilled CPU model; throughput table vs ANARCI/ANARCII.
- [ ] Public web server (no login), API, pip package, Docker image.
- [ ] Zenodo DOIs for code, weights and splits; model card with limitations.
- [ ] Target: NAR (Web Server issue) or Briefings in Bioinformatics for the tool;
      Genome Biology only with a biological application, e.g. annotating IG/TR loci
      in newly sequenced vertebrate genomes or finding unannotated IgSF domains.

## Out of scope (for now)

- Nucleotide input (covered by IMGT/V-QUEST and IgBLAST).
