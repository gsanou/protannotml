# data/

Nothing in this directory is tracked except this file (see `.gitignore`).
Datasets are rebuilt from versioned sources and described by a manifest.

## Expected layout

```
data/
  raw/                   # downloads, untouched; one sub-folder per source and release
    imgt/<YYYY-MM-DD>/   # IMGT/GENE-DB gapped V-REGIONs, IMGT/3Dstructure-DB, ...
    oas/<release>/
  interim/               # parsed, deduplicated, domain-split sequences
  splits/<split_id>/     # assignment.csv, removed.txt, audit.json, mmseqs_command.txt
  corpora/               # DAPT / TAPT corpora after filtering against evaluation splits
  manifest.json          # source URL, release/date, SHA-256, licence for every raw file
```

## Rules

1. Every raw file is listed in `manifest.json` with its SHA-256 and licence.
2. Split folders are immutable once an evaluation has been run on them;
   a new split gets a new `split_id`.
3. Test splits are sealed: their SHA-256 is recorded in `ROADMAP.md` (phase 0)
   before any model is evaluated on them.

## Data licensing

IMGT data are not redistributed here. The IMGT terms-of-use page states
CC BY-NC-ND 4.0, while a July 2026 report announces a move to CC BY 4.0 from
1 July 2026. Confirm the applicable terms with IMGT before publishing any
derived dataset or trained weights.
