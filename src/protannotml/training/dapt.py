"""Domain- and task-adaptive MLM pretraining (DAPT / TAPT) with region-aware masking.

Sketch for PyTorch + Hugging Face transformers. Not executed in the authoring
environment; the masking logic it calls is tested (tests/test_masking.py).

Usage
-----
    python -m protannotml.training.dapt --model facebook/esm2_t33_650M_UR50D \
        --train dapt_train.jsonl --valid dapt_valid.jsonl --out runs/dapt_esm2_650m \
        --cdr3-rate 0.25 --seed 13

Each JSONL line: {"id": ..., "sequence": "EVQL...", "regions": ["FR1", ...]}
(``regions`` may be omitted for uniform masking).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from protannotml.data.masking import MaskingConfig, mask_sequence, region_fixed_rate_probs, uniform_probs

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class RegionAwareCollator:
    """Tokenize, mask with region-aware rates, pad."""

    def __init__(self, tokenizer, fixed_rates: dict[str, float] | None = None,
                 overall_rate: float = 0.15, seed: int = 0):
        self.tok = tokenizer
        self.fixed_rates = fixed_rates or {}
        self.overall_rate = overall_rate
        self.rng = np.random.default_rng(seed)
        self.aa_ids = [tokenizer.convert_tokens_to_ids(a) for a in AMINO_ACIDS]

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        inputs, labels = [], []
        for item in batch:
            enc = self.tok(item["sequence"], return_special_tokens_mask=True)
            ids = np.array(enc["input_ids"])
            special = np.array(enc["special_tokens_mask"], dtype=bool)
            n = int((~special).sum())
            regions = item.get("regions")
            if regions and self.fixed_rates:
                if len(regions) != n:
                    raise ValueError(f"{item.get('id')}: {len(regions)} region labels for {n} residues")
                probs = region_fixed_rate_probs(regions, self.fixed_rates, self.overall_rate)
            else:
                probs = uniform_probs(n, self.overall_rate)
            x, y = mask_sequence(ids, special, probs, self.tok.mask_token_id, self.aa_ids,
                                 MaskingConfig(rate=self.overall_rate), self.rng)
            inputs.append(torch.tensor(x))
            labels.append(torch.tensor(y))
        pad = self.tok.pad_token_id
        x = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad)
        y = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": x, "attention_mask": (x != pad).long(), "labels": y}


def read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    from transformers import (AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pin the exact model commit")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cdr3-rate", type=float, default=None)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForMaskedLM.from_pretrained(args.model, revision=args.revision)
    fixed = {"CDR3": args.cdr3_rate} if args.cdr3_rate else None
    collator = RegionAwareCollator(tok, fixed, args.rate, seed=args.seed)
    train, valid = read_jsonl(args.train), read_jsonl(args.valid)

    targs = TrainingArguments(
        output_dir=args.out, learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        warmup_ratio=0.05, lr_scheduler_type="cosine", weight_decay=0.01,
        eval_strategy="steps", eval_steps=2000, save_steps=2000, logging_steps=100,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        remove_unused_columns=False, seed=args.seed, bf16=torch.cuda.is_available(),
    )
    Trainer(model=model, args=targs, train_dataset=train, eval_dataset=valid,
            data_collator=collator).train()


if __name__ == "__main__":
    main()
