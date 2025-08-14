
#!/usr/bin/env python3
import argparse
import pathlib
import random

def main():
    ap = argparse.ArgumentParser(description="Split a .jsonl file into train/valid/test (8:1:1).")
    ap.add_argument("input", type=pathlib.Path, help="Path to the .jsonl file")
    ap.add_argument("--outdir", type=pathlib.Path, default=None, help="Where to write outputs (default: input's folder)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = ap.parse_args()

    outdir = args.outdir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Read non-empty lines
    with args.input.open("r", encoding=args.encoding) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        raise SystemExit("No lines found in input.")

    # Shuffle deterministically
    # removed args.seed and just did:
    # random.shuffle(lines), get a different train/valid/test split each run.
    rng = random.Random(args.seed) # creates its own independent random number generator
    rng.shuffle(lines) # reorders the list lines in place using that RNG.

    # 8:1:1 split
    n = len(lines)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    n_test = n - n_train - n_valid

    train = lines[:n_train]
    valid = lines[n_train:n_train + n_valid]
    test  = lines[n_train + n_valid:]

    # Write out
    (outdir / "train.txt").write_text("\n".join(train) + "\n", encoding=args.encoding)
    (outdir / "valid.txt").write_text("\n".join(valid) + "\n", encoding=args.encoding)
    (outdir / "test.txt").write_text("\n".join(test) + "\n", encoding=args.encoding)

    print(f"Wrote {len(train)} train, {len(valid)} valid, {len(test)} test to {outdir}")

if __name__ == "__main__":

    # python split_jsonl.py data_sharing_agreement_mt20_labels_o4-mini-more-tags-seeds.jsonl
    # # optional:
    # # python split_jsonl.py your.jsonl --outdir splits --seed 123
    main()
