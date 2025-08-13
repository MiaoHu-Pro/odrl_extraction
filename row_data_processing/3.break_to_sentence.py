#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import sys

PROTECT_DOT = "§"

ABBR = [
    # common abbreviations that end with a period
    "e.g.", "i.e.", "etc.", "vs.", "cf.",
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Sr.", "Jr.",
    "Fig.", "Ch.", "pp.", "para.", "paras.", "No.", "Nos.", "Art.", "Sec.", "Pt.", "Sch.", "Tab.", "Tbl.",
    # months
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."
]

def normalize(text: str) -> str:
    # Convert bullets and weird separators to clean sentence boundaries
    text = text.replace("●", ". ").replace("•", ". ").replace(" – ", " - ")
    text = re.sub(r"\s+\.\s+", ". ", text)            # " . " → ". "
    text = re.sub(r"\s*([.?!])\s*", r"\1 ", text)     # tidy spaces around .?!
    text = re.sub(r"\s+", " ", text).strip()
    return text

def protect(text: str) -> str:
    # 1) protect known abbreviations (swap inner dots)
    for abbr in ABBR:
        text = text.replace(abbr, abbr.replace(".", PROTECT_DOT))
    # 2) protect single-letter legal clauses: S.20, s.20, r.5, etc.
    text = re.sub(r"\b([A-Za-z])\.\s*(?=\d)", rf"\1{PROTECT_DOT}", text)
    # 3) protect digit-dot-digit (handles 5.1.2 and '6. 1')
    text = re.sub(r"(?<=\d)\.\s*(?=\d)", PROTECT_DOT, text)
    return text

def unprotect(text: str) -> str:
    return text.replace(PROTECT_DOT, ".")

def split_sentences(line: str) -> list[str]:
    line = normalize(line)
    if not re.search(r"[.?!]", line):
        return [line] if line else []
    safe = protect(line)
    parts = re.split(r"(?<=[.?!])\s+", safe)  # split on .?! regardless of next char
    return [unprotect(p).strip() for p in parts if p.strip()]

def main():
    ap = argparse.ArgumentParser(description="Split each input line into sentences and write to TXT.")
    ap.add_argument("-i", "--input", default="agreement_text_row_clean.txt",
                    help="Path to input .txt (default: agreement_text_row_clean.txt)")
    ap.add_argument("-o", "--output", default="agreement_text_row_clean_sentences.txt",
                    help="Path to output .txt (default: agreement_text_row_clean_sentences.txt)")
    ap.add_argument("--mode", choices=["plain", "tsv"], default="plain",
                    help="plain: one sentence per line; tsv: line_no\\tsent_no\\tsentence")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    lines = in_path.read_text(encoding="utf-8").splitlines()

    out_lines = []
    if args.mode == "tsv":
        out_lines.append("line_no\tsent_no\tsentence")

    for i, line in enumerate(lines, 1):
        sents = split_sentences(line)
        if args.mode == "plain":
            out_lines.extend(sents)
        else:
            for j, s in enumerate(sents, 1):
                out_lines.append("{}\t{}\t{}".format(i, j, s.replace("\t", " ").strip()))



    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Done. Wrote {len(out_lines)} lines to {out_path}")

if __name__ == "__main__":
    main()
