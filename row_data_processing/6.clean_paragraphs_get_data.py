#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

# --- 1) Strip leading "section indices" at the start of a line (safe) ---
LEADING_LABEL = re.compile(
    r"""^
        \s*
        (?:
            (?:(?:Section|Clause|Article|Part|Schedule|Appendix)\s+)?   # optional keyword
            (?:                                                        # one enumerator
                \(?\d{1,3}(?:\.\d+)*\)?[.)]?                           # 5.2  1.2.3  5)  5.
              | \(?[ivxlcdmIVXLCDM]+\)?[.)]?                           # (i)  ii)  II.
              | \(?[A-Za-z]\)?[.)]                                     # (a)  a)  A)  A.
            )
            (?:\s*(?:\(?\d{1,3}(?:\.\d+)*\)?[.)]?                      # optional chained enumerators
               |\(?[ivxlcdmIVXLCDM]+\)?[.)]?
               |\(?[A-Za-z]\)?[.)]))*
            (?:\s*[-–—:])?                                             # optional tail punctuation
        )
        (?=\s|$)                                                       # must be followed by sep/EOL
        \s*
    """,
    re.VERBOSE,
)

def strip_leading_indices(text: str) -> str:
    s = text
    prev = None
    while s != prev:
        prev = s
        m = LEADING_LABEL.match(s)
        if not m:
            break
        s = s[m.end():]
    return s.lstrip()

# --- 2) Remove dot-leader-like sequences ANYWHERE (3+ repeated leader chars) ---
LEADER_RUN = re.compile(
    r"""[.\u2024\u2027\u2219·•_\-\u2013\u2014]{3,}""",  # dots, middots, bullets, underscores, en/em dashes
    re.UNICODE,
)

def remove_leader_runs(text: str) -> str:
    return LEADER_RUN.sub("", text)

# --- 3) Remove specific unwanted characters globally ---
#   (, ), ", “, ”, ‘, ’, /, — (em dash), • (bullet)
UNWANTED_CHARS = re.compile(r"""[()"/\u201C\u201D\u2018\u2019\u2014\u2022]""")

def remove_unwanted_chars(text: str) -> str:
    return UNWANTED_CHARS.sub("", text)

# --- 4) Remove explicit tag [DSA] (case-insensitive) ---
DSA_TAG = re.compile(r"\[DSA\]", re.IGNORECASE)

def remove_dsa_tag(text: str) -> str:
    return DSA_TAG.sub("", text)

# --- Tokenizer: capture words and common compounds after cleaning ---
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)*")

def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))

def clean_line(raw: str) -> str:
    line = raw.rstrip()
    if not line:
        return ""
    # order matters: use parentheses for index parsing before we remove them
    line = strip_leading_indices(line)
    # now remove leader runs anywhere (e.g., ".... 4", "———", "_____")
    line = remove_leader_runs(line)
    # remove [DSA]
    line = remove_dsa_tag(line)
    # remove unwanted characters
    line = remove_unwanted_chars(line)
    # collapse whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line

def process_dir(in_dir: Path, min_tokens: int) -> list[str]:
    out_lines: list[str] = []
    removed_lines: list[str] = []
    files = sorted(in_dir.glob("*.txt"))
    for p in files:
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = clean_line(raw)
            if not line:
                continue
            # NEW: drop any line that contains a question mark (e.g., "...?," or ends with "?")
            if "?" in line:
                continue
            if count_tokens(line) < min_tokens:
                removed_lines.append(line)
                continue

            out_lines.append(line)
    return out_lines, removed_lines

def main():
    ap = argparse.ArgumentParser(description="Merge cleaned paragraphs into one file (token threshold + sanitization).")
    ap.add_argument("-i", "--input_dir", default="paragraphs_cleaned",
                    help="Folder containing the cleaned part files")
    ap.add_argument("-o", "--output", default="data_sharing_agreement_mt20.txt",
                    help="Output file path")
    ap.add_argument("--min-tokens", type=int, default=20,
                    help="Drop lines with fewer than this many tokens after cleaning")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    lines, removed_lines = process_dir(in_dir, args.min_tokens)
    Path(args.output+"removed.txt").write_text("\n".join(removed_lines) + "\n", encoding="utf-8")
    # Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Done. Files: {len(list(in_dir.glob('*.txt')))} | Lines kept: {len(lines)}")
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
