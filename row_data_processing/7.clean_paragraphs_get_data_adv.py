
#!/usr/bin/env python3
import re
import argparse
import unicodedata as ud
from pathlib import Path
from statistics import mean


def unify_spaces(text: str) -> str:
    # Map any unicode whitespace (category Z*, or .isspace()) to ASCII space
    return "".join(" " if ch.isspace() else ch for ch in text)

# ---------- 0) Unicode normalization + invisible/control cleanup ----------

ZERO_WIDTH = {
    "\u200B", "\u200C", "\u200D", "\u200E", "\u200F",
    "\u2060", "\uFEFF"
}
SOFT_HYPHEN = "\u00AD"
NBSP = "\u00A0"

def normalize_unicode(text: str) -> str:
    # NFKC squashes ligatures (ﬁ -> fi), compatibility forms, etc.
    s = ud.normalize("NFKC", text)
    # Convert zero-widths to real spaces so tokens don't stick together
    s = "".join((" " if ch in ZERO_WIDTH else ch) for ch in s)
    # Drop soft hyphen (it's a discretionary hyphen)
    s = s.replace(SOFT_HYPHEN, "")
    # Replace NBSP with plain space
    s = s.replace(NBSP, " ")
    return s

def strip_combining_marks(text: str) -> str:
    # Drop combining diacritics frequently produced by OCR noise
    return "".join(ch for ch in text if ud.category(ch) != "Mn")

def remove_control_chars(text: str) -> str:
    # Remove general Cc/Cf controls while keeping whitespace (handled separately)
    return "".join(ch for ch in text if ud.category(ch) not in {"Cc","Cf"})

# ---------- 1) Strip leading "section indices" at the start of a line ----------
LEADING_LABEL = re.compile(
    r"""^
        \s*
        (?:
            (?:(?:Section|Clause|Article|Part|Schedule|Appendix)\s+)?
            (?:
                \(?\d{1,3}(?:\.\d+)*\)?[.)]?
              | \(?[ivxlcdmIVXLCDM]+\)?[.)]?
              | \(?[A-Za-z]\)?[.)]
            )
            (?:\s*(?:\(?\d{1,3}(?:\.\d+)*\)?[.)]?|\(?[ivxlcdmIVXLCDM]+\)?[.)]?|\(?[A-Za-z]\)?[.)]))*
            (?:\s*[-–—:])?
        )
        (?=\s|$)
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

# ---------- 2) Remove dot/bullet leaders and long dash/underscore runs ----------
LEADER_RUN = re.compile(
    r"""[.\u2024\u2027\u2219\u00B7\u2022\u25CF\u25E6·•∙\u2043_\-\u2013\u2014]{3,}""",
    re.UNICODE,
)

def remove_leader_runs(text: str) -> str:
    return LEADER_RUN.sub("", text)

# ---------- 3) Remove specific unwanted characters globally ----------
# Added: en dash, more bullet shapes, ellipsis, middle dots, black circle, white bullet, small squares, etc.
UNWANTED_CHARS = re.compile(
    r"""[()"@#*`~^|<>\[\]/\\
        \u201C\u201D\u2018\u2019\u2013\u2014
        \u2022\u2027\u00B7\u2219\u2043\u2026
        \u25CF\u25E6\u25AA\u25AB\u25A0
    ]""",
    re.VERBOSE,
)

def remove_unwanted_chars(text: str) -> str:
    return UNWANTED_CHARS.sub("", text)

# ---------- 4) Remove explicit tag [DSA] (case-insensitive) ----------
DSA_TAG = re.compile(r"\[DSA\]", re.IGNORECASE)

def remove_dsa_tag(text: str) -> str:
    return DSA_TAG.sub("", text)

# ---------- Tokenizer ----------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)*")

def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


PUNCT_AFTER = re.compile(r'([;,:\.])([A-Za-z0-9])')

def fix_punct_spacing(text: str) -> str:
    return PUNCT_AFTER.sub(r'\1 \2', text)


def clean_line(raw: str) -> str:
    line = raw.rstrip()
    if not line:
        return ""
    # Unicode passes
    line = normalize_unicode(line)
    line = unify_spaces(line)
    line = strip_combining_marks(line)
    line = remove_control_chars(line)

    # Structural cleanups
    line = strip_leading_indices(line)
    line = remove_leader_runs(line)
    line = remove_dsa_tag(line)
    line = remove_unwanted_chars(line)
    line = fix_punct_spacing(line)

    # Collapse whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line

def process_dir(in_dir: Path, min_tokens: int) -> list[str]:
    out_lines: list[str] = []
    short_lines: list[str] = []
    files = sorted(in_dir.glob("*.txt"))
    for p in files:
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = clean_line(raw)
            if not line:
                continue
            if "?" in line:
                continue
            if count_tokens(line) < min_tokens:
                short_lines.append(line)
                continue
            out_lines.append(line)
    return out_lines, short_lines

def compute_stats(lines: list[str]) -> dict:
    if not lines:
        return {"lines": 0, "min_tokens": 0, "max_tokens": 0, "avg_tokens": 0.0}
    lens = [count_tokens(s) for s in lines]
    return {
        "lines": len(lines),
        "min_tokens": min(lens),
        "max_tokens": max(lens),
        "avg_tokens": float(f"{mean(lens):.2f}"),
    }

def main():
    ap = argparse.ArgumentParser(description="Merge cleaned paragraphs into one file (token threshold + sanitization).")
    ap.add_argument("-i", "--input_dir", default="paragraphs_cleaned", help="Folder containing the cleaned part files")
    ap.add_argument("-o", "--output", default="data_sharing_agreement_mt20.txt", help="Output file path")
    ap.add_argument("--min-tokens", type=int, default=20, help="Drop lines with fewer than this many tokens after cleaning")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")

    lines, short_lines = process_dir(in_dir, args.min_tokens)

    # Path(args.output).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    Path(args.output+"removed.txt").write_text("\n".join(short_lines) + ("\n" if short_lines else ""), encoding="utf-8")

    # stats = compute_stats(lines)
    # print(f"Done. Files: {len(list(in_dir.glob('*.txt')))} | Lines kept: {stats['lines']}")
    # print(f"Wrote: {args.output}")
    # print(f"Token stats -> min: {stats['min_tokens']} | max: {stats['max_tokens']} | avg: {stats['avg_tokens']}")

if __name__ == "__main__":
    main()
