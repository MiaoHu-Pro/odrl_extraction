#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

# ---------- heuristics ----------

INDEX_WORDS = re.compile(r"\b(index|contents|table of contents)\b", re.I)
TITLE_WORDS = re.compile(
    r"\b(data sharing agreement|controller to controller|agreement for the sharing of data|schedule|appendix|annex)\b",
    re.I
)

TOC_ITEM_RE = re.compile(
    r"""^
        \s*(?:                                  # numbering
            \d+(?:\.\d+)*[.)]?                  # 1. 1.2 1.2.3 1)
            |[ivxlcdm]+[.)]                     # i) iv)
            |[A-Za-z][.)]                       # A) a)
        )
        \s+[A-Z][A-Za-z0-9’'\-]*(?:\s+[A-Za-z0-9’'\-]+){0,7}\s*$  # short section title-ish
    """,
    re.VERBOSE
)

ALL_CAPS_LINE = re.compile(r"^[A-Z0-9 ,.&/()'’\-:]+$")

# lines that are just entity/title blocks
TITLE_BLOCK_CUES = re.compile(
    r"\b(between|and|of|for|agreement|controller|controllers?)\b", re.I
)

# section name single-word or very short
SECTION_NAME_WORDS = {
    "BACKGROUND","PURPOSE","INTRODUCTION","DEFINITIONS","INTERPRETATION","SCOPE",
    "PARTIES","RECITALS","INDEX","CONTENTS","GLOSSARY"
}

def is_all_caps_short(line: str, max_words=8) -> bool:
    s = line.strip()
    if not s or len(s) < 3: return False
    if not ALL_CAPS_LINE.match(s): return False
    words = len(s.split())
    return words <= max_words

def token_count(line: str) -> int:
    return len(re.findall(r"\b\w+\b", line))

def looks_like_toc_item(line: str) -> bool:
    return bool(TOC_ITEM_RE.match(line)) or bool(INDEX_WORDS.search(line))

def looks_like_title_line(line: str) -> bool:
    s = line.strip()
    if TITLE_WORDS.search(s): return True
    # a title block line tends to be all caps and short with prepositions like BETWEEN/AND/OF
    return is_all_caps_short(s, 12) and TITLE_BLOCK_CUES.search(s) is not None

def is_section_name_only(line: str) -> bool:
    s = line.strip().strip(":")
    if s.upper() in SECTION_NAME_WORDS: return True
    # “1. INTERPRETATION” style headings — drop if it’s short and no period at end
    if (re.match(r"^\s*(\d+(\.\d+)*)[.)]?\s+[A-Z][A-Z \-]+$", s) and s == s.upper()):
        return True
    # naked all-caps short line
    return is_all_caps_short(s, 6)

def clean_file(inp: Path, out: Path, keep_reason=False):
    lines = inp.read_text(encoding="utf-8", errors="ignore").splitlines()

    cleaned = []
    reasons = []

    # detect and drop TOC blocks (5+ consecutive toc-looking lines)
    toc_mask = [looks_like_toc_item(l) for l in lines]
    drop_idx = set()
    run_start = None
    for i, is_toc in enumerate(toc_mask + [False]):  # sentinel false to flush
        if is_toc and run_start is None:
            run_start = i
        if not is_toc and run_start is not None:
            run_len = i - run_start
            if run_len >= 5:
                drop_idx.update(range(run_start, i))
            run_start = None

    for i, raw in enumerate(lines):
        line = raw.strip()

        # Blank guard
        if not line:
            continue

        # Drop if part of a detected TOC block
        if i in drop_idx:
            if keep_reason: reasons.append((line, "toc_block"))
            continue

        # Drop solo/short junk
        if token_count(line) <= 4:
            if keep_reason: reasons.append((line, "≤4_tokens"))
            continue

        # Drop explicit index/contents headers
        if INDEX_WORDS.search(line):
            if keep_reason: reasons.append((line, "index_header"))
            continue

        # Drop section-name-only lines
        if is_section_name_only(line):
            if keep_reason: reasons.append((line, "section_name_only"))
            continue

        # Drop obvious title/cover lines
        if looks_like_title_line(line):
            if keep_reason: reasons.append((line, "title_line"))
            continue

        cleaned.append(line)

    out.write_text("\n".join(cleaned) + "\n", encoding="utf-8")

    if keep_reason:
        log = out.with_suffix(".log.tsv")
        log.write_text(
            "reason\tline\n" + "\n".join(f"{r}\t{l}" for l, r in reasons) + "\n",
            encoding="utf-8"
        )

def main():
    ap = argparse.ArgumentParser(description="Clean DSA paragraphs: remove TOC, section names, titles, and very short lines.")
    ap.add_argument("-i","--input", default="data_sharing_agreement_paragraphs.txt")
    ap.add_argument("-o","--output", default="data_sharing_agreement_paragraphs_cleaned.txt")
    ap.add_argument("--log", action="store_true", help="Write a .log.tsv with reasons for dropped lines")
    args = ap.parse_args()
    clean_file(Path(args.input), Path(args.output), keep_reason=args.log)
    print(f"Done → {args.output}")

if __name__ == "__main__":
    main()
