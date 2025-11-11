#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch clean + sentence-split TXT files, no argparse.

Usage:
  python clean_sentences_no_argparse.py [INPUT_DIR] [OUTPUT_DIR] [--recursive] [--alpha=0.5] [--audit]
"""

import re
import sys
import unicodedata
from pathlib import Path

# ----------------------------
# User settings (defaults)
# ----------------------------
INPUT_DIR  = Path("./data_sharing_agreement_2_text")
OUTPUT_DIR = Path("./output_after_process_script1")
RECURSIVE  = False
ALPHA_THRESHOLD = 0.5
AUDIT = False

# ----------------------------
# Minimal CLI override (optional)
# ----------------------------
args = sys.argv[1:]
positional = [a for a in args if not a.startswith("--")]
flags = [a for a in args if a.startswith("--")]

if len(positional) >= 1:
    INPUT_DIR = Path(positional[0])
if len(positional) >= 2:
    OUTPUT_DIR = Path(positional[1])

for fl in flags:
    if fl == "--recursive":
        RECURSIVE = True
    elif fl.startswith("--alpha="):
        try:
            ALPHA_THRESHOLD = float(fl.split("=", 1)[1])
        except ValueError:
            pass
    elif fl == "--audit":
        AUDIT = True

# ----------------------------
# Regexes & helpers
# ----------------------------
PAGE_NUM_RE = re.compile(r'^(?:Page\s*:?\s*)?\d+\s*$')
TOC_ROW_RE = re.compile(r'^\s*\d+(?:\.\d+)*\.\s+.+?\s+\d+\s*$')      # “1.2.3. Heading .... 12”
ALLCAPS_SHORT_HEADING_RE = re.compile(r'^[A-Z][A-Z0-9\s/&\-\.,]{3,60}$')
# Bullet (stripper) — allow zero or more spaces after symbol; include ●
BULLET_RE = re.compile(r'^\s*([•●◦▪\-–—]|\d+[.)])\s*')
PURE_NUMERIC_LINE_RE = re.compile(r'^[\d./() \-]+$')

# Boundaries & flow
BULLET_START_RE = BULLET_RE  # use same start test
TOC_LIKE_RE = TOC_ROW_RE
SECTION_NUM_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s+[A-Z]')            # “4.1 Purpose...”
HARD_END_RE = re.compile(r'[.!?]["\')\]]*\s*$')

# Abbreviations that end with a dot but MUST NOT split
ABBREV = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|No|Art|Sec|Dept|Inc|Ltd|Co|Corp|Univ|Ref|Fig|Eq|Ch|Vol|Para|Annex|App|Dir|Pg|pp|vs|e\.g|i\.e|etc|cf)'
ABBREV_DOT_RE = re.compile(rf'\b{ABBREV}\.', flags=re.IGNORECASE)

# Clause numbers like 4.1.5. (protect the final dot)
CLAUSE_NUM_RE = re.compile(r'\b\d+(?:\.\d+)+\.')

# Token used to protect dots during splitting
DOT_PROTECT = '§DOT§'

# Ghosts often found at line starts from PDF→TXT
LEADING_GHOSTS_RE = re.compile(r'^[\ufeff\u200b\u200c\u200d\u2060\u00a0\s]+')

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text

def drop_noise_lines(text: str) -> str:
    out_lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        # keep blank lines as paragraph breaks
        if s == "":
            out_lines.append(ln)
            continue
        if PAGE_NUM_RE.match(s):
            continue
        if TOC_ROW_RE.match(s):
            continue
        if ALLCAPS_SHORT_HEADING_RE.match(s):
            # comment next line to KEEP headings
            continue
        # strip leading bullets (works with or without a space)
        ln = BULLET_RE.sub("", ln)
        out_lines.append(ln)
    return "\n".join(out_lines)

def fix_hyphenation(text: str) -> str:
    # join foo-\nbar -> foobar (allow next to be upper/lower)
    text = re.sub(r'([A-Za-z])-\s*\n\s*([A-Za-z])', r'\1\2', text)
    return text

def is_hard_boundary(ln: str) -> bool:
    s = ln.strip()
    if not s:
        return True
    if BULLET_START_RE.match(s): return True
    if TOC_LIKE_RE.match(s): return True
    if SECTION_NUM_RE.match(s): return True
    if ALLCAPS_SHORT_HEADING_RE.match(s): return True
    return False

def unwrap_soft_linebreaks(text: str) -> str:
    """
    Merge soft wraps inside paragraphs while preserving:
      - blank lines (paragraph breaks)
      - bullets / numbered items
      - short ALL-CAPS headings
      - TOC-like lines and obvious section headings
    """
    lines = text.split('\n')
    out, buf = [], []

    def flush():
        if not buf:
            return
        joined = buf[0].strip()
        for nxt in buf[1:]:
            prev_trim = joined.rstrip()
            nxt_trim = nxt.strip()
            if re.search(r'[A-Za-z]-\s*$', prev_trim):
                joined = re.sub(r'-\s*$', '', prev_trim) + nxt_trim
            else:
                joined = prev_trim + ' ' + nxt_trim
        out.append(joined)
        buf.clear()

    for ln in lines:
        if is_hard_boundary(ln):
            flush()
            out.append(ln.strip())
            continue

        if not buf:
            buf.append(ln)
            continue

        prev = buf[-1].rstrip()

        if HARD_END_RE.search(prev):
            flush()
            buf.append(ln)
        else:
            buf.append(ln)

    flush()
    text2 = '\n'.join(out)
    text2 = re.sub(r'\n{3,}', '\n\n', text2)
    return text2

def protect_dots(text: str) -> str:
    def repl_abbrev(m):
        return m.group(0).rstrip('.') + DOT_PROTECT
    def repl_clause(m):
        return m.group(0).rstrip('.') + DOT_PROTECT
    text = ABBREV_DOT_RE.sub(repl_abbrev, text)
    text = CLAUSE_NUM_RE.sub(repl_clause, text)
    return text

def unprotect_dots(text: str) -> str:
    return text.replace(DOT_PROTECT, '.')

def join_soft_wraps(text: str) -> str:
    """Final safety join for single newlines inside paragraphs where previous char isn't .!?
    (Most work is done by unwrap_soft_linebreaks.)"""
    return re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)

# --- Sentence splitting (procedural; newline-aware; bullet-aware) ---
BULLET_CHARS = set("•●◦▪-–—")
OPENERS = set("\"'([“‘")
END_WS_RE = re.compile(r'[.!?]["\')\]]*\s+')

def sentence_split(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # ensure space after closing quotes/brackets
    text = re.sub(r'([.!?])([)\]"}\'])', r'\1 \2', text)
    text = protect_dots(text)

    parts = []
    start = 0
    n = len(text)

    for m in END_WS_RE.finditer(text):
        ws = m.group(0)            # punctuation + trailing whitespace
        span_end = m.end()
        left = text[start:span_end].rstrip()

        # If protected dot just before ws, it's not a true end (abbrev/clauses)
        if left.endswith(DOT_PROTECT):
            continue

        newline_in_ws = '\n' in ws

        # Next non-space character
        k = span_end
        while k < n and text[k].isspace():
            k += 1
        next_ch = text[k] if k < n else ''

        looks_like_start = (
            newline_in_ws or
            (next_ch and (next_ch.isupper() or next_ch.isdigit() or next_ch in OPENERS or next_ch in BULLET_CHARS))
        )

        if looks_like_start:
            parts.append(unprotect_dots(text[start:k]).strip())
            start = k

    if start < n:
        parts.append(unprotect_dots(text[start:]).strip())

    return [p for p in parts if p]

def is_wordy(s: str, alpha_threshold: float = 0.5) -> bool:
    s2 = s.strip()
    if not s2:
        return False
    if PURE_NUMERIC_LINE_RE.match(s2):
        return False
    alpha = sum(ch.isalpha() for ch in s2)
    nonspace = sum(ch not in ' \t\n' for ch in s2)
    if nonspace == 0:
        return False
    return (alpha / nonspace) >= alpha_threshold

# --- Remove leading enumerators/bullets (even when tight like "●ensuring") ---
LEAD_ENUM_MASTER = re.compile(r"""
^                                   # start of string
(?:                                  # one of:
   [•●◦▪\-–—]                        #   bullet symbols
 | \(\s*(?:\d+(?:\.\d+){0,4}         #   (1) or (1.2.3.4)
            | [A-Za-z]               #   (a) / (A)
            | [ivxlcdmIVXLCDM]{1,5}  #   (i) / (III)
        )\s*\)                       #   closing )
 | \d+(?:\.\d+){0,4}[.)]?            #   1. or 1.2.3. or 1.2)   (dot/paren optional)
 | [A-Za-z][.)]                      #   a) or b. or A)
 | [ivxlcdmIVXLCDM]{1,5}[.)]         #   i. / ii) / IV.
)
\s*                                  # allow ZERO OR MORE spaces after enumerator
""", re.VERBOSE)

def strip_leading_enumerator(s: str) -> str:
    """
    Remove leading numbering/bullets like:
      '1.', '1.2.', '●', 'i.', 'ii)', 'c)', '(a)', '(1.2.)'
    Repeats until nothing more can be stripped. Handles BOM/ZW/NBSP.
    """
    # strip ghosts & normal spaces first
    s = LEADING_GHOSTS_RE.sub('', s)
    prev = None
    while prev != s:
        prev = s
        s = LEAD_ENUM_MASTER.sub('', s, count=1)
        s = LEADING_GHOSTS_RE.sub('', s)
    return s

def clean_and_split(content: str, alpha_threshold: float) -> list[str]:
    content = normalize_text(content)
    content = drop_noise_lines(content)
    content = fix_hyphenation(content)
    content = unwrap_soft_linebreaks(content)  # robust unwrapping
    content = join_soft_wraps(content)         # last-mile join

    sents = sentence_split(content)
    # strip leading enumerators and re-filter
    sents = [strip_leading_enumerator(s) for s in sents]
    sents = [s for s in sents if s]  # drop empties
    sents = [s for s in sents if is_wordy(s, alpha_threshold)]
    return sents

def process_file(in_path: Path, out_path: Path, alpha_threshold: float) -> list[str]:
    try:
        text = in_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = in_path.read_text(encoding="latin-1", errors="ignore")
    sentences = clean_and_split(text, alpha_threshold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sentences) + ("\n" if sentences else ""), encoding="utf-8")
    return sentences

def audit_sentences(original_text: str, sentences: list[str]) -> list[str]:
    issues = []
    for s in sentences:
        if '\n' in s:
            issues.append(f"[Leftover newline in sentence] {s[:140]}...")
    lines = original_text.splitlines()
    for i in range(len(lines)-1):
        a = lines[i].strip()
        b = lines[i+1].strip()
        if not a or not b:
            continue
        if not a.endswith(('.', '!', '?')) and (b[:1].islower() or (b and b[0].isdigit())):
            snippet = (a + " " + b)[:160]
            issues.append(f"[Likely soft wrap] L{i+1}-{i+2}: {snippet} ...")
    out = []
    seen = set()
    for x in issues:
        if x not in seen:
            out.append(x); seen.add(x)
        if len(out) >= 30:
            break
    return out

def main():
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"[ERROR] Input folder not found or not a directory: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.rglob("*.txt") if RECURSIVE else INPUT_DIR.glob("*.txt"))
    if not files:
        print("[WARN] No .txt files found.")
        sys.exit(2)

    total = 0
    for f in files:
        rel = f.relative_to(INPUT_DIR) if RECURSIVE else Path(f.name)
        out_file = OUTPUT_DIR / rel

        sentences = process_file(f, out_file, ALPHA_THRESHOLD)
        total += 1

        if AUDIT:
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                raw = f.read_text(encoding="latin-1", errors="ignore")
            issues = audit_sentences(raw, sentences)
            if issues:
                print(f"\n[AUDIT] {f.name} – {len(issues)} hints:")
                for hint in issues[:10]:
                    print("  -", hint)

    print(f"[OK] Processed {total} file(s) into: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
