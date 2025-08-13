#!/usr/bin/env python3
import re
import argparse
import unicodedata
from pathlib import Path

# ---------- junk & structure patterns ----------

# Obvious page junk / headers / footers
DROP_PATTERNS = [
    r"^\s*Page\s+\d+\s+of\s+\d+\s*$",   # Page 2 of 23
    r"^\s*\d{1,3}\s*$",                 # orphan page numbers: "2"
    r"^\s*TfL RESTRICTED\s*$",
    r"^\s*Contents\s*$",
    r"^\s*Document History\s*$",
]
drop_re = [re.compile(p, re.I) for p in DROP_PATTERNS]

# ALL-CAPS headings that should start a new block
ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9 ,.&/()'’\-:]+$")

# Numbered/lettered/bulleted starts (e.g. 1., 1.2., a), (i), -, •)
LIST_OR_CLAUSE_START_RE = re.compile(
    r"""
    ^\s*(
        (?:\d+(?:\.\d+)*[.)]?) |       # 1. 1.2 1.2.3 1)
        (?:[ivxlcdm]+[.)])      |      # i) iv) v)
        (?:[A-Za-z][.)])        |      # a) A)
        (?:\(\d+\))             |      # (1)
        (?:[-–•●*])                    # bullets
    )\s+
    """,
    re.VERBOSE | re.IGNORECASE
)

# Section-like starts
SECTION_START_RE = re.compile(
    r"^\s*(Schedule|Appendix|Annex|Background|Purpose|Introduction|Definitions|Agreed Terms)\b",
    re.I
)

# Signature blocks: triggers & field lines
SIGNATURE_BLOCK_START_RE = re.compile(
    r"""^\s*(Signatures?|Signatories|Signature Block|Execution|Executed|
            Counterparts\s+and\s+Signature|Authori[sz]ed\s+Signatory)\b""",
    re.I | re.X
)
SIGNATURE_FIELD_RE = re.compile(
    r"""^\s*((Signed( by)?( for and on behalf of)?)
             |(For and on behalf of)
             |(Signature)
             |(Print(ed)? Name)
             |(Name[:\s])
             |(Title|Role|Position)
             |(Witness)
             |(Date[:\s]?)
             |(By[:\s]?)
             )""",
    re.I | re.X
)

# Decorative “special symbols” / lines (underscores, dashes, dot leaders, boxes)
DECOR_LINE_RE = re.compile(r"""^\s*([_.\-–—=]{3,}|[.•]{5,}|[╔═╗╚╝│]+)\s*$""")

# Remove bullet prefix but keep numbering/clauses
STRIP_BULLET_PREFIX_RE = re.compile(r"^\s*[-–•●*]\s+")

# ---------- helpers ----------

def normalize_unicode(s: str) -> str:
    # NFKC fold + strip zero-width & control chars
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = "".join(ch for ch in s if (ch == "\n" or unicodedata.category(ch)[0] != "C"))
    return s

def is_drop_line(line: str) -> bool:
    if DECOR_LINE_RE.match(line):
        return True
    return any(p.match(line) for p in drop_re)

def is_all_caps_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or not ALL_CAPS_HEADING_RE.match(s):
        return False
    return any(c.isalpha() for c in s) and s == s.upper()

def starts_new_block(line: str) -> bool:
    return (
        LIST_OR_CLAUSE_START_RE.match(line) is not None
        or is_all_caps_heading(line)
        or SECTION_START_RE.match(line) is not None
    )

def strip_bullet_prefix(line: str) -> str:
    # Remove leading bullets like "• " or "-" but keep numbering like "1.2"
    return STRIP_BULLET_PREFIX_RE.sub("", line, count=1)

def clean_linebreak_hyphen(prev: str, cur: str) -> tuple[str, str]:
    # Heal word- hyphenation across broken lines
    if prev.endswith("-") and not prev.endswith(" -") and cur and cur[0].islower():
        return prev[:-1], cur.lstrip()
    return prev, cur

def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def looks_like_mostly_symbols(s: str) -> bool:
    # Drop lines that are mostly punctuation/symbols
    if not s:
        return True
    letters = sum(ch.isalnum() for ch in s)
    return letters / max(len(s), 1) < 0.25

# ---------- core paragraphizer ----------

def paragraphize(text: str) -> list[str]:
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    paras = []
    cur = ""
    in_signature = False
    sig_buffer = 0  # how many subsequent lines to examine while in signature state

    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()

        # Skip empties early; also end paragraphs
        if not stripped:
            if cur:
                paras.append(normalize_whitespace(cur))
                cur = ""
            # reset signature short-circuit on blank
            if in_signature:
                sig_buffer += 1
            continue

        # Drop obvious junk lines
        if is_drop_line(stripped) or looks_like_mostly_symbols(stripped):
            continue

        # Signature: start detection
        if SIGNATURE_BLOCK_START_RE.match(stripped):
            in_signature = True
            sig_buffer = 0
            # flush any paragraph before signature area
            if cur:
                paras.append(normalize_whitespace(cur))
                cur = ""
            continue

        # If we're already inside a signature area, keep skipping while we see signature-ish lines
        if in_signature:
            if SIGNATURE_FIELD_RE.match(stripped) or DECOR_LINE_RE.match(stripped):
                sig_buffer = 0  # reset as long as we keep seeing signature fields
                continue
            # If we hit a clear new section/heading/list, leave signature mode
            if starts_new_block(stripped):
                in_signature = False
                sig_buffer = 0
            else:
                # Allow a little non-signature noise inside, then auto-exit
                sig_buffer += 1
                if sig_buffer <= 2:
                    continue
                in_signature = False  # fall through to normal handling

        # New block? flush current
        if cur and starts_new_block(stripped):
            paras.append(normalize_whitespace(cur))
            cur = ""

        # Clean bullet prefix and append
        stripped = strip_bullet_prefix(stripped)

        if not cur:
            cur = stripped
        else:
            before, after = clean_linebreak_hyphen(cur, stripped)
            if before is not cur:
                cur = before + after
            else:
                cur = cur + " " + stripped

    if cur:
        paras.append(normalize_whitespace(cur))

    # Final pass: remove any remnants we don’t want
    out = []
    for p in paras:
        if is_drop_line(p) or looks_like_mostly_symbols(p):
            continue
        out.append(p)
    return out

# ---------- batch runner ----------

def read_all_txt(dirs: list[Path]) -> list[tuple[str, str]]:
    out = []
    for d in dirs:
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.txt")):
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = fp.read_text(errors="ignore")
            out.append((str(fp), txt))
    return out

def main():
    ap = argparse.ArgumentParser(description="Split DSA .txt files into paragraphs, removing symbols and signature blocks.")
    ap.add_argument("--dirs", nargs="+",
                    default=["data_sharing_agreement_1_text", "data_sharing_agreement_2_text"],
                    help="Folders containing .txt files")
    ap.add_argument("-o", "--output", default="data_sharing_agreement_paragraphs.txt",
                    help="Output file path (txt)")
    ap.add_argument("--mode", choices=["plain", "tsv"], default="plain",
                    help="plain: one paragraph per line; tsv: file\\tpara_no\\tparagraph")
    args = ap.parse_args()

    dirs = [Path(p) for p in args.dirs]
    srcs = read_all_txt(dirs)

    total = 0
    lines_out = []
    if args.mode == "tsv":
        lines_out.append("file\tpara_no\tparagraph")

    for src, txt in srcs:
        print("src", src)

        paras = paragraphize(txt)
        if args.mode == "plain":
            lines_out.extend(paras)
        else:
            for i, p in enumerate(paras, 1):
                safe = p.replace("\t", " ").strip()
                lines_out.append(f"{src}\t{i}\t{safe}")
        total += len(paras)

    Path(args.output).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Done. Wrote {total} paragraphs from {len(srcs)} files to {args.output}")

if __name__ == "__main__":
    main()
