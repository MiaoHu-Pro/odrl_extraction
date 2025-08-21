"""
Deterministic post-processing rules for ODRL-style NER tags.

Use this after CRF decoding (no retraining). It adjusts the BIO tag sequence
based on legal-domain cues and simple lexicons.

Schema: matches your predict-time BIO_HEAD
    ACTION, ASSET, PARTY_ASSIGNER, PARTY_ASSIGNEE, RULE_TYPE, CONSTRAINT, DUTY

Example (inside run_test in predict.py):
    from hybrid_rules import apply_rules_to_predictions

    # pred_label, gold_label already recovered to token-level BIO
    pred_label = apply_rules_to_predictions(pred_label, test_features)

If you need per-sample control, call `postprocess_tags(tokens, tags)` directly.
"""
from __future__ import annotations
import re
from typing import List, Sequence

# --- Lexicons (kept in sync with the canvas labeler) -------------------------
ACTION_VERBS = {
    "allow","perform","process","use","share","disclose","collect","receive","provide",
    "transfer","transmit","send","store","retain","delete","destroy","anonymize","pseudonymize",
    "aggregate","publish","access","view","monitor","profile","sell","reidentify","export","import","distribute"
}
ACTION_NOMINALS = {
    "collection","processing","use","disclosure","transfer","retention","deletion","publication",
    "aggregation","profiling","tracking","access","storage","sharing","distribution"
}
ASSET_HEADS = {
    "data","information","record","records","content","document","documents","file","files","dataset","datasets",
    "metadata","log","logs","metric","metrics","identifier","identifiers","email","emails","name","names",
    "address","addresses","phone","phones","location","locations","biometric","financial","health","phi","pii",
    "account","accounts","credential","credentials","password","passwords","token","tokens","ip","cookie","cookies",
    "device","devices","telemetry","analytics","user","users","personal","special","category", "report", "confidential report"
}
DUTY_VERBS = {
    "ensure","implement","maintain","provide","notify","report to","cooperate","obtain","document",
    "permit","audit","comply","protect","encrypt","pseudonymize","anonymize","delete","return","certify",
    "segregate","restrict","access","log","train"
}
JURISDICTIONS = {"EU","EEA","UK","United","States","US","California","Switzerland","Canada","Brazil","China","India"}

# --- Helpers -----------------------------------------------------------------

def _tag_range(tags: List[str], start: int, length: int, btag: str, itag: str):
    if start < 0 or start >= len(tags):
        return
    tags[start] = btag
    for k in range(start+1, min(start+length, len(tags))):
        tags[k] = itag


def _strip_special_tokens(toks: Sequence[str]) -> List[str]:
    return [t for t in toks if t not in ("[CLS]", "[SEP]")]

# --- Core rules --------------------------------------------------------------

def postprocess_tags(tokens: Sequence[str], tags: Sequence[str], *, purpose_as_constraint: bool = True) -> List[str]:
    """
    Deterministic fixes using tokens + BIO tags.
    IMPORTANT: tokens may be shorter than tags; we only read tokens up to T_tok
               but keep 'out' at full tag length.
    """
    T_tags = len(tags)
    T_tok  = len(tokens)
    out = list(tags)

    # Work only with the aligned prefix of tokens
    tokens_view = list(tokens[:T_tok])
    low = [t.lower() for t in tokens_view]
    txt = " ".join(low)

    # 1) RULE_TYPE cues -------------------------------------------------------
    for m in re.finditer(r"\b(shall not|must not|may not|prohibited from)\b", txt):
        s = txt[:m.start()].count(" ")
        if s < T_tags:
            _tag_range(out, s, len(m.group(1).split()), "B-RULE_TYPE", "I-RULE_TYPE")
    for m in re.finditer(r"\b(may|can|permitted to|authorized to|shall|must|is required to|will ensure)\b", txt):
        s = txt[:m.start()].count(" ")
        if s < T_tags and not out[s].startswith("B-RULE_TYPE"):  # keep stronger prohibition
            _tag_range(out, s, len(m.group(1).split()), "B-RULE_TYPE", "I-RULE_TYPE")

    # 2) ASSET expansion ------------------------------------------------------
    i = 0
    while i < min(T_tags, T_tok):
        if out[i].startswith(("B-ASSET","I-ASSET")):
            j = i
            # walk current ASSET span (bounded by tags) but only read tokens when j < T_tok
            while j < T_tags and out[j].endswith("ASSET"):
                if j < T_tok and low[j] in ASSET_HEADS:
                    L = j
                    while (L-1) >= 0 and L-1 < T_tok and tokens_view[L-1][:1].isupper() and out[L-1] in ("I-ASSET","O"):
                        L -= 1
                    if L < i:
                        out[L] = "B-ASSET"
                        for p in range(L+1, j+1):
                            if p < T_tags:
                                out[p] = "I-ASSET"
                    break
                j += 1
            # skip to end of ASSET span (tags only)
            while i < T_tags and out[i].endswith("ASSET"):
                i += 1
            continue
        i += 1

    # 3) PARTY determiners ----------------------------------------------------
    for k in range(0, min(T_tags-1, T_tok-1)):
        if out[k].startswith("B-PARTY") and low[k] in {"the","a","an"}:
            out[k] = "O"
            if out[k+1].startswith("I-PARTY"):
                out[k+1] = out[k+1].replace("I-","B-")

    # 4) Location constraints -------------------------------------------------
    for i, w in enumerate(low):               # iterates up to T_tok
        if i >= T_tags: break                 # don't tag beyond tag sequence
        if w in {"outside","within","in", "up to"}:
            end = i+1
            max_end = min(T_tok, i+8)         # never walk past tokens
            found = False
            while end < max_end:
                tok_e = tokens_view[end]
                if tok_e in JURISDICTIONS:
                    found = True
                if tok_e in {",",";","."}:
                    break
                end += 1
            if found:
                out[i] = "B-CONSTRAINT"
                for p in range(i+1, min(end, T_tags)):
                    if out[p] == "O":
                        out[p] = "I-CONSTRAINT"

    # 5) Temporal constraints (regex over txt; tags only) --------------------
    for m in re.finditer(r"\b(within|no\s+later\s+than|at\s+least|for|up\s+to|prior\s+to|after|before)\b\s+\d{1,4}\s+(day|days|week|weeks|month|months|year|years|hour|hours)\b", txt):
        s = txt[:m.start()].count(" ")
        if s < T_tags:
            l = len(m.group(0).split())
            _tag_range(out, s, l, "B-CONSTRAINT", "I-CONSTRAINT")

    # 6) Purpose -> CONSTRAINT (no PURPOSE tag in schema) --------------------
    for m in re.finditer(r"\bin order to\b", txt):
        s = txt[:m.start()].count(" ")
        if s < T_tags:
            _tag_range(out, s, 3, "B-CONSTRAINT", "I-CONSTRAINT")
    for i, w in enumerate(low):
        if i >= T_tags: break
        if w == "to" and i+1 < T_tok and re.match(r"^[a-z]+$", low[i+1]):
            end = i+2
            while end < T_tok and tokens_view[end] not in {",",";","."}:
                end += 1
            out[i] = "B-CONSTRAINT"
            for p in range(i+1, min(end, T_tags)):
                out[p] = "I-CONSTRAINT"

    # 7) DUTY clauses ---------------------------------------------------------
    for m in re.finditer(r"\b(provided\s+that|so\s+long\s+as|on\s+condition\s+that)\b", txt):
        s = txt[:m.start()].count(" ")
        if s < T_tags:
            l = len(m.group(1).split())
            _tag_range(out, s, l, "B-DUTY", "I-DUTY")
    for i, w in enumerate(low):
        if i >= T_tags: break
        if w in DUTY_VERBS:
            end = i+1
            while end < T_tok and tokens_view[end] not in {",",";","."}:
                end += 1
            out[i] = "B-DUTY"
            for p in range(i+1, min(end, T_tags)):
                if out[p] == "O":
                    out[p] = "I-DUTY"

    return out


# --- Batch helper ------------------------------------------------------------

def apply_rules_to_predictions(pred_bio: List[List[str]], features: List[object]) -> List[List[str]]:
    """Apply `postprocess_tags` to each sequence using tokens from features.
    Assumes each feature has `text` with tokens including specials that we'll strip.
    """
    fixed = []
    for i, tags in enumerate(pred_bio):
        if i < len(features):
            feat = features[i]
            toks = feat.text if isinstance(feat.text, list) else str(feat.text).split()
            toks = _strip_special_tokens(toks)
            if len(toks) != len(tags):
                toks = toks[:len(tags)]  # keep alignment safe
            fixed.append(postprocess_tags(toks, tags))
        else:
            fixed.append(list(tags))
    return fixed

# --- Quick self-test ---------------------------------------------------------
if __name__ == "__main__":
    sent = "Provider may transfer Customer Data outside the EEA to provide support."
    toks = ["Provider","may","transfer","Customer","Data","outside","the","EEA","to","provide","support","."]
    tags = ["B-PARTY_ASSIGNER","O","B-ACTION","B-ASSET","I-ASSET","O","O","O","O","O","O","O"]
    out = postprocess_tags(toks, tags)
    assert len(out) == len(tags)
    print("OK len match; sample:", list(zip(toks, out[:len(toks)])))
    print("out:", out)


    print(list(zip(toks, tags, out)))
