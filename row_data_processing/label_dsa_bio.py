
import json, re, sys
from typing import List, Tuple, Dict

def tokenize(text: str) -> List[str]:
    # Split into words and keep punctuation as separate tokens
    # Handles em dashes/quotes/# numerals reasonably
    tokens = re.findall(r"[A-Za-z0-9#]+|[“”\"'()\-–—:;,\.]", text)
    # Merge paired quotes into their own tokens (already captured)
    return tokens

def to_lower_tokens(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens]

def find_phrase_spans(tokens: List[str], phrase: str) -> List[Tuple[int,int]]:
    # Returns list of (start, length) for literal phrase match over tokens
    ptoks = tokenize(phrase)
    ltoks = to_lower_tokens(tokens)
    lptoks = [p.lower() for p in ptoks]
    spans = []
    if not ptoks:
        return spans
    for i in range(len(tokens) - len(ptoks) + 1):
        if ltoks[i:i+len(ptoks)] == lptoks:
            spans.append((i, len(ptoks)))
    return spans

def label_span(labels: List[str], start: int, length: int, tag: str):
    # Only label O tokens and don't overwrite existing labels
    for j in range(length):
        idx = start + j
        if labels[idx] != "O":
            continue
        prefix = "B-" if j == 0 else "I-"
        labels[idx] = prefix + tag

def safe_label_span(labels: List[str], start: int, length: int, tag: str):
    # Avoid overlapping spans by only writing where label is O; stop early on conflicts
    wrote_any = False
    for j in range(length):
        idx = start + j
        if labels[idx] != "O":
            # conflict -> abort and revert what we've written so far
            for k in range(j):
                labels[start+k] = "O"
            return False
        prefix = "B-" if j == 0 else "I-"
        labels[idx] = prefix + tag
        wrote_any = True
    return wrote_any

def label_rule_type(tokens: List[str], labels: List[str]):
    ltoks = to_lower_tokens(tokens)
    # Negatives: two-token patterns first
    neg_pairs = [("will","not"), ("shall","not"), ("must","not"), ("may","not")]
    for i in range(len(tokens)-1):
        for a,b in neg_pairs:
            if ltoks[i] == a and ltoks[i+1] == b:
                if labels[i] == "O":
                    labels[i] = "B-RULE_TYPE"
                if labels[i+1] == "O":
                    labels[i+1] = "I-RULE_TYPE"
    # Single modals
    for i,t in enumerate(ltoks):
        if t in {"shall","must","will","should"}:
            if labels[i] == "O":
                labels[i] = "B-RULE_TYPE"

def label_actions(tokens: List[str], labels: List[str]):
    # Longer multi-word actions first
    action_phrases = [
        "granted access",
        "be returned",
        "be released",
        "be utilized",
        "be used",
    ]
    single_actions = {
        "release","disclose","share","publish","present","remain","return","comply",
        "utilize","utilised","utilized","governing","acceptance","provided","grant","granted","govern"
    }
    for phrase in action_phrases:
        for (s,l) in find_phrase_spans(tokens, phrase):
            safe_label_span(labels, s, l, "ACTION")
    ltoks = to_lower_tokens(tokens)
    for i,t in enumerate(ltoks):
        if t in single_actions:
            if labels[i] == "O":
                labels[i] = "B-ACTION"

def label_assets(tokens: List[str], labels: List[str]):
    # Order by decreasing length to prefer longer matches
    asset_phrases = [
        "results of data analysis",
        "findings or conclusions",
        "names of individuals",
        "individual addresses",
        "conclusions",
        "findings",
        "information",
        "maps",
        "addresses",
        "data",
    ]
    for phrase in asset_phrases:
        for (s,l) in find_phrase_spans(tokens, phrase):
            safe_label_span(labels, s, l, "ASSET")

def is_party_token(tok: str) -> bool:
    t = tok.lower()
    return t in {"recipient","provider","party","parties","ridoh","tpp"} or t.endswith("party") or t.endswith("parties")

def label_parties(tokens: List[str], labels: List[str]):
    ltoks = to_lower_tokens(tokens)
    # Literal phrases first
    assignee_phrases = [
        "data recipient",
        "third party",
        "third parties",
        "both parties",
        "tpp"
    ]
    assigner_phrases = [
        "data provider",
        "ridoh"
    ]
    for phrase in assignee_phrases:
        for (s,l) in find_phrase_spans(tokens, phrase):
            safe_label_span(labels, s, l, "PARTY_ASSIGNEE")
    for phrase in assigner_phrases:
        for (s,l) in find_phrase_spans(tokens, phrase):
            safe_label_span(labels, s, l, "PARTY_ASSIGNER")

    # Contextual heuristics: after "by" => assigner; after "to" => assignee; after "from" in approval => assigner
    for i,t in enumerate(ltoks):
        if t in {"by","to","from"}:
            j = i+1
            # gather up to 4 tokens that look like a party NP
            span_tokens = []
            while j < len(tokens) and (is_party_token(tokens[j]) or re.match(r"[A-Z]{2,}$", tokens[j]) or tokens[j].lower() in {"the","data"}):
                span_tokens.append(tokens[j])
                j += 1
            if span_tokens:
                length = len(span_tokens)
                tag = None
                if t == "by":
                    tag = "PARTY_ASSIGNER"
                elif t == "to":
                    tag = "PARTY_ASSIGNEE"
                elif t == "from":
                    # If "approval from", prefer assigner
                    if i>0 and ltoks[i-1] in {"approval","approval,"} or (i>1 and ltoks[i-2] == "approval"):
                        tag = "PARTY_ASSIGNER"
                if tag:
                    safe_label_span(labels, i+1, length, tag)

def label_constraints(tokens: List[str], labels: List[str]):
    ltoks = to_lower_tokens(tokens)

    # Chunk constraints that start with known markers and extend to comma/period
    starts = {"without","except","solely","pursuant","under","upon","before"}
    # Special short phrase
    in_writing = [("in","writing")]
    for i in range(len(tokens)-1):
        if ltoks[i] == "in" and ltoks[i+1] == "writing":
            if labels[i] == "O":
                labels[i] = "B-CONSTRAINT"
            if labels[i+1] == "O":
                labels[i+1] = "I-CONSTRAINT"

    i = 0
    while i < len(tokens):
        if ltoks[i] in starts:
            j = i+1
            # extend until a hard stop
            while j < len(tokens) and tokens[j] not in {",",";","."}:
                j += 1
            length = j - i
            # Avoid labeling if this is actually an ASSET (e.g., "under condition #2" still fine as constraint)
            safe_label_span(labels, i, length, "CONSTRAINT")
            i = j
        else:
            i += 1

    # "as permitted under ..." -> start at "as permitted under"
    for i in range(len(tokens)-2):
        if ltoks[i] == "as" and ltoks[i+1] == "permitted" and ltoks[i+2] == "under":
            j = i+3
            while j < len(tokens) and tokens[j] not in {",",";","."}:
                j += 1
            safe_label_span(labels, i, j-i, "CONSTRAINT")

    # "governing the confidentiality ..." as constraint
    for i in range(len(tokens)-1):
        if ltoks[i] in {"governing"}:
            j = i+1
            while j < len(tokens) and tokens[j] not in {".",";"}:
                j += 1
            safe_label_span(labels, i, j-i, "CONSTRAINT")

def label_sentence(text: str) -> Dict:
    tokens = tokenize(text)
    labels = ["O"] * len(tokens)

    # Order matters: parties/assets before actions often okay; constraints last-ish
    label_parties(tokens, labels)
    label_assets(tokens, labels)
    label_rule_type(tokens, labels)
    label_actions(tokens, labels)
    label_constraints(tokens, labels)

    return {"text": text, "tokens": tokens, "labels": labels}

def main():
    import argparse, sys, json
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL with {'text': ...}")
    ap.add_argument("output", help="JSONL with {'text','tokens','labels'}")
    ap.add_argument("--preview", type=int, default=0, help="print first N labeled examples to stdout")
    args = ap.parse_args()

    count = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            res = label_sentence(obj["text"])
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            count += 1
            if args.preview and count <= args.preview:
                print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
