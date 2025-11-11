#!/usr/bin/env python3
"""
Label DSA sentences with BIO tags via OpenAI Responses API (Structured Outputs),
with optional few-shot examples. Works with a plain text file (one sentence per line).

Install:
  pip install openai python-dotenv tenacity tqdm transformers

.env (example):
  OPENAI_API_KEY=sk-...
  MODEL_NAME=gpt-4o-mini
  WORKERS=8
  INPUT_TXT=data_sharing_agreement_mt20.txt
  OUTPUT_JSONL=labels.jsonl
  FEWSHOT_JSONL=fewshot_examples.jsonl   # optional
  MAX_FEWSHOT=8                          # optional
  TOKENIZER_PATH=bert-base-cased         # or absolute path to your local model dir
"""

import os
import re
import json
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, OpenAIError
from transformers import AutoTokenizer

# ------------------ env & config ------------------

load_dotenv()  # reads .env in current dir

API_KEY        = os.getenv("OPENAI_API_KEY")
MODEL          = os.getenv("MODEL_NAME", "o4-mini")
WORKERS        = int(os.getenv("WORKERS", "6"))
INPUT_TXT      = os.getenv("INPUT_TXT", "data_sharing_agreement_mt20_examples.txt")
OUTPUT_JSONL   = os.getenv("OUTPUT_JSONL", "data_sharing_agreement_mt20_examples_labels.jsonl")
FEWSHOT_JSONL  = os.getenv("FEWSHOT_JSONL", "")      # optional path to few-shot examples JSONL
MAX_FEWSHOT    = int(os.getenv("MAX_FEWSHOT", "32")) # max few-shot examples to load
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", os.getenv("WORD_EMBEDDING_PATH", "bert-base-cased"))

print(MODEL, INPUT_TXT, OUTPUT_JSONL, FEWSHOT_JSONL, MAX_FEWSHOT)

if not API_KEY:
    raise SystemExit("OPENAI_API_KEY not found in environment/.env")

logging.basicConfig(level=logging.INFO, format="%(message)s")

client = AsyncOpenAI(api_key=API_KEY)

# ------------------ tokenizer (HF-based, BIO-friendly) ------------------

HF_TOK = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

# Prefer HF pre-tokenization (works for BERT/DeBERTa/RoBERTa and T5 fast)
USE_MODEL_PRETOKENIZATION = True

def tokenize(text: str) -> list[str]:
    if USE_MODEL_PRETOKENIZATION:
        if hasattr(HF_TOK, "basic_tokenizer") and HF_TOK.basic_tokenizer is not None:
            return HF_TOK.basic_tokenizer.tokenize(
                text, never_split=getattr(HF_TOK, "all_special_tokens", [])
            )
        if getattr(HF_TOK, "is_fast", False) and hasattr(HF_TOK, "backend_tokenizer"):
            pre = HF_TOK.backend_tokenizer.pre_tokenizer
            if pre is not None:
                return [t for t, _ in pre.pre_tokenize_str(text)]
    # Fallback: split words & punctuation (keeps 133-155 -> 133 '-' 155)
    return re.findall(r"[A-Za-z0-9#]+|[“”\"'()\-–—:;,\.\[\]\{\}]", text)

# ------------------ schema & instructions ------------------

LABEL_ENUM = [
   "O",
   "B-PARTY_ASSIGNEE", "I-PARTY_ASSIGNEE",
   "B-ASSET", "I-ASSET",
   "B-ACTION", "I-ACTION",
   "B-RULE_TYPE", "I-RULE_TYPE",
   "B-CONSTRAINT", "I-CONSTRAINT",
   "B-PURPOSE", "I-PURPOSE",
]

SYSTEM_INSTRUCTIONS = """You are an expert BIO annotator for Data Sharing Agreements (DSA).

You will receive:
- tokens: the exact token sequence (punctuation kept as separate tokens),
- seed_labels: heuristic BIO tags to START from.

Your job:
- Correct and improve seed_labels to produce final BIO tags.
- Output EXACTLY one tag per token, same order and length as tokens.
- Return JSON ONLY: {"label":[...]} with tags from the allowed enum.

Tokenization rules:
- Do NOT alter, merge, or invent tokens.
- Punctuation tokens (e.g., '#', ',', '.', '(', ')', '-', ':', ';', quotes) are always 'O'.
- Hyphenated forms appear with '-' as a separate token (e.g., "share - alike").
- CamelCase words may appear split into separate tokens (e.g., "Commercial", "Use").

BIO rules:
- Begin a span with B-<TYPE>, continue with I-<TYPE>.
- Never start an I- span after 'O' or a different type; convert such cases to 'B-'.
- No overlapping spans; one tag per token.

Label meanings (be consistent and minimal):
- ACTION: Verbs/verb phrases describing permitted/forbidden/required activities (e.g., share, publish, be released, grant use, commercial use, share alike, obtain consent).
- ASSET: The governed resource/object (e.g., data, personal data, results of analysis, names, addresses, source code).
- RULE_TYPE: Modal/operational operators expressing permission/prohibition/obligation/conditions (e.g., shall, must, may, will not, is required to, obligated to, responsible for, provided that, subject to, only if).
  * IMPORTANT: There is NO separate DUTY label; any duty-like operator tokens are RULE_TYPE.
  * Include negation with RULE_TYPE (e.g., “will not” → B-RULE_TYPE I-RULE_TYPE).
- CONSTRAINT: Conditions/limits refining scope (time, count, location, approvals; “without…”, “except…”, “prior to…”, “before…”, “in writing”, “between … and …”, “by <date>”, “at least <n> per …”).
- PURPOSE: purposive clauses (“for … purpose(s)”, “for research/statistical purposes”, “in order to …”, “to ensure/enable/prevent…”) — favor PURPOSE for “to …” intent clauses.
- PARTY_ASSIGNEE: Mentions of the receiving party/user/consumer role (e.g., data recipient, third party, licensee, user(s), processor, service provider). Assigner roles are NOT labeled in this task.

Precedence:
1) RULE_TYPE applies to the operator tokens only (modals/duty-like heads/connectors/negations). The main activity verbs remain ACTION (or PURPOSE/CONSTRAINT).
2) If a span could be PURPOSE vs ACTION (e.g., “to ensure compliance”), prefer PURPOSE for the “to …” clause.
3) If a span could be CONSTRAINT vs RULE_TYPE connector, keep the connector head(s) as RULE_TYPE (e.g., “provided that”, “only if”, “subject to”), but label the following content per its type.

Other details:
- Numbers, section refs, symbols are 'O' unless clearly part of a CONSTRAINT.
- Keep spans minimal; do not include trailing punctuation.
- Do NOT output all 'O' when seed_labels contain valid non-'O' spans unless clearly wrong.
- Keep correct seed tags; fix incorrect ones; add missing spans.

Return JSON ONLY per schema: {"label":[...]} with one tag per token.
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "BioSequenceLabels",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {
                    "type": "array",
                    "items": {"type": "string", "enum": LABEL_ENUM}
                }
            },
            "required": ["label"]
        }
    }
}

# ------------------ utilities ------------------

def find_phrase_spans(tokens: List[str], phrase: str) -> List[tuple]:
    ptoks = tokenize(phrase)
    ltoks = [t.lower() for t in tokens]
    lpt = [p.lower() for p in ptoks]
    spans = []
    if not ptoks:
        return spans
    for i in range(len(tokens) - len(ptoks) + 1):
        if ltoks[i:i+len(ptoks)] == lpt:
            spans.append((i, len(ptoks)))
    return spans

def safe_label_span(labels: List[str], start: int, length: int, tag: str) -> bool:
    # write only where O to avoid overwrites
    for j in range(length):
        if labels[start+j] != "O":
            return False
    for j in range(length):
        labels[start+j] = ("B-" if j == 0 else "I-") + tag
    return True

# ------------------ seeders ------------------

def seed_rule_type(tokens: List[str], labels: List[str]):
    """
    RULE_TYPE seeder:
      - Modal and negation (will not, cannot, etc.)
      - Permission/prohibition/obligation phrases
      - Duty-like operators (required/obligated/obliged/responsible/duty/agree to/undertake to)
      - Connectors that introduce obligations/conditions (provided that / only if / subject to)
    Operator tagging is restricted to the operator heads/markers; we do NOT swallow
    the full clause so ACTION/PURPOSE/CONSTRAINT can still tag content.
    """
    lt = [t.lower() for t in tokens]
    n = len(tokens)

    # -------- helpers for 'May' (month) --------
    PREP_BEFORE_DATE = {
        "in","on","of","by","before","after","during","since","until","from","through","throughout","around"
    }
    ORD_RE = re.compile(r"^\d{1,2}(st|nd|rd|th)$", re.I)
    YEAR_RE = re.compile(r"^\d{4}$")
    TIME_RE = re.compile(r"^\d{1,2}:\d{2}([ap]m)?$", re.I)

    def _is_day_like(tok: str) -> bool:
        t = tok.strip()
        return (t.isdigit() and 1 <= int(t) <= 31) or bool(ORD_RE.match(t))

    def _is_year_like(tok: str) -> bool:
        return bool(YEAR_RE.match(tok))

    def _is_time_like(tok: str) -> bool:
        return bool(TIME_RE.match(tok))

    def _is_may_month(i: int) -> bool:
        if not (0 <= i < n): return False
        if tokens[i] not in {"May","MAY"}: return False
        prev_is_prep = (i-1 >= 0 and lt[i-1] in PREP_BEFORE_DATE)
        j = i + 1
        if j < n and (_is_day_like(tokens[j]) or _is_year_like(tokens[j]) or _is_time_like(tokens[j])):
            return True
        if j+1 < n and tokens[j] == "," and (_is_year_like(tokens[j+1]) or _is_day_like(tokens[j+1])):
            return True
        return prev_is_prep

    def _label_span(s: int, l: int):
        # Label s..s+l-1 as RULE_TYPE if all O
        for k in range(l):
            if labels[s+k] != "O":
                return
        labels[s] = "B-RULE_TYPE"
        for k in range(1, l):
            labels[s+k] = "I-RULE_TYPE"

    # -------- negated modals --------
    neg_bigrams = {
        ("will","not"), ("shall","not"), ("must","not"),
        ("may","not"), ("can","not"),
    }
    for i in range(n-1):
        a, b = lt[i], lt[i+1]
        if (a, b) in neg_bigrams and not (tokens[i] in {"May","MAY"} and _is_may_month(i)):
            _label_span(i, 2)

    # 'cannot'
    for i, w in enumerate(lt):
        if w == "cannot" and labels[i] == "O":
            labels[i] = "B-RULE_TYPE"

    # simple contractions
    CONTRACTIONS = {"can't","won't","shan't","shouldn't","mustn't","couldn't","wouldn't","mightn't","mayn't"}
    for i, w in enumerate(lt):
        if w in CONTRACTIONS and labels[i] == "O":
            labels[i] = "B-RULE_TYPE"

    # -------- phrase catalogs --------
    obligation_phrases = [
        "required to", "is required to", "are required to",
        "is obligated to", "are obligated to",
        "is obliged to", "are obliged to",
        "has to", "have to", "needs to", "need to",
        "is to", "are to",
        "is expected to", "are expected to",
        "is assigned to", "are assigned to",
        "is charged with", "are charged with",
        "is tasked with", "are tasked with",
        "obligation to", "commitment to",
        "undertakes to", "undertake to",
        "agrees to", "agree to",
        "is responsible for", "are responsible for",
        "required by",
    ]
    permission_phrases = [
        "is permitted to", "are permitted to",
        "is allowed to", "are allowed to",
        "allow to", "allows to", "allowed to",
        "is allowed", "are allowed",
        "is authorized to", "are authorized to",
        "is entitled to", "are entitled to",
        "has the right to", "have the right to",
        "is enabled to", "are enabled to",
        "is free to", "are free to",
        "at their discretion", "at its discretion", "at the discretion of",
        "subject to approval", "upon consent", "is granted leave to", "are granted leave to",
    ]
    prohibition_phrases = [
        "must not", "shall not", "may not", "can not", "cannot",
        "is not permitted to", "are not permitted to",
        "is not allowed to", "are not allowed to",
        "do not allow", "does not allow", "did not allow",
        "is not authorized to", "are not authorized to",
        "is not entitled to", "are not entitled to",
        "is prohibited from", "are prohibited from",
        "is forbidden to", "are forbidden to",
        "is barred from", "are barred from",
        "shall refrain from", "restricted from",
    ]

    def label_phr_list(plist: List[str]):
        for p in plist:
            for s, l in find_phrase_spans(tokens, p):
                # guard: if phrase starts with 'may' and token is month 'May', skip
                if p.split()[0] == "may" and s < n and tokens[s] in {"May","MAY"} and _is_may_month(s):
                    continue
                _label_span(s, l)

    label_phr_list(obligation_phrases)
    label_phr_list(permission_phrases)
    label_phr_list(prohibition_phrases)

    # -------- single-word markers (incl. duty-like heads) --------
    base_single = {
        # obligation/necessity/expectation
        "must","shall","will","should","needs","need","mandatory","compulsory",
        # permission
        "may","can","could","might","would",
        # allow family
        "allow","allows","allowed","allowing",
        # prohibition adjectives
        "forbidden","prohibited","restricted","barred",
        # normative nouns/adjs (duty-like)
        "duty","obligation","responsibility","responsible","required","obligated","obliged","expected",
        "commitment","committed","commit","commits","binding","bound"
    }
    for i, w in enumerate(lt):
        if labels[i] != "O":
            continue
        if tokens[i] in {"May","MAY"} and _is_may_month(i):
            continue
        if w in base_single:
            labels[i] = "B-RULE_TYPE"

    # -------- connectors as operator heads (RULE_TYPE only on head tokens) --------
    connector_heads = [
        ["provided","that"],
        ["provided"],  # (e.g., 'provided, however, that')
        ["only","if"],
        ["subject","to"],
        ["so","long","as"],
        ["as","long","as"],
        ["on","condition","that"],
    ]
    def label_connector_head(i: int, length: int):
        for k in range(length):
            if labels[i+k] != "O":
                return
        labels[i] = "B-RULE_TYPE"
        for k in range(1, length):
            labels[i+k] = "I-RULE_TYPE"

    L = len(tokens)
    for form in connector_heads:
        fl = len(form)
        j = 0
        while j <= L - fl:
            if [t.lower() for t in tokens[j:j+fl]] == form:
                if form == ["provided"]:
                    label_connector_head(j, 1)
                else:
                    label_connector_head(j, fl)
                j += fl
            else:
                j += 1

def seed_actions(tokens: List[str], labels: List[str]):
    """
    ACTION seeder with modal+infinitive support and multi-token phrases.
    """
    phrase_list = [
        "granted access", "be returned", "be released", "be utilized", "be used",
        "commercial use", "commerical use", "concurrent use", "secondary use", "grant use",
        "attach policy", "attach source", "review policy", "next policy",
        "obtain consent", "ensure exclusivity", "source code",
        "write to", "append to",
        "share alike", "text to speech", "ad hoc share",
        "extract char", "extract page", "extract word",
        "derivative works", "accept tracking",
        "text-to-speech", "ad-hoc share", "share-alike",
    ]
    for p in phrase_list:
        for s, l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "ACTION")

    base_actions = {
        "release","disclose","share","publish","present","remain","return",
        "comply","utilize","utilized","facilitate","agree","handle","handling",
        "provide","assist","assistance","download","notice",
        "aggregate", "annotate", "anonymize", "append", "archive", "attribute",
        "commercialize", "compensate", "copy", "delete", "derive", "digitize",
        "display", "distribute", "execute", "export", "extract", "give", "include",
        "index", "inform", "install", "lease", "lend", "license", "modify", "move",
        "pay", "play", "preview", "print", "read", "reproduce", "sell", "stream",
        "synchronize", "transfer", "transform", "translate", "uninstall", "use",
        "watermark", "write",
    }

    def _norm(tok: str) -> str:
        return tok.lower().replace("_", "-")

    def _morph_forms(n: str) -> set:
        forms = {n}
        if n.endswith("ies") and len(n) > 4:
            forms.add(n[:-3] + "y")
        if n.endswith("es") and len(n) > 3:
            forms.add(n[:-2])
            forms.add(n[:-2] + "e")
        if n.endswith("s") and len(n) > 3:
            forms.add(n[:-1])
            forms.add(n[:-1] + "e")
        return forms

    def _is_action_like(tok: str) -> bool:
        n = _norm(tok)
        return any(f in base_actions for f in _morph_forms(n))

    def _label_phrase_at(pos: int) -> bool:
        for p in phrase_list:
            for s, l in find_phrase_spans(tokens, p):
                if s == pos:
                    if safe_label_span(labels, s, l, "ACTION"):
                        return True
        return False

    MODALS = {"can","may","might","must","shall","should","will","would","could","cannot"}
    NOT_TOKENS = {"not","n't","n’t"}

    i = 0
    while i < len(tokens):
        t = tokens[i].lower()
        if t in MODALS:
            j = i + 1
            if t != "cannot" and j < len(tokens) and tokens[j].lower() in NOT_TOKENS:
                j += 1
            if j < len(tokens) and tokens[j] in {"'", "’"} and j + 1 < len(tokens) and tokens[j+1].lower() == "t":
                j += 2

            if j < len(tokens):
                if tokens[j].lower() == "be":
                    if _label_phrase_at(j):
                        i = j + 1
                        continue
                    if j + 1 < len(tokens) and labels[j+1] == "O" and _is_action_like(tokens[j+1]):
                        labels[j+1] = "B-ACTION"
                        i = j + 2
                        continue
                if _label_phrase_at(j):
                    i = j + 1
                    continue
                if labels[j] == "O" and _is_action_like(tokens[j]):
                    labels[j] = "B-ACTION"
                    i = j + 1
                    continue
        i += 1

    for i, tok in enumerate(tokens):
        if labels[i] != "O":
            continue
        if _is_action_like(tok):
            labels[i] = "B-ACTION"

def seed_assets(tokens: List[str], labels: List[str]):
    """
    Expanded ASSET seeder for DSA/ODRL:
    1) Reserve known multi-word phrases.
    2) Label modifier chains before head nouns.
    3) Label safe single-word heads.
    """
    phrase_list = [
        "results of data analysis", "results of the data analysis", "results of analysis",
        "findings or conclusions", "findings and conclusions", "findings of the analysis",
        "statistical outputs", "statistical output",
        "personal data", "special category data", "special category personal data",
        "criminal data", "biometric data", "genetic data", "health data",
        "financial data", "payment information", "payment data",
        "contact information", "contact details",
        "location data", "geolocation data", "shared personal data",
        "ip address", "ip addresses", "email address", "email addresses",
        "phone number", "phone numbers", "names of individuals",
        "individual addresses", "identity of individuals", "identity of an individual",
        "source code", "training data", "test data", "research data",
        "documents", "document", "files", "file",
        "images", "image", "photographs", "videos", "video",
        "audio recordings", "audio", "transcripts", "logs", "audit logs", "metadata",
        "maps", "map", "tables", "table",
        "confidential information", "confidential report",
    ]
    for p in phrase_list:
        for s, l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "ASSET")

    modifiers = {
        "personal","special","category","criminal","biometric","genetic","health",
        "financial","payment","contact","location","geolocation","confidential",
        "aggregated","aggregate","de-identified","deidentified","anonymized","anonymised",
        "pseudonymised","pseudonymized","research","training","test","audit","statistical",
        "email","phone","ip","identity"
    }
    heads = {
        "data","information","details","identifiers","identifier",
        "address","addresses","name","names","number","numbers",
        "results","findings","conclusions","records","record",
        "dataset","datasets","outputs","output",
        "report","reports","map","maps","table","tables",
        "document","documents","file","files",
        "image","images","video","videos","audio","transcript","transcripts",
        "log","logs","metadata","code"
    }

    def is_mod(tok: str) -> bool:
        return tok.lower() in modifiers

    def is_head(tok: str) -> bool:
        return tok.lower() in heads

    i = 0
    while i < len(tokens) - 1:
        if labels[i] != "O":
            i += 1
            continue
        j = i
        chain_idxs = []
        saw_mod = False
        while j < len(tokens) and labels[j] == "O":
            tj = tokens[j]
            if tj == "-":
                chain_idxs.append(j); j += 1; continue
            if is_mod(tj):
                chain_idxs.append(j); saw_mod = True; j += 1; continue
            break
        if saw_mod and j < len(tokens) and labels[j] == "O" and is_head(tokens[j]):
            span = chain_idxs + [j]
            span = [k for k in span if tokens[k] != "-"]
            if span:
                for idx_pos, idx in enumerate(span):
                    labels[idx] = "B-ASSET" if idx_pos == 0 else "I-ASSET"
                i = j + 1
                continue
        i += 1

    single_heads = {
        "data","information","records","record","dataset","datasets",
        "results","findings","conclusions","metadata","logs","log",
        "documents","document","files","file","maps","map","tables","table",
        "images","image","videos","video","audio","transcripts","transcript",
        "addresses","address","names","name","numbers","number","outputs","output",
    }
    for idx, tok in enumerate(tokens):
        if labels[idx] != "O":
            continue
        if tok.lower() in single_heads:
            labels[idx] = "B-ASSET"

def seed_parties(tokens: List[str], labels: List[str]):
    """
    PARTY_ASSIGNEE only (assigner removed).
    """
    lt = [t.lower() for t in tokens]
    DETS = {"the", "a", "an"}

    def label_phrases(phrases: List[str], tag: str):
        for p in phrases:
            for s, l in find_phrase_spans(tokens, p):
                s2, l2 = s, l
                if l2 > 1 and tokens[s2].lower() in DETS:
                    s2 += 1; l2 -= 1
                if l2 > 0:
                    safe_label_span(labels, s2, l2, tag)

    # Common assignee phrases (receiver/consumer/user roles)
    assignee_phr = [
        "the data recipient", "data recipient", "recipient",
        "the third party", "third party", "the third parties", "third parties",
        "both parties", "each party", "either party", "the parties", "parties",
        "the licensee", "licensee",
        "the end user", "end user",
        "the authorized user", "authorized user",
        "the authorized users", "authorized users",
        "the users", "users", "the user", "user",
        "the data processor", "data processor", "processor",
        "the service provider", "service provider", "service providers",
        "the vendor", "vendor", "vendors",
        "the contractor", "contractor", "contractors",
        "the researcher", "researcher", "researchers",
        "the requester", "requester", "requestor",
        "the applicant", "applicant", "applicants",
        "the partner", "partner", "partners",
        "the data importer", "data importer",
        "the receiving party", "receiving party",
        "the data receiver", "data receiver", "receiver",
    ]
    label_phrases(assignee_phr, "PARTY_ASSIGNEE")

    single_assignee = {
        "recipient","licensee","processor","researcher","contractor","vendor",
        "user","users","requester","requestors","requestor","applicant","partner","partners",
        "importer","receiver","party","parties"
    }
    for i, tok in enumerate(tokens):
        if labels[i] != "O":
            continue
        if tok.lower() in single_assignee:
            labels[i] = "B-PARTY_ASSIGNEE"

    # NP capture after prepositions (assignee only on "to ...")
    DETS = {"the", "a", "an"}
    CONNECTORS = {"of","and","for","in","on","&"}
    PUNCT = {",",".",";","(",")","[","]","{","}","-","—","–",":","/"}

    def is_proper_word(tok: str) -> bool:
        lw = tok.lower()
        return (tok[:1].isupper() or tok.isupper()
                or lw in single_assignee
                or lw in {"police","probation","services","department","departments",
                          "authority","authorities","council","commission","brigade","team","teams",
                          "university","hospital","trust","college","board","ministry","office","service","services",
                          "disclosing","receiving","party","parties","agency","agencies"})

    def label_np_after(j: int, tag: str):
        # skip determiners (not part of party span)
        while j < len(tokens) and tokens[j].lower() in DETS and labels[j] == "O":
            j += 1
        span = []
        while j < len(tokens) and labels[j] == "O":
            t = tokens[j]
            if t in PUNCT: break
            if is_proper_word(t) or t.lower() in CONNECTORS:
                span.append(j); j += 1
            else:
                break
        while span and tokens[span[0]].lower() in CONNECTORS:
            span.pop(0)
        while span and tokens[span[-1]].lower() in CONNECTORS:
            span.pop()
        for k, idx in enumerate(span):
            labels[idx] = ("B-" if k == 0 else "I-") + tag

    # Only "to <NP>" -> PARTY_ASSIGNEE (assignee side)
    for i, t in enumerate(lt):
        if t == "to" and i+1 < len(tokens):
            label_np_after(i+1, "PARTY_ASSIGNEE")

    # "on behalf of <NP>" -> typically the represented assignee
    for i in range(len(tokens) - 3):
        if lt[i] == "on" and lt[i+1] == "behalf" and lt[i+2] == "of":
            label_np_after(i+3, "PARTY_ASSIGNEE")

def seed_purpose(tokens: List[str], labels: List[str]):
    """
    PURPOSE = Defined purpose for exercising an action.
    """
    lt = [t.lower() for t in tokens]
    PUNCT_STOP = {",", ";", ".", ")", "]", "}"}
    DETS = {"the","a","an","this","that","these","those"}

    def end_at_punct(start: int) -> int:
        j = start
        while j < len(tokens) and tokens[j] not in PUNCT_STOP:
            j += 1
        return j

    phrase_list = [
        "for the purpose of", "for the purposes of", "for purposes of",
        "for law enforcement purposes", "for research purposes", "for statistical purposes",
        "for audit purposes", "for monitoring purposes", "for compliance purposes",
        "for educational use", "for internal use", "for commercial use", "for non commercial use",
        "for service delivery", "for safeguarding", "for safeguarding children",
        "for training", "for testing", "for evaluation", "for administrative purposes",
        "for use in", "for use by", "for use of",
        "in order to", "so as to", "so that", "with the purpose of",
        "with the aim of", "with the intention of", "with a view to",
    ]
    for p in phrase_list:
        for s, l in find_phrase_spans(tokens, p):
            e = end_at_punct(s + l)
            safe_label_span(labels, s, e - s, "PURPOSE")

    purpose_heads = {
        "research","education","educational","training","testing","evaluation","audit","monitoring",
        "compliance","enforcement","law","safeguarding","service","services","delivery","operations",
        "administration","security","fraud","prevention","analysis","statistical","statistics",
        "reporting","review","learning","development","improvement","public","interest"
    }
    purpose_use_adjs = {"educational","internal","commercial","non","non-commercial","noncommercial"}
    for i, w in enumerate(lt):
        if w != "for" or labels[i] != "O":
            continue
        j = i + 1
        if j < len(tokens) and lt[j] in DETS:
            j += 1
        k = j
        seen_signal = False
        while k < len(tokens) and tokens[k] not in PUNCT_STOP:
            lw = lt[k]
            if lw in purpose_heads or lw.endswith("ing"):
                seen_signal = True
                k += 1
                continue
            if lw in purpose_use_adjs:
                k2 = k + 1
                if k2 < len(tokens) and tokens[k2] == "-":
                    k2 += 1
                if k2 < len(tokens) and lt[k2] == "use":
                    seen_signal = True
                    k = k2 + 1
                    continue
            if lw == "use" and k + 1 < len(tokens) and lt[k+1] in {"in","by","of"}:
                seen_signal = True
                k += 2
                continue
            k += 1
        if seen_signal:
            e = end_at_punct(j)
            safe_label_span(labels, i, e - i, "PURPOSE")

    intent_verbs = {
        "enable","ensure","prevent","promote","support","facilitate",
        "comply","investigate","detect","prosecute","protect","safeguard",
        "improve","monitor","evaluate","assess","administer","deliver","provide",
        "process","share","use","inform","analyze","analyse","review","enforce"
    }
    NOT_PURPOSE_PREV = {
        "agree","agrees","agreed","agreeing",
        "undertake","undertakes","undertook","undertaking",
        "required","require","requires","obligated","obliged","obligation","responsible","duty",
        "right","consent","permission","authorize","authorise","authorized","authorised"
    }
    for i in range(len(tokens) - 1):
        if lt[i] != "to" or labels[i] != "O":
            continue
        if i >= 1 and lt[i-1] in NOT_PURPOSE_PREV:
            continue
        prev_ok = (
            i == 0 or tokens[i-1] in {",",";","("} or
            (i >= 2 and lt[i-2] == "in" and lt[i-1] == "order") or
            (i >= 2 and lt[i-2] == "so" and lt[i-1] == "as")
        )
        if prev_ok or lt[i+1] in intent_verbs:
            e = end_at_punct(i + 2)
            safe_label_span(labels, i, e - i, "PURPOSE")

def seed_constraints(tokens: List[str], labels: List[str]):
    """
    CONSTRAINT seeder aligned with ODRL constraints.
    """
    lt = [t.lower() for t in tokens]
    PUNCT_STOP = {",", ";", ".", ")", "]", "}"}

    def span_to_punct(start_idx: int) -> int:
        j = start_idx
        while j < len(tokens) and tokens[j] not in PUNCT_STOP:
            j += 1
        return j

    starter_phrases = [
        "without", "except", "solely",
        "pursuant to", "in accordance with", "in compliance with",
        "under", "upon", "prior to", "before", "after", "during", "within",
        "no later than", "no earlier than", "on or before", "on or after",
        "up to", "at least", "at most", "no more than", "not more than",
        "not less than", "more than", "fewer than",
        "in writing",
        "as permitted", "as required", "as defined", "as set forth", "as applicable",
    ]
    for phrase in starter_phrases:
        for s, l in find_phrase_spans(tokens, phrase):
            e = span_to_punct(s + l)
            safe_label_span(labels, s, e - s, "CONSTRAINT")

    for i, w in enumerate(lt):
        if w == "between" and labels[i] == "O":
            j = i + 1
            has_and = False
            while j < len(tokens) and tokens[j] not in PUNCT_STOP:
                if lt[j] == "and":
                    has_and = True
                j += 1
            if has_and:
                safe_label_span(labels, i, j - i, "CONSTRAINT")

    PER_UNITS = {
        "day","days","month","months","year","years","week","weeks","times",
        "user","users","device","devices","account","accounts",
        "request","requests","disclosure","disclosures","date","period"
    }
    for i, w in enumerate(lt):
        if w == "per" and labels[i] == "O" and i + 1 < len(tokens):
            if lt[i+1] in PER_UNITS and labels[i+1] == "O":
                safe_label_span(labels, i, 2, "CONSTRAINT")

    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{2,4}$|^\d{1,2}:\d{2}([ap]m)?$", re.I)
    MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
    for i, w in enumerate(lt):
        if w == "by" and labels[i] == "O" and i + 1 < len(tokens):
            nxt = tokens[i+1]
            nxtl = lt[i+1]
            looks_date = bool(DATE_RE.match(nxt)) or (nxtl in MONTHS)
            if looks_date:
                e = span_to_punct(i + 2)
                safe_label_span(labels, i, e - i, "CONSTRAINT")

    comp_ops = [
        "at least", "at most", "no more than", "not more than", "not less than", "no less than",
        "less than", "less than or equal to", "greater than", "greater than or equal to",
        "more than", "equal to", "equals", "not equal to",
        "before", "after", "on or before", "on or after", "no later than", "no earlier than",
        "within", "by"
    ]
    PCT = {"%","percent","per","percentage"}
    CURR = {"$","£","€","usd","gbp","eur"}
    TIME_UNITS = {"second","seconds","minute","minutes","hour","hours","day","days","week","weeks","month","months","year","years","business","working"}

    def looks_numeric_chunk(idx: int) -> bool:
        if idx >= len(tokens): return False
        t = tokens[idx]; l = lt[idx]
        if t.isdigit(): return True
        if any(ch.isdigit() for ch in t): return True
        if l in PCT or l in CURR: return True
        if l in TIME_UNITS: return True
        return False

    for phrase in comp_ops:
        for s, l in find_phrase_spans(tokens, phrase):
            j = s + l
            k = j
            consumed = False
            while k < len(tokens) and tokens[k] not in PUNCT_STOP and (looks_numeric_chunk(k) or lt[k] in {"of","and","or","to","the","a","an","for","in"}):
                k += 1
                consumed = True
            if not consumed:
                k = span_to_punct(j)
            safe_label_span(labels, s, k - s, "CONSTRAINT")

    left_operands = [
        "absolute position","absolute size","absolute spatial position","absolute temporal position",
        "relative position","relative size","relative spatial position","relative temporal position",
        "count","date time","delay period","elapsed time","time interval","unit of count",
        "delivery channel","device","system","system device",
        "event","industry","language","media","file format","resolution",
        "pay amount","percentage","product","purpose","recipient","version","virtual location",
        "spatial","spatial coordinates",
    ]
    op_phr = [
        "is", "are", "be", "must be", "shall be", "equals", "equal to", "not equal to",
        "less than", "less than or equal to", "greater than", "greater than or equal to",
        "is any of", "is all of", "is none of", "is a", "is part of", "has part",
        "before", "after", "during", "within", "by", "between"
    ]

    MAX_LOOKAHEAD = 14
    for lop in left_operands:
        for s, l in find_phrase_spans(tokens, lop):
            if any(labels[s+i] != "O" for i in range(l)):
                continue
            end_search = min(len(tokens), s + l + MAX_LOOKAHEAD)
            op_hit = None
            for op in op_phr:
                for os, ol in find_phrase_spans(tokens[s+l:end_search], op):
                    op_hit = (s + l + os, ol)
                    break
                if op_hit: break
            if op_hit:
                op_s, op_l = op_hit
                e = span_to_punct(op_s + op_l)
                safe_label_span(labels, s, e - s, "CONSTRAINT")
            else:
                e = span_to_punct(s + l)
                safe_label_span(labels, s, e - s, "CONSTRAINT")

    # Ensure "in writing" gets captured even if missed above
    for i in range(len(tokens) - 1):
        if lt[i] == "in" and lt[i+1] == "writing":
            if labels[i] == "O": labels[i] = "B-CONSTRAINT"
            if labels[i+1] == "O": labels[i+1] = "I-CONSTRAINT"

# ------------------ pipeline ------------------

def has_non_O(labels: List[str]) -> bool:
    return any(l != "O" for l in labels)

def seed_label(tokens: List[str]) -> List[str]:
    labels = ["O"] * len(tokens)
    seed_parties(tokens, labels)       # assignee only
    seed_assets(tokens, labels)
    seed_rule_type(tokens, labels)     # includes duty-like markers as RULE_TYPE
    seed_actions(tokens, labels)
    seed_purpose(tokens, labels)
    seed_constraints(tokens, labels)
    return labels

# ------------------ few-shot loading (optional) ------------------

def load_fewshot_examples(path: str, max_items: int) -> List[Dict[str, Any]]:
    examples = []
    if not path or not os.path.exists(path):
        return examples
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            obj = json.loads(ln)
            toks = tokenize(obj["text"])
            lab = obj.get("label", [])
            if len(toks) != len(lab):
                logging.info(f"[skip few-shot] mismatch: tokens={len(toks)} labels={len(lab)}")
                continue
            examples.append({"tokens": toks, "label": lab})
            if len(examples) >= max_items:
                break
    logging.info(f"[few-shot] loaded {len(examples)} examples")
    return examples

FEWSHOT = load_fewshot_examples(FEWSHOT_JSONL, MAX_FEWSHOT)

def build_prompt(tokens: List[str], seed_labels_: List[str]) -> List[Dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]

    for ex in FEWSHOT:
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": (
                    "EXAMPLE — Correct seed BIO labels for tokens.\n"
                    f"tokens = {json.dumps(ex['tokens'], ensure_ascii=False)}\n"
                    f"seed_labels = {json.dumps(ex['label'], ensure_ascii=False)}"
                 )}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "output_text",
                 "text": json.dumps({"label": ex["label"]}, ensure_ascii=False)}
            ]
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text",
             "text": (
                "Correct and improve the seed BIO labels. "
                "Return JSON only, per schema.\n"
                f"tokens = {json.dumps(tokens, ensure_ascii=False)}\n"
                f"seed_labels = {json.dumps(seed_labels_, ensure_ascii=False)}"
             )}
        ]
    })
    return messages

def extract_labels_from_response(resp) -> List[str]:
    try:
        content_item = resp.output[0].content[0]
    except Exception:
        maybe_text = getattr(resp, "output_text", None) or getattr(resp, "text", None)
        if maybe_text:
            return json.loads(maybe_text)["label"]
        raise
    data = getattr(content_item, "json", None)
    if data is None:
        txt = getattr(content_item, "text", "")
        data = json.loads(txt)
    return data["label"]

@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(OpenAIError),
)
async def call_model(tokens: List[str], seed_labels_: List[str]) -> List[str]:
    resp = await client.responses.create(
        model=MODEL,
        input=build_prompt(tokens, seed_labels_),
        response_format=RESPONSE_FORMAT,
        temperature=0
    )
    return extract_labels_from_response(resp)

async def repair_length(tokens: List[str], got: List[str]) -> List[str]:
    msg = [
        {"role": "system",
         "content": "Output strictly one BIO tag per token. Return JSON only per schema."},
        {"role": "user",
         "content": [
             {"type": "input_text",
              "text": (
                  f"Length mismatch: {len(got)} labels for {len(tokens)} tokens.\n"
                  f"tokens = {json.dumps(tokens, ensure_ascii=False)}\n"
                  "Respond with JSON {'label':[...]} only."
              )}
         ]}
    ]
    resp = await client.responses.create(
        model=MODEL,
        input=msg,
        response_format=RESPONSE_FORMAT,
        temperature=0
    )
    return extract_labels_from_response(resp)

async def repair_all_O(tokens: List[str], seed_labels_: List[str]) -> List[str]:
    msg = [
        {"role": "system",
         "content": ("Do NOT output all 'O' when tokens contain annotable spans. "
                     "Return JSON only per schema.")},
        {"role": "user",
         "content": [
             {"type": "input_text",
              "text": (
                 "Your previous output was all 'O'. Re-label correctly.\n"
                 f"tokens = {json.dumps(tokens, ensure_ascii=False)}\n"
                 f"seed_labels = {json.dumps(seed_labels_, ensure_ascii=False)}\n"
                 "Keep correct seed tags; fix incorrect ones; add missing spans. "
                 "Punctuation remains 'O'. Respond with JSON {'label':[...]} only."
              )}
         ]}
    ]
    resp = await client.responses.create(
        model=MODEL,
        input=msg,
        response_format=RESPONSE_FORMAT,
        temperature=0
    )
    return extract_labels_from_response(resp)

# ------------------ driver ------------------

@dataclass
class Unit:
    idx: int
    text: str
    tokens: List[str]
    seed: List[str]

async def annotate_one(u: Unit, max_repairs: int = 2) -> Dict[str, Any]:
    tokens = u.tokens
    seed = u.seed
    try:
        labels = await call_model(tokens, seed)

        repairs = 0
        while len(labels) != len(tokens) and repairs < max_repairs:
            labels = await repair_length(tokens, labels)
            repairs += 1

        if set(labels) == {"O"} and has_non_O(seed):
            labels = await repair_all_O(tokens, seed)
            if len(labels) != len(tokens):
                labels = await repair_length(tokens, labels)

        if len(labels) != len(tokens):
            if len(labels) < len(tokens):
                labels += ["O"] * (len(tokens) - len(labels))
            else:
                labels = labels[:len(tokens)]

        if set(labels) == {"O"} and has_non_O(seed):
            labels = seed

    except Exception:
        labels = seed if has_non_O(seed) else ["O"] * len(tokens)

    return {"idx": u.idx, "text": u.text, "label": labels}

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

async def main():
    lines = load_lines(INPUT_TXT)

    units = []
    for i, t in enumerate(lines):
        toks = tokenize(t)
        seed = seed_label(toks)
        units.append(Unit(i, t, toks, seed))

    sem = asyncio.Semaphore(WORKERS)
    results: List[Optional[Dict[str, Any]]] = [None] * len(units)

    async def bound(u: Unit):
        async with sem:
            res = await annotate_one(u)
            results[u.idx] = res

    tasks = [asyncio.create_task(bound(u)) for u in units]
    for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Labeling"):
        await _

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for res in results:
            obj = {"text": res["text"], "label": res["label"]}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    total = len(results)
    allO = sum(1 for r in results if set(r["label"]) == {"O"})
    print(f"Done. Wrote {total} items -> {OUTPUT_JSONL} | All-'O' lines: {allO}")

if __name__ == "__main__":
    asyncio.run(main())
