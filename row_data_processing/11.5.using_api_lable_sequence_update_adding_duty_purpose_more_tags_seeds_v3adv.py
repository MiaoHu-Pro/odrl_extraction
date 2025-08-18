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

API_KEY       = os.getenv("OPENAI_API_KEY")
MODEL         = os.getenv("MODEL_NAME", "o4-mini")
WORKERS       = int(os.getenv("WORKERS", "6"))
INPUT_TXT     = os.getenv("INPUT_TXT", "data_sharing_agreement_mt20_examples.txt")
OUTPUT_JSONL  = os.getenv("OUTPUT_JSONL", "data_sharing_agreement_mt20_examples_labels.jsonl")
FEWSHOT_JSONL = os.getenv("FEWSHOT_JSONL", "")      # optional path to few-shot examples JSONL
MAX_FEWSHOT   = int(os.getenv("MAX_FEWSHOT", "20"))  # max few-shot examples to load
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", os.getenv("WORD_EMBEDDING_PATH", "bert-base-cased"))

if not API_KEY:
    raise SystemExit("OPENAI_API_KEY not found in environment/.env")

logging.basicConfig(level=logging.INFO, format="%(message)s")

client = AsyncOpenAI(api_key=API_KEY)

# ------------------ tokenizer (HF-based, BIO-friendly) ------------------

HF_TOK = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

# at top of your labeling script
USE_MODEL_PRETOKENIZATION = True  # set False to use model-agnostic regex

def tokenize(text: str) -> list[str]:
    if USE_MODEL_PRETOKENIZATION:
        # Prefer HF pre-tokenization (works for BERT/DeBERTa/RoBERTa and T5 fast)
        if hasattr(HF_TOK, "basic_tokenizer") and HF_TOK.basic_tokenizer is not None:
            return HF_TOK.basic_tokenizer.tokenize(
                text, never_split=getattr(HF_TOK, "all_special_tokens", [])
            )
        if getattr(HF_TOK, "is_fast", False) and hasattr(HF_TOK, "backend_tokenizer"):
            pre = HF_TOK.backend_tokenizer.pre_tokenizer
            if pre is not None:
                return [t for t, _ in pre.pre_tokenize_str(text)]
    # Model-agnostic fallback: split words & punctuation (keeps 133-155 -> 133 '-' 155)
    import re
    return re.findall(r"[A-Za-z0-9#]+|[“”\"'()\-–—:;,\.]", text)


# ------------------ schema & instructions ------------------

LABEL_ENUM = [
   "O",
   "B-PARTY_ASSIGNEE", "I-PARTY_ASSIGNEE",
   "B-PARTY_ASSIGNER", "I-PARTY_ASSIGNER",
   "B-ASSET", "I-ASSET",
   "B-ACTION", "I-ACTION",
   "B-RULE_TYPE", "I-RULE_TYPE",
   "B-CONSTRAINT", "I-CONSTRAINT",
   "B-DUTY", "I-DUTY",
   "B-PURPOSE", "I-PURPOSE",
]

# SYSTEM_INSTRUCTIONS = """You are an expert BIO annotator for Data Sharing Agreements.
# You will receive:
# - tokens: the exact token sequence (punctuation kept as separate tokens),
# - seed_labels: heuristic BIO tags to START from.
#
# Task:
# - Correct and improve seed_labels to produce final BIO tags.
# - Output EXACTLY one tag per token, same order and length.
# - Punctuation tokens (e.g., ',', '.', '(', ')', '-', ':', ';', quotes) must be 'O'.
# - Include negation with RULE_TYPE (e.g., 'will not' -> B-RULE_TYPE I-RULE_TYPE).
# - Each head verb in coordination is ACTION (e.g., 'share, publish, or release').
# - ASSET is the governed noun phrase;
# - CONSTRAINT covers conditions ('without prior approval ...', 'solely for ...', etc).
# - PARTY_ASSIGNEE/ASSIGNER are actor mentions per role; keep minimal spans via BIO.
# - DUTY: obligations introduced by connectors like 'provided that', 'as long as', 'subject to', 'only if', or phrases like 'be subject to the terms and conditions', 'be provided in writing', etc.
# - PURPOSE: purposes introduced by 'for ... purpose(s)', 'for law enforcement purposes', 'in order to ...', or infinitival purpose 'to <verb>' stating intent, etc.
#
# Important:
# - Do NOT output all 'O' when seed_labels contain non-'O' unless those non-'O' are clearly wrong.
# - If a seed label looks correct, keep it; otherwise fix it..
# - Keep correct seed tags; fix incorrect ones; add missing spans.
# - The seed tags I provided are narrow; you can label the words to the corresponding tag if the word represents the meaning of the tag.
# Return JSON ONLY per the schema.
# """

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
- ACTION: the head verb(s) denoting an activity or permission (e.g., share, publish, distribute, copy, use, grant use, commercial use, share alike, text to speech). 
  * In coordinated lists, tag each head verb (e.g., “share, publish, or release”).
  * For common multiword actions, tag the phrase (e.g., “grant use”, “commercial use”, “share alike”, “text to speech”, “ad hoc share”, “attach policy”, “review policy”, “obtain consent”, “ensure exclusivity”, “source code”, “extract page/word/char”).
  * Auxiliaries like “will/shall/must” are RULE_TYPE, not ACTION. In passives like “be released”, tag “be released” as ACTION.
- ASSET: The Asset entity can be any form of identifiable resource, such as data/information, content/media, applications, or services. Furthermore, it can be used to represent other Asset entities that are needed to undertake the Policy expression, such as with the Duty entity. The governed resource/object (e.g., “data”, “personal data”, “results of data analysis”, “names of individuals”, “addresses”, “information”, “maps”, “source code” when it is the object). 
- RULE_TYPE: words/phrases that express permission/prohibition/obligation (e.g., “shall”, “must”, “may”, “will not”, “shall not”, “nor will”).
  * Include negation with RULE_TYPE (e.g., “will not” → B-RULE_TYPE I-RULE_TYPE).
- CONSTRAINT: A boolean expression that refines the semantics of an Action and Party/Asset Collection or declare the conditions applicable to a Rule. conditions/limits (time, count, purpose-like qualifiers when expressed as constraints, location, approvals) introduced by markers like “without…”, “except…”, “solely…”, “pursuant…”, “under…”, “upon…”, “before…”, “as…”, “in writing”.
- DUTY: additional obligations introduced by connectors such as “provided that…”, “as long as…”, “only if…”, “subject to…”, or formulae like “be provided in writing”, “be subject to the terms and conditions…”.
- PURPOSE: purposive clauses introduced by “for … purpose(s)”, “for law enforcement purposes”, “in order to …”, or infinitival purpose “to <verb> …” when it states intent (e.g., “to enable/ensure/prevent/promote/comply…”).
- PARTY_ASSIGNEE/ASSIGNER are Party mentions per role. The Party entity could be a person, group of people, organisation, or agent. An agent is a person or thing that takes an active role or produces a specified effect

Precedence (resolve conflicts and avoid double-labeling):
1) RULE_TYPE applies only to the modal/negation tokens; adjacent verbs remain ACTION or O per context.
2) If a span could be PURPOSE vs ACTION (e.g., “to ensure compliance”), prefer PURPOSE for the “to …” clause; do NOT also tag the head verb inside as ACTION.
3) If a span could be DUTY vs CONSTRAINT (e.g., “subject to …”), prefer DUTY.
4) If a token could be ASSET vs ACTION, use context: verbs → ACTION; noun phrases → ASSET.

Other details:
- Numbers, section references, and symbols are 'O' unless they are part of a clear CONSTRAINT (e.g., “condition # 2 before …”: “before …” is CONSTRAINT; '#' stays 'O').
- Keep spans minimal and semantically tight (no trailing punctuation).

Important:
- Do NOT output all 'O' when seed_labels contain valid non-'O' spans unless those are clearly wrong.
- If a seed label looks correct, keep it; otherwise fix it.
- Keep correct seed tags; fix incorrect ones; add missing spans.
- The seed tags are narrow; you may add valid spans when tokens clearly express one of the label meanings.

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

# ------------------ RULE-BASED SEED TAGGER (lightweight) ------------------

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

def seed_rule_type(tokens: List[str], labels: List[str]):
    """
    RULE_TYPE seeder with:
      - month 'May' disambiguation,
      - expanded single_markers derived from operator phrases (obligation/permission/prohibition),
      - modal+NOT (incl. 'can not', 'cannot', simple contractions),
      - will NOT overwrite non-'O' tags.
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

    # simple one-token contractions (tokenizer dependent)
    CONTRACTIONS = {"can't","won't","shan't","shouldn't","mustn't","couldn't","wouldn't","mightn't","mayn't"}
    for i, w in enumerate(lt):
        if w in CONTRACTIONS and labels[i] == "O":
            labels[i] = "B-RULE_TYPE"

    # -------- phrase catalogs --------
    obligation_phrases = [
        "required to", "is required to", "are required to",
        "is obligated to", "are obligated to",
        "is obliged to", "are obliged to",
        "is bound to", "are bound to",
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
        "allow to", "allows to", "allowed to",     # active allow-patterns
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

    # Label phrase spans (operator words only)
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

    # -------- extended single markers (derived from phrases) --------
    # Base single words we already treated
    base_single = {
        # obligation/necessity/expectation
        "must","shall","will","should","needs","need","mandatory","compulsory",
        # permission
        "may","can","could","might","would",
        # allow family
        "allow","allows","allowed","allowing",
        # prohibition adjectives
        "forbidden","prohibited","restricted","barred",
        # normative nouns
        "duty","obligation","responsibility",
    }
    # Extra heads pulled from phrases (plus common inflections)
    extra_single = {
        # obligation heads
        "required","require","requires","requiring",
        "obligated","obligate","obligates","obligating",
        "obliged","oblige","obliges","obliging",
        "bound","bind","binds","binding",
        "expected","expect","expects","expecting",
        "assigned","assign","assigns","assigning",
        "charged","charge","charges","charging",
        "tasked","task","tasks","tasking",
        "responsible","responsibility",
        "commitment","committed","commit","commits","committing",
        "responsible",  # keep adj form
        # permission heads
        "permitted","permit","permits","permitting",
        "authorized","authorised","authorize","authorise","authorizes","authorises","authorizing","authorising",
        "entitled","entitle","entitles","entitling",
        "enabled","enable","enables","enabling",
        # (avoid 'free' alone as too ambiguous; it’s handled in phrases)
        "right",  # cautious: often appears in 'right to'; still useful as seed
        "granted","grant","grants","granting",  # to support 'granted leave to'
        "approval","consent",  # nouns used in permission context ('subject to approval', 'upon consent')
        # prohibition heads
        "prohibit","prohibits","prohibiting",
        "forbid","forbids","forbidding",
        "bar","bars","barring",
        "restrict","restricts","restricting",
        "refrain","refrains","refrained","refraining",
    }

    single_markers = base_single | extra_single

    # Apply single markers (with 'May' month disambiguation)
    for i, w in enumerate(lt):
        if labels[i] != "O":
            continue
        # skip 'May' as month
        if tokens[i] in {"May","MAY"} and _is_may_month(i):
            continue
        if w in single_markers:
            labels[i] = "B-RULE_TYPE"

    # -------- plain modals as fallback --------
    plain_modals = {"shall","must","will","should","may","can","could","might","would"}
    for i, w in enumerate(lt):
        if labels[i] != "O":
            continue
        if tokens[i] in {"May","MAY"} and _is_may_month(i):
            continue
        if w in plain_modals:
            labels[i] = "B-RULE_TYPE"

def seed_actions(tokens: List[str], labels: List[str]):
    """
    ACTION seeder with modal+infinitive support.

    New: If a modal (can/cannot/can not/may/might/must/shall/should/will/would/could)
    precedes a verb/phrase, label that verb (or phrase like 'be returned', 'grant use', etc.)
    as ACTION (without touching the modal itself, which is typically RULE_TYPE).
    """

    # --- Multi-token phrases (lower-cased) ---
    phrase_list = [
        # your originals
        "granted access", "be returned", "be released", "be utilized", "be used",
        # ODRL-style phrases
        "commercial use", "commerical use", "concurrent use", "secondary use", "grant use",
        "attach policy", "attach source", "review policy", "next policy",
        "obtain consent", "ensure exclusivity", "source code",
        "write to", "append to",
        "share alike", "text to speech", "ad hoc share",
        "extract char", "extract page", "extract word",
        "derivative works", "accept tracking",
        # explicit hyphen variants for tokenizers that split '-' as a token
        "text-to-speech", "ad-hoc share", "share-alike",
    ]

    # Label phrase spans first (reserve them)
    for p in phrase_list:
        for s, l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "ACTION")

    # --- Single-token actions (normalized) ---
    base_actions = {
        # originals
        "release","disclose","share","publish","present","remain","return",
        "comply","utilize","utilized","facilitate","agree","handle","handling",
        "provide","assist","assistance","download",
        # ODRL-like singles
        # "attribution","distribution","notice","reproduction","sharing",
        # "aggregate","annotate","anonymize","append","archive","attribute",
        # "commercialize","compensate","copy","delete","derive","digitize",
        # "display","distribute","execute","export","extract","give","include",
        # "index","inform","install","lease","lend","license","modify","move",
        # "pay","play","preview","print","read","reproduce","sell","stream",
        # "synchronize","transfer","transform","translate","uninstall","use",
        # "watermark","write",

         "notice",
        "aggregate", "annotate", "anonymize", "append", "archive", "attribute",
        "commercialize", "compensate", "copy", "delete", "derive", "digitize",
        "display", "distribute", "execute", "export", "extract", "give", "include",
        "index", "inform", "install", "lease", "lend", "license", "modify", "move",
        "pay", "play", "preview", "print", "read", "reproduce", "sell", "stream",
        "synchronize", "transfer", "transform", "translate", "uninstall", "use",
        "watermark", "write",
    }

    def _norm(tok: str) -> str:
        # lowercase and collapse underscores (hyphens are separate tokens already)
        return tok.lower().replace("_", "-")

    def _morph_forms(n: str) -> set:
        """
        Light morphology:
          - used/using/uses -> use
          - copies -> copy; distributes -> distribute
          - shared -> share (handle base+e)
        """
        forms = {n}
        # if n.endswith("ing") and len(n) > 4:
        #     forms.add(n[:-3])          # using -> us
        #     forms.add(n[:-3] + "e")    # using -> use
        # if n.endswith("ed") and len(n) > 3:
        #     forms.add(n[:-2])          # shared -> shar
        #     forms.add(n[:-2] + "e")    # shared -> share
        if n.endswith("ies") and len(n) > 4:
            forms.add(n[:-3] + "y")    # copies -> copy
        if n.endswith("es") and len(n) > 3:
            forms.add(n[:-2])          # distributes -> distribut
            forms.add(n[:-2] + "e")    # distributes -> distribute; uses -> use
        if n.endswith("s") and len(n) > 3:
            forms.add(n[:-1])          # shares -> share(sans e)
            forms.add(n[:-1] + "e")    # shares -> share
        return forms

    def _is_action_like(tok: str) -> bool:
        n = _norm(tok)
        return any(f in base_actions for f in _morph_forms(n))

    # helper: try to label a phrase that STARTS at position pos
    def _label_phrase_at(pos: int) -> bool:
        for p in phrase_list:
            for s, l in find_phrase_spans(tokens, p):
                if s == pos:
                    if safe_label_span(labels, s, l, "ACTION"):
                        return True
        return False

    # --- Modal + infinitive support ---
    MODALS = {"can","may","might","must","shall","should","will","would","could","cannot"}
    NOT_TOKENS = {"not","n't","n’t"}  # handle can't -> "can", "'", "t" in some tokenizers separately

    i = 0
    while i < len(tokens):
        t = tokens[i].lower()
        if t in MODALS:
            # find the first verb token position after optional negator
            j = i + 1
            # treat 'cannot' as having implicit 'not'
            if t != "cannot" and j < len(tokens) and tokens[j].lower() in NOT_TOKENS:
                j += 1
            # also skip stray apostrophe tokens in "can ' t"
            if j < len(tokens) and tokens[j] in {"'", "’"} and j + 1 < len(tokens) and tokens[j+1].lower() == "t":
                j += 2

            if j < len(tokens):
                # passive phrase like "be returned"/"be used" etc.
                if tokens[j].lower() == "be":
                    if _label_phrase_at(j):  # e.g., 'be returned'
                        i = j + 1
                        continue
                    # otherwise try to label the next word (past participle) if action-like
                    if j + 1 < len(tokens) and labels[j+1] == "O" and _is_action_like(tokens[j+1]):
                        labels[j+1] = "B-ACTION"
                        i = j + 2
                        continue

                # try multi-word action phrase at the verb position (grant use, attach policy, etc.)
                if _label_phrase_at(j):
                    i = j + 1
                    continue

                # single verb right after modal
                if labels[j] == "O" and _is_action_like(tokens[j]):
                    labels[j] = "B-ACTION"
                    i = j + 1
                    continue
        i += 1

    # --- Singles pass (labels any remaining standalone verbs) ---
    for i, tok in enumerate(tokens):
        if labels[i] != "O":
            continue
        if _is_action_like(tok):
            labels[i] = "B-ACTION"


# def seed_assets(tokens: List[str], labels: List[str]):
#     phrases = [
#         "results of data analysis", "findings or conclusions",
#         "names of individuals", "individual addresses",
#         "personal data breach", "personal data",
#         "information", "maps", "addresses", "data",
#         "confidential report"
#     ]
#     for p in phrases:
#         for s,l in find_phrase_spans(tokens, p):
#             safe_label_span(labels, s, l, "ASSET")

def seed_assets(tokens: List[str], labels: List[str]):
    """
    Expanded ASSET seeder for DSA/ODRL:
    1) Reserve known multi-word phrases (incl. plural/singular).
    2) Label modifier(+modifier ...)+head patterns (e.g., 'special category personal data').
       Handles hyphens since your tokenizer emits '-' as its own token.
    3) Label safe single-word heads (generic asset nouns).
    Never overwrites existing non-'O' tags.
    """
    # 1) Multi-token phrases first (reserve spans)
    phrase_list = [
        # analysis/results
        "results of data analysis", "results of the data analysis", "results of analysis",
        "findings or conclusions", "findings and conclusions", "findings of the analysis",
        "statistical outputs", "statistical output",
        # personal/special/criminal etc.
        "personal data", "special category data", "special category personal data",
        "criminal data", "biometric data", "genetic data", "health data",
        "financial data", "payment information", "payment data",
        "contact information", "contact details",
        "location data", "geolocation data", "shared personal data",
        "ip address", "ip addresses", "email address", "email addresses",
        "phone number", "phone numbers", "names of individuals",
        "individual addresses", "identity of individuals", "identity of an individual",
        # artifacts / files / media / code
        "source code", "training data", "test data", "research data",
        "documents", "document", "files", "file",
        "images", "image", "photographs", "videos", "video",
        "audio recordings", "audio", "transcripts", "logs", "audit logs", "metadata",
        # maps/tables/etc.
        "maps", "map", "tables", "table",
        # confidentiality
        "confidential information", "confidential report",
    ]
    for p in phrase_list:
        for s, l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "ASSET")

    # 2) Modifier + head patterns (e.g., "special category personal data", "payment information")
    #    - allow chains of modifiers and embedded hyphens before a head noun.
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
        "log","logs","metadata","code"  # 'code' usually with 'source', but allow as head if not risky
    }

    def is_mod(tok: str) -> bool:
        t = tok.lower()
        return t in modifiers

    def is_head(tok: str) -> bool:
        return tok.lower() in heads

    i = 0
    while i < len(tokens) - 1:
        if labels[i] != "O":
            i += 1
            continue

        # accumulate modifiers (and hyphens) until a head noun
        j = i
        chain_idxs = []
        saw_mod = False
        while j < len(tokens) and labels[j] == "O":
            tj = tokens[j]
            if tj == "-":  # allow hyphen separators in the modifier chain
                chain_idxs.append(j)
                j += 1
                continue
            if is_mod(tj):
                chain_idxs.append(j)
                saw_mod = True
                j += 1
                continue
            break

        # if after mods we have a head, label chain + head as ASSET
        if saw_mod and j < len(tokens) and labels[j] == "O" and is_head(tokens[j]):
            span = chain_idxs + [j]
            # remove leading/trailing hyphens from the span if any
            span = [k for k in span if tokens[k] != "-"]
            if span:
                for idx_pos, idx in enumerate(span):
                    labels[idx] = "B-ASSET" if idx_pos == 0 else "I-ASSET"
                i = j + 1
                continue

        i += 1

    # 3) Safe single-word heads (only if still 'O')
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
    lt = [t.lower() for t in tokens]

    DETS = {"the", "a", "an"}  # determiners we do NOT want in party spans

    # --- helper: label all matches of phrase strings (space-separated), trimming leading determiner ---
    def label_phrases(phrases: List[str], tag: str):
        for p in phrases:
            for s, l in find_phrase_spans(tokens, p):
                s2, l2 = s, l
                # trim leading determiner (e.g., "the Data Recipient" -> start at "Data")
                if l2 > 1 and tokens[s2].lower() in DETS:
                    s2 += 1
                    l2 -= 1
                if l2 > 0:
                    safe_label_span(labels, s2, l2, tag)

    # Common assignee phrases (receives rights / data)
    assignee_phr = [
        # with & without determiners (we'll trim if a determiner is present)
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

    # Common assigner phrases (grants/discloses)
    assigner_phr = [
        "the data provider", "data provider",
        "the data owner", "data owner", "owner",
        "the licensor", "licensor",
        "the discloser", "discloser",
        "the disclosing party", "disclosing party",
        "the data discloser", "data discloser",
        "the data controller", "data controller", "controller",
        "the authority", "authority", "the authorities", "authorities",
        "the competent authority", "competent authority",
        "the competent authorities", "competent authorities",
        "the government department", "government department",
        "the government departments", "government departments",
        "the public authority", "public authority",
        "the public authorities", "public authorities",
        "the police", "police",
        "the probation services", "probation services",
        "the youth offending team", "youth offending team",
        "the youth offending teams", "youth offending teams",
        "the london fire brigade", "london fire brigade",
        "the public protection desk", "public protection desk", "ppd",
        "the data exporter", "data exporter", "exporter",
    ]

    label_phrases(assignee_phr, "PARTY_ASSIGNEE")
    label_phrases(assigner_phr, "PARTY_ASSIGNER")

    # --- single-word role nouns (only if still 'O') ---
    single_assignee = {
        "recipient","licensee","processor","researcher","contractor","vendor",
        "user","users","requester","requestors","requestor","applicant","partner","partners",
        "importer","receiver"
    }
    single_assigner = {
        "provider","owner","licensor","discloser","controller","exporter",
        "authority","authorities","commission","commissioner","council","agency","agencies",
        "police","department","departments","brigade"
    }
    for i, tok in enumerate(tokens):
        if labels[i] != "O":
            continue
        w = tok.lower()
        if w in single_assignee:
            labels[i] = "B-PARTY_ASSIGNEE"
        elif w in single_assigner:
            labels[i] = "B-PARTY_ASSIGNER"

    # --- hyphenated defined-terms fallback: "Data - Discloser", "Data - Receiver" (skip any leading 'the') ---
    role_heads_assigner = {"discloser","exporter","provider","controller","licensor","owner"}
    role_heads_assignee = {"receiver","recipient","importer","processor","licensee"}
    i = 0
    while i < len(tokens) - 2:
        # skip a leading determiner if present
        start = i + 1 if tokens[i].lower() in DETS else i
        if (start + 2 < len(tokens)
            and tokens[start].lower() == "data"
            and tokens[start+1] == "-"
            and tokens[start+2].lower() in (role_heads_assigner | role_heads_assignee)):
            tag = "PARTY_ASSIGNER" if tokens[start+2].lower() in role_heads_assigner else "PARTY_ASSIGNEE"
            safe_label_span(labels, start, 3, tag)
            i = start + 3
        else:
            i += 1

    # --- NP capture after prepositions (already skips determiners) ---
    CONNECTORS = {"of","and","for","in","on","&"}
    PUNCT = {",",".",";","(",")","[","]","{","}","-","—","–",":","/"}

    def is_proper_word(tok: str) -> bool:
        lw = tok.lower()
        return (tok[:1].isupper() or tok.isupper()
                or lw in single_assigner or lw in single_assignee
                or lw in {"police","probation","services","department","departments",
                          "authority","authorities","council","commission","brigade","team","teams",
                          "university","hospital","trust","college","board","ministry","office","service","services",
                          "disclosing","receiving","party"})

    def label_np_after(j: int, tag: str):
        # skip determiners (policy: determiners are not part of the party span)
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
        # trim leading/trailing connectors
        while span and tokens[span[0]].lower() in CONNECTORS:
            span.pop(0)
        while span and tokens[span[-1]].lower() in CONNECTORS:
            span.pop()
        for k, idx in enumerate(span):
            labels[idx] = ("B-" if k == 0 else "I-") + tag

    # by/from -> assigner ; to -> assignee
    for i, t in enumerate(lt):
        if t == "by" and i+1 < len(tokens):
            label_np_after(i+1, "PARTY_ASSIGNER")
        elif t == "from" and i+1 < len(tokens):
            label_np_after(i+1, "PARTY_ASSIGNER")
        elif t == "to" and i+1 < len(tokens):
            label_np_after(i+1, "PARTY_ASSIGNEE")

    # on behalf of -> represented party (often assignee)
    for i in range(len(tokens) - 3):
        if lt[i] == "on" and lt[i+1] == "behalf" and lt[i+2] == "of":
            label_np_after(i+3, "PARTY_ASSIGNEE")



# def seed_constraints(tokens: List[str], labels: List[str]):
#     lt = [t.lower() for t in tokens]
#     starters = {"without","except","solely","pursuant","under","upon",
#                 "before","as", "up to"}
#     i = 0
#     while i < len(tokens):
#         if lt[i] in starters:
#             j = i+1
#             while j < len(tokens) and tokens[j] not in {",",";","."}:
#                 j += 1
#             safe_label_span(labels, i, j-i, "CONSTRAINT")
#             i = j
#         else:
#             i += 1
#     # "in writing"
#     for i in range(len(tokens)-1):
#         if lt[i] == "in" and lt[i+1] == "writing":
#             if labels[i] == "O": labels[i] = "B-CONSTRAINT"
#             if labels[i+1] == "O": labels[i+1] = "I-CONSTRAINT"


def seed_constraints(tokens: List[str], labels: List[str]):
    """
    Expanded CONSTRAINT seeder aligned with ODRL's LeftOperand–Operator–RightOperand idea.
    Strategy:
      1) Label phrase starters (multi-token) and expand to punctuation.
      2) Label operator-based comparatives (at least / no more than / before / after / between / within / by <date> / per <unit>).
      3) Label LeftOperand+Operator combos (e.g., 'file format is pdf', 'language equals english') up to punctuation.
    Never overwrites existing non-'O' labels.
    """
    lt = [t.lower() for t in tokens]
    PUNCT_STOP = {",", ";", ".", ")", "]", "}"}
    # Allow these tokens inside a constraint span
    CONNECT = {"of", "and", "or", "to", "the", "a", "an", "for", "in"}

    def span_to_punct(start_idx: int) -> int:
        """Return end exclusive index by running until punctuation/close-delim."""
        j = start_idx
        while j < len(tokens) and tokens[j] not in PUNCT_STOP:
            j += 1
        return j

    # ---------- 1) Phrase starters (multi-token) ----------
    # Keep these specific to reduce false positives.
    starter_phrases = [
        "without", "except", "solely",
        "pursuant to", "in accordance with", "in compliance with",
        "under", "upon", "prior to", "before", "after", "during", "within",
        "no later than", "no earlier than", "on or before", "on or after",
        "up to", "at least", "at most", "no more than", "not more than",
        "not less than", "more than", "fewer than",
        # documentation/formatting
        "in writing",
        # role/policy references
        "as permitted", "as required", "as defined", "as set forth", "as applicable",
    ]
    for phrase in starter_phrases:
        for s, l in find_phrase_spans(tokens, phrase):
            e = span_to_punct(s + l)
            safe_label_span(labels, s, e - s, "CONSTRAINT")

    # ---------- 2) Operator-anchored comparatives & patterns ----------
    # a) BETWEEN X AND Y
    for i, w in enumerate(lt):
        if w == "between" and labels[i] == "O":
            j = i + 1
            # go until punctuation; heuristic: ensure there's an 'and' in the slice
            has_and = False
            while j < len(tokens) and tokens[j] not in PUNCT_STOP:
                if lt[j] == "and":
                    has_and = True
                j += 1
            if has_and:
                safe_label_span(labels, i, j - i, "CONSTRAINT")

    # b) PER <unit>  (per day/month/year/user/device/account/request/disclosure)
    PER_UNITS = {
        "day","days","month","months","year","years","week","weeks", "times",
        "user","users","device","devices","account","accounts",
        "request","requests","disclosure","disclosures", "date", "period"
    }
    for i, w in enumerate(lt):
        if w == "per" and labels[i] == "O" and i + 1 < len(tokens):
            if lt[i+1] in PER_UNITS and labels[i+1] == "O":
                e = i + 2
                safe_label_span(labels, i, e - i, "CONSTRAINT")

    # c) BY <date/time>  (by 2025-01-01, by 31 December 2025, by 10:30)
    import re
    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$|^\d{1,2}/\d{1,2}/\d{2,4}$|^\d{1,2}:\d{2}([ap]m)?$", re.I)
    MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
    for i, w in enumerate(lt):
        if w == "by" and labels[i] == "O" and i + 1 < len(tokens):
            nxt = tokens[i+1]
            nxtl = lt[i+1]
            looks_date = bool(DATE_RE.match(nxt)) or (nxtl in MONTHS)
            if looks_date:
                e = span_to_punct(i + 1 + 1)  # include trailing tokens like "31 2025"
                safe_label_span(labels, i, e - i, "CONSTRAINT")

    # d) COMPARATIVE OPERATORS (at least / no more than / < / > / etc.)
    comp_ops = [
        "at least", "at most", "no more than", "not more than", "not less than", "no less than",
        "less than", "less than or equal to", "greater than", "greater than or equal to",
        "more than", "equal to", "equals", "not equal to",
        "before", "after", "on or before", "on or after", "no later than", "no earlier than",
        "within", "by"
    ]
    # Percent/currency detector (light)
    PCT = {"%","percent","per","percentage"}
    CURR = {"$","£","€","usd","gbp","eur"}
    TIME_UNITS = {"second","seconds","minute","minutes","hour","hours","day","days","week","weeks","month","months","year","years","business","working"}

    def looks_numeric_chunk(idx: int) -> bool:
        if idx >= len(tokens): return False
        t = tokens[idx]
        l = lt[idx]
        if t.isdigit(): return True
        if any(ch.isdigit() for ch in t): return True
        if l in PCT or l in CURR: return True
        if l in TIME_UNITS: return True
        return False

    for phrase in comp_ops:
        for s, l in find_phrase_spans(tokens, phrase):
            # Extend rightwards to cover a numeric/date chunk or NP until punctuation
            j = s + l
            # greedily include next few tokens if they look like right-operand (number/unit/date)
            k = j
            consumed = False
            while k < len(tokens) and tokens[k] not in PUNCT_STOP and (looks_numeric_chunk(k) or lt[k] in CONNECT):
                k += 1
                consumed = True
            if not consumed:
                # still include until punctuation (general case)
                k = span_to_punct(j)
            safe_label_span(labels, s, k - s, "CONSTRAINT")

    # ---------- 3) LeftOperand + Operator combos ----------
    # LeftOperands (ODRL-inspired; multiwords included)
    left_operands = [
        "absolute position","absolute size","absolute spatial position","absolute temporal position",
        "relative position","relative size","relative spatial position","relative temporal position",
        "count","date time","delay period","elapsed time","time interval","unit of count",
        "delivery channel","device","system","system device",
        "event","industry","language","media","file format","resolution",
        "pay amount","percentage","product","purpose","recipient","version","virtual location",
        "spatial","spatial coordinates",
    ]
    # Operator phrases to look for after LeftOperand
    op_phr = [
        "is", "are", "be", "must be", "shall be", "equals", "equal to", "not equal to",
        "less than", "less than or equal to", "greater than", "greater than or equal to",
        "is any of", "is all of", "is none of", "is a", "is part of", "has part",
        "before", "after", "during", "within", "by", "between"
    ]

    # try: [LeftOperand] ... [Operator] ... (to punctuation)
    MAX_LOOKAHEAD = 14
    for lop in left_operands:
        for s, l in find_phrase_spans(tokens, lop):
            if any(labels[s+i] != "O" for i in range(l)):
                continue
            end_search = min(len(tokens), s + l + MAX_LOOKAHEAD)
            # find first operator occurrence in the lookahead window
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
                # If no explicit operator found, still label LeftOperand + following NP up to punctuation
                e = span_to_punct(s + l)
                safe_label_span(labels, s, e - s, "CONSTRAINT")

    # Keep your explicit "in writing" (already covered via phrase starters, but harmless)
    for i in range(len(tokens) - 1):
        if lt[i] == "in" and lt[i+1] == "writing":
            if labels[i] == "O": labels[i] = "B-CONSTRAINT"
            if labels[i+1] == "O": labels[i+1] = "I-CONSTRAINT"


# def seed_duty(tokens: List[str], labels: List[str]):
#     lt = [t.lower() for t in tokens]
#     duty_phrases = [
#         "be subject to the terms and conditions of this agreement",
#         "be provided in writing",
#         "has an appropriate policy document in place",
#         "undergo an assessment check",
#     ]
#     for p in duty_phrases:
#         for s, l in find_phrase_spans(tokens, p):
#             safe_label_span(labels, s, l, "DUTY")
#     connectors = [
#         ["provided","that"], ["provided"], ["obligation"], ["as","long","as"],
#         ["on","condition","that"], ["subject","to"], ["only","if"]
#     ]
#     for conn in connectors:
#         clen = len(conn)
#         for i in range(len(tokens) - clen + 1):
#             if [t.lower() for t in tokens[i:i+clen]] == conn:
#                 j = i + clen
#                 while j < len(tokens) and tokens[j] not in {",",";","."}:
#                     j += 1
#                 safe_label_span(labels, i, j-i, "DUTY")
#     for i in range(len(tokens)-2):
#         if lt[i] in {"must","shall"} and lt[i+1] == "be":
#             j = i + 1
#             while j < len(tokens) and tokens[j] not in {",",";","."}:
#                 j += 1
#             safe_label_span(labels, i+1, j-(i+1), "DUTY")

def seed_duty(tokens: List[str], labels: List[str]):
    """
    DUTY = obligation that must be fulfilled (obligation to perform/achieve something).
    Strategy:
      1) Label explicit duty phrases.
      2) Label duty connectors (provided that / subject to / only if / as long as / so long as / on condition that),
         including 'provided, however, that' → up to punctuation.
      3) Label modal + duty verb content (leave the modal itself to RULE_TYPE),
         e.g., 'must ensure ...', 'shall maintain ...', 'will provide ...', 'must be provided ...'.
      4) Label 'is/are required/obligated/obliged to', 'has/have a duty to', 'responsible for',
         'agrees/undertakes to', and 'ensure that ...' clauses.
    Never overwrites existing non-'O' labels.
    """
    lt = [t.lower() for t in tokens]
    PUNCT_STOP = {",", ";", ".", ")", "]", "}"}

    def end_at_punct(start: int) -> int:
        j = start
        while j < len(tokens) and tokens[j] not in PUNCT_STOP:
            j += 1
        return j

    # ---------------- 1) Explicit duty phrases ----------------
    phrase_list = [
        "be subject to the terms and conditions of this agreement",
        "be provided in writing",
        "provided in writing",
        "has an appropriate policy document in place",
        "appropriate policy document in place",
        "undergo an assessment check",
        "be kept confidential",
        "be returned",  # overlap w/ ACTION elsewhere; here used when clearly obligation phrasing
    ]
    for p in phrase_list:
        for s, l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "DUTY")

    # ---------------- 2) Duty connectors (whole clause) ----------------
    # include 'provided, however, that' variant
    connector_forms = [
        ["provided", "that"],
        ["provided"],  # we'll look ahead for 'that' optionally
        ["as", "long", "as"],
        ["so", "long", "as"],
        ["on", "condition", "that"],
        ["subject", "to"],
        ["only", "if"],
        ["obligation"],  # noun cue: often followed by 'to <verb>'
    ]
    for conn in connector_forms:
        clen = len(conn)
        i = 0
        while i <= len(tokens) - clen:
            if [t.lower() for t in tokens[i:i+clen]] == conn:
                # If this is bare 'provided', try to allow comma/“however”/comma/“that”
                j = i + clen
                if conn == ["provided"] and j < len(tokens):
                    k = j
                    # optional , however , that
                    if k < len(tokens) and tokens[k] == ",":
                        k += 1
                        if k < len(tokens) and lt[k] == "however":
                            k += 1
                            if k < len(tokens) and tokens[k] == ",":
                                k += 1
                    if k < len(tokens) and lt[k] == "that":
                        j = k + 1  # start after 'that'
                # extend to punctuation
                e = end_at_punct(j)
                safe_label_span(labels, i, e - i, "DUTY")
                i = e
            else:
                i += 1

    # ---------------- 3) Modal + duty verb (content only) ----------------
    # Leave the modal token(s) (must/shall/will/should) to RULE_TYPE;
    # mark the content from the duty verb onward (or 'be' in passives).
    DUTY_MAIN_VERBS = {
        "ensure","maintain","retain","provide","return","destroy","keep","implement",
        "follow","comply","protect","safeguard","secure","notify","report","document",
        "record","obtain","seek","limit","restrict","apply","adhere","encrypt",
        "de-identify","deidentified","deidentify","anonymize","anonymise","pseudonymize","pseudonymise"
    }
    MODALS = {"must","shall","will","should"}
    for i, w in enumerate(lt):
        if w in MODALS and labels[i] == "O":
            # passive: modal + be + <past participle/...>  -> start at 'be'
            if i + 1 < len(tokens) and lt[i+1] == "be":
                start = i + 1
                e = end_at_punct(start + 1)
                safe_label_span(labels, start, e - start, "DUTY")
                continue
            # active: modal + <duty-verb> ...
            if i + 1 < len(tokens) and labels[i+1] == "O":
                v = lt[i+1]
                # simple morphology check (ensure/ensures/ensured/ensuring)
                candidates = {v}
                if v.endswith("ing") and len(v) > 4: candidates.add(v[:-3])
                if v.endswith("ed")  and len(v) > 3: candidates.add(v[:-2])
                if v.endswith("es")  and len(v) > 3: candidates.add(v[:-2])
                if v.endswith("s")   and len(v) > 3: candidates.add(v[:-1])
                if candidates & DUTY_MAIN_VERBS:
                    start = i + 1
                    e = end_at_punct(start + 1)
                    safe_label_span(labels, start, e - start, "DUTY")

    # ---------------- 4) 'required/obligated/obliged to', 'duty to', 'responsible for', 'agrees/undertakes to' ----------------
    # Forms:
    #   is/are required to <verb ...>
    #   is/are obligated to / obliged to <verb ...>
    #   has/have a duty to <verb ...>
    #   responsible for <gerund/NP ...>
    #   agrees to / undertake(s) to / undertakes to <verb ...>
    patterns = [
        (["is","required","to"], "to"),
        (["are","required","to"], "to"),
        (["is","obligated","to"], "to"),
        (["are","obligated","to"], "to"),
        (["is","obliged","to"], "to"),
        (["are","obliged","to"], "to"),
        (["has","a","duty","to"], "to"),
        (["have","a","duty","to"], "to"),
        (["agrees","to"], "to"),
        (["agree","to"], "to"),
        (["undertakes","to"], "to"),
        (["undertake","to"], "to"),
        (["responsible","for"], "for"),
        (["ensure","that"], "that"),
        (["must","ensure","that"], "that"),   # modal included; safe_label_span won't overwrite RULE_TYPE modal
        (["shall","ensure","that"], "that"),
        (["will","ensure","that"], "that"),
    ]
    for pat, anchor in patterns:
        L = len(pat)
        i = 0
        while i <= len(tokens) - L:
            if [t.lower() for t in tokens[i:i+L]] == pat and all(labels[i+k] == "O" for k in range(L)):
                start = i
                # If pattern contains a modal (must/shall/will), start labeling AFTER the modal token
                # so modal remains available for RULE_TYPE.
                if pat[0] in {"must","shall","will"}:
                    start = i + 1  # begin at 'ensure' (or 'be' etc.)
                e = end_at_punct(i + L)
                safe_label_span(labels, start, e - start, "DUTY")
                i = e
            else:
                i += 1


# def seed_purpose(tokens: List[str], labels: List[str]):
#     lt = [t.lower() for t in tokens]
#     purpose_phrases = [
#         "for law enforcement purposes",
#         "for the purpose of",
#         "for the purposes of",
#         "in order to",
#     ]
#     for p in purpose_phrases:
#         for s, l in find_phrase_spans(tokens, p):
#             j = s + l
#             while j < len(tokens) and tokens[j] not in {",",";","."}:
#                 j += 1
#             safe_label_span(labels, s, j - s, "PURPOSE")
#     for i, t in enumerate(lt):
#         if t == "for":
#             j = i + 1
#             seen_purpose = False
#             while j < len(tokens) and tokens[j] not in {",",";","."}:
#                 if tokens[j].lower() in {"purpose","purposes"}:
#                     seen_purpose = True
#                 j += 1
#             if seen_purpose:
#                 safe_label_span(labels, i, j - i, "PURPOSE")
#     intent_verbs = {
#         "enable","ensure","prevent","promote","carry","perform","process",
#         "share","safeguard","improve","comply","investigate","detect","prosecute","protect"
#     }
#     for i in range(len(tokens)-1):
#         if lt[i] == "to" and tokens[i+1].lower() in intent_verbs:
#             j = i + 2
#             while j < len(tokens) and tokens[j] not in {",",";","."}:
#                 j += 1
#             safe_label_span(labels, i, j - i, "PURPOSE")


def seed_purpose(tokens: List[str], labels: List[str]):
    """
    PURPOSE = Defined purpose for exercising an action (e.g., 'for educational use', 'in order to ensure compliance').
    Labels purpose clauses/phrases; avoids overwriting existing non-'O' tags.
    """
    lt = [t.lower() for t in tokens]
    PUNCT_STOP = {",", ";", ".", ")", "]", "}"}
    DETS = {"the","a","an","this","that","these","those"}

    def end_at_punct(start: int) -> int:
        j = start
        while j < len(tokens) and tokens[j] not in PUNCT_STOP:
            j += 1
        return j

    # ---------- 1) Explicit purpose phrases ----------
    phrase_list = [
        # strong explicit forms
        "for the purpose of", "for the purposes of", "for purposes of",
        "for law enforcement purposes", "for research purposes", "for statistical purposes",
        "for audit purposes", "for monitoring purposes", "for compliance purposes",
        "for educational use", "for internal use", "for commercial use", "for non commercial use",
        "for service delivery", "for safeguarding", "for safeguarding children",
        "for training", "for testing", "for evaluation", "for administrative purposes",
        "for use in", "for use by", "for use of",
        # intent markers
        "in order to", "so as to", "so that", "with the purpose of",
        "with the aim of", "with the intention of", "with a view to",
    ]
    for p in phrase_list:
        for s, l in find_phrase_spans(tokens, p):
            e = end_at_punct(s + l)
            safe_label_span(labels, s, e - s, "PURPOSE")

    # ---------- 2) Generic 'for …' purpose (nouns/gerunds) ----------
    purpose_heads = {
        # common heads/nouns signalling purpose
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
        # skip determiners
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
            # patterns like "for educational use" / "for non - commercial use"
            if lw in purpose_use_adjs:
                k2 = k + 1
                # allow optional '-' between non and commercial
                if k2 < len(tokens) and tokens[k2] == "-":
                    k2 += 1
                if k2 < len(tokens) and lt[k2] == "use":
                    seen_signal = True
                    k = k2 + 1
                    continue
            # 'for use in/by/of …'
            if lw == "use" and k + 1 < len(tokens) and lt[k+1] in {"in","by","of"}:
                seen_signal = True
                k += 2
                continue
            k += 1
        if seen_signal:
            e = end_at_punct(j)
            safe_label_span(labels, i, e - i, "PURPOSE")

    # ---------- 3) Bare 'to <verb …>' purpose clauses (guarded) ----------
    # allow when clearly intent, e.g., ", to ensure …", "in order to …", "so as to …"
    intent_verbs = {
        "enable","ensure","prevent","promote","support","facilitate",
        "comply","investigate","detect","prosecute","protect","safeguard",
        "improve","monitor","evaluate","assess","administer","deliver","provide",
        "process","share","use","inform","analyze","analyse","review","enforce"
    }
    # preceding patterns that are NOT purpose (duty/rights verbs)
    NOT_PURPOSE_PREV = {
        "agree","agrees","agreed","agreeing",
        "undertake","undertakes","undertook","undertaking",
        "required","require","requires","obligated","obliged","obligation","responsible","duty",
        "right","consent","permission","authorize","authorise","authorized","authorised"
    }

    for i in range(len(tokens) - 1):
        if lt[i] != "to" or labels[i] != "O":
            continue
        nxt = lt[i+1]
        # guard: skip patterns like "agree to", "required to", "right to", "consent to"
        if i >= 1 and lt[i-1] in NOT_PURPOSE_PREV:
            continue
        # accept if (a) sentence/segment start, (b) preceded by comma/semicolon,
        # or (c) introduced by 'in order' / 'so as'
        prev_ok = (
            i == 0 or tokens[i-1] in {",",";","("} or
            (i >= 2 and lt[i-2] == "in" and lt[i-1] == "order") or
            (i >= 2 and lt[i-2] == "so" and lt[i-1] == "as")
        )
        if prev_ok or nxt in intent_verbs:
            e = end_at_punct(i + 2)
            safe_label_span(labels, i, e - i, "PURPOSE")

def seed_label(tokens: List[str]) -> List[str]:
    labels = ["O"] * len(tokens)
    seed_parties(tokens, labels)
    seed_assets(tokens, labels)
    seed_rule_type(tokens, labels)
    seed_actions(tokens, labels)
    seed_duty(tokens, labels)
    seed_purpose(tokens, labels)
    seed_constraints(tokens, labels)
    return labels

def has_non_O(labels: List[str]) -> bool:
    return any(l != "O" for l in labels)

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

# ------------------ prompts & schema ------------------

def build_prompt(tokens: List[str], seed_labels_: List[str]) -> List[Dict[str, Any]]:
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]

    # few-shot (optional)
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

    # actual request
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

# ------------------ parse Responses API ------------------

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

# ------------------ API helpers with retry & repair ------------------

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

# ------------------ pipeline ------------------

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

        # length repair
        repairs = 0
        while len(labels) != len(tokens) and repairs < max_repairs:
            labels = await repair_length(tokens, labels)
            repairs += 1

        # all-O guard
        if set(labels) == {"O"} and has_non_O(seed):
            labels = await repair_all_O(tokens, seed)
            if len(labels) != len(tokens):
                labels = await repair_length(tokens, labels)

        # last-resort alignment
        if len(labels) != len(tokens):
            if len(labels) < len(tokens):
                labels += ["O"] * (len(tokens) - len(labels))
            else:
                labels = labels[:len(tokens)]

        # if still all O but seed had signal, fallback to seed
        if set(labels) == {"O"} and has_non_O(seed):
            labels = seed

    except Exception:
        # total failure -> seed if useful, else all O
        labels = seed if has_non_O(seed) else ["O"] * len(tokens)

    return {"idx": u.idx, "text": u.text, "label": labels}

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

async def main():
    # Load lines
    lines = load_lines(INPUT_TXT)

    # Build units with HF-tokenized words + seed labels
    units = []
    for i, t in enumerate(lines):
        toks = tokenize(t)          # HF word-level tokens (punctuation split)
        seed = seed_label(toks)     # heuristic seeds on the same tokens
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

    # Write JSONL in original order
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for res in results:
            obj = {"text": res["text"], "label": res["label"]}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # simple summary
    total = len(results)
    allO = sum(1 for r in results if set(r["label"]) == {"O"})
    print(f"Done. Wrote {total} items -> {OUTPUT_JSONL} | All-'O' lines: {allO}")

if __name__ == "__main__":
    asyncio.run(main())
