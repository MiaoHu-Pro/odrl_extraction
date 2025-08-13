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

# LABEL_ENUM = [
#    "O",
#    "B-PARTY_ASSIGNEE", "I-PARTY_ASSIGNEE",
#    "B-PARTY_ASSIGNER", "I-PARTY_ASSIGNER",
#    "B-ASSET", "I-ASSET",
#    "B-ACTION", "I-ACTION",
#    "B-RULE_TYPE", "I-RULE_TYPE",
#    "B-CONSTRAINT", "I-CONSTRAINT",
#    "B-DUTY", "I-DUTY"
#    "B-PURPOSE", "I-PURPOSE"
# ]


SYSTEM_INSTRUCTIONS = """You are an expert BIO annotator for Data Sharing Agreements.
You will receive:
- tokens: the exact token sequence (punctuation kept as separate tokens),
- seed_labels: heuristic BIO tags to START from.

Task:
- Correct and improve seed_labels to produce final BIO tags.
- Output EXACTLY one tag per token, same order and length.
- Punctuation tokens (e.g., ',', '.', '(', ')', '-', ':', ';', quotes) must be 'O'.
- Include negation with RULE_TYPE (e.g., 'will not' -> B-RULE_TYPE I-RULE_TYPE).
- Each head verb in coordination is ACTION (e.g., 'share, publish, or release').
- ASSET is the governed noun phrase; CONSTRAINT covers conditions ('without prior approval ...', 'solely for ...', etc).
- PARTY_ASSIGNEE/ASSIGNER are actor mentions per role; keep minimal spans via BIO.
- DUTY: obligations introduced by connectors like 'provided that', 'as long as', 'subject to', 'only if', or phrases like 'be subject to the terms and conditions', 'be provided in writing', etc.
- PURPOSE: purposes introduced by 'for ... purpose(s)', 'for law enforcement purposes', 'in order to ...', or infinitival purpose 'to <verb>' stating intent, etc.

Important:
- Do NOT output all 'O' when seed_labels contain non-'O' unless those non-'O' are clearly wrong.
- If a seed label looks correct, keep it; otherwise fix it..
- Keep correct seed tags; fix incorrect ones; add missing spans.
Return JSON ONLY per the schema.
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
    lt = [t.lower() for t in tokens]
    for i in range(len(tokens)-1):
        if (lt[i], lt[i+1]) in {("will","not"), ("shall","not"), ("must","not"), ("may","not")}:
            if labels[i] == "O": labels[i] = "B-RULE_TYPE"
            if labels[i+1] == "O": labels[i+1] = "I-RULE_TYPE"
    for i,t in enumerate(lt):
        if t in {"shall","must","will","should"} and labels[i] == "O":
            labels[i] = "B-RULE_TYPE"

def seed_actions(tokens: List[str], labels: List[str]):
    phrases = ["granted access", "be returned", "be released", "be utilized", "be used"]
    singles = {"release","disclose","share","publish","present","remain","return",
               "comply","utilize","utilized","facilitate","agree","handle","handling",
               "provide","assist","assistance"}
    for p in phrases:
        for s,l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "ACTION")
    for i,tok in enumerate(tokens):
        if tok.lower() in singles and labels[i] == "O":
            labels[i] = "B-ACTION"

def seed_assets(tokens: List[str], labels: List[str]):
    phrases = [
        "results of data analysis", "findings or conclusions",
        "names of individuals", "individual addresses",
        "personal data breach", "personal data",
        "information", "maps", "addresses", "data",
    ]
    for p in phrases:
        for s,l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "ASSET")

def seed_parties(tokens: List[str], labels: List[str]):
    lt = [t.lower() for t in tokens]
    assignee_phr = ["data recipient","third party","third parties","both parties","parties","tpp"]
    assigner_phr = ["data provider","ridoh"]
    for p in assignee_phr:
        for s,l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "PARTY_ASSIGNEE")
    for p in assigner_phr:
        for s,l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "PARTY_ASSIGNER")
    # simple prepositional cues
    for i,t in enumerate(lt):
        if t == "by" and i+1 < len(tokens):
            j = i+1
            if tokens[j][0].isupper() or tokens[j].lower() in {"ridoh","tpp","provider","recipient","party","parties"}:
                safe_label_span(labels, j, 1, "PARTY_ASSIGNER")
        if t == "to" and i+1 < len(tokens):
            j = i+1
            if tokens[j][0].isupper() or tokens[j].lower() in {"party","parties","recipient","tpp"}:
                safe_label_span(labels, j, 1, "PARTY_ASSIGNEE")

def seed_constraints(tokens: List[str], labels: List[str]):
    lt = [t.lower() for t in tokens]
    starters = {"without","except","solely","pursuant","under","upon","before","as"}
    i = 0
    while i < len(tokens):
        if lt[i] in starters:
            j = i+1
            while j < len(tokens) and tokens[j] not in {",",";","."}:
                j += 1
            safe_label_span(labels, i, j-i, "CONSTRAINT")
            i = j
        else:
            i += 1
    # "in writing"
    for i in range(len(tokens)-1):
        if lt[i] == "in" and lt[i+1] == "writing":
            if labels[i] == "O": labels[i] = "B-CONSTRAINT"
            if labels[i+1] == "O": labels[i+1] = "I-CONSTRAINT"

def seed_duty(tokens: List[str], labels: List[str]):
    lt = [t.lower() for t in tokens]
    duty_phrases = [
        "be subject to the terms and conditions of this agreement",
        "be provided in writing",
        "has an appropriate policy document in place",
        "undergo an assessment check",
    ]
    for p in duty_phrases:
        for s, l in find_phrase_spans(tokens, p):
            safe_label_span(labels, s, l, "DUTY")
    connectors = [
        ["provided","that"], ["provided"], ["as","long","as"],
        ["on","condition","that"], ["subject","to"], ["only","if"]
    ]
    for conn in connectors:
        clen = len(conn)
        for i in range(len(tokens) - clen + 1):
            if [t.lower() for t in tokens[i:i+clen]] == conn:
                j = i + clen
                while j < len(tokens) and tokens[j] not in {",",";","."}:
                    j += 1
                safe_label_span(labels, i, j-i, "DUTY")
    for i in range(len(tokens)-2):
        if lt[i] in {"must","shall"} and lt[i+1] == "be":
            j = i + 1
            while j < len(tokens) and tokens[j] not in {",",";","."}:
                j += 1
            safe_label_span(labels, i+1, j-(i+1), "DUTY")

def seed_purpose(tokens: List[str], labels: List[str]):
    lt = [t.lower() for t in tokens]
    purpose_phrases = [
        "for law enforcement purposes",
        "for the purpose of",
        "for the purposes of",
        "in order to",
    ]
    for p in purpose_phrases:
        for s, l in find_phrase_spans(tokens, p):
            j = s + l
            while j < len(tokens) and tokens[j] not in {",",";","."}:
                j += 1
            safe_label_span(labels, s, j - s, "PURPOSE")
    for i, t in enumerate(lt):
        if t == "for":
            j = i + 1
            seen_purpose = False
            while j < len(tokens) and tokens[j] not in {",",";","."}:
                if tokens[j].lower() in {"purpose","purposes"}:
                    seen_purpose = True
                j += 1
            if seen_purpose:
                safe_label_span(labels, i, j - i, "PURPOSE")
    intent_verbs = {
        "enable","ensure","prevent","promote","carry","perform","process",
        "share","safeguard","improve","comply","investigate","detect","prosecute","protect"
    }
    for i in range(len(tokens)-1):
        if lt[i] == "to" and tokens[i+1].lower() in intent_verbs:
            j = i + 2
            while j < len(tokens) and tokens[j] not in {",",";","."}:
                j += 1
            safe_label_span(labels, i, j - i, "PURPOSE")

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
