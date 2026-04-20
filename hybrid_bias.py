from __future__ import annotations
import re
from typing import List, Sequence, Dict, Any
import torch


"""
Hybrid Bias — Rule‑Biased Viterbi Decode for BERT+BiLSTM+CRF
===========================================================

Adds small positive logits to emissions where rules fire, then runs CRF
Viterbi decode. Works with your `BERT_LSTM_CRF` that exposes either:
  • `compute_emissions(input_ids, attention_mask)` or
  • `forward(..., return_emissions=True)` (returns object with `.emissions`)

If neither is available, we try `get_output_score(...)` or a forward
side‑effect attribute `.emissions`.

Use in eval/predict (no retraining):

    from hybrid_bias import decode_with_rule_bias

    pred_ids = decode_with_rule_bias(
        model, batch, batch_features, label2id=cfg.l2i, alpha_pos=3.0
    )

Then call your existing `recover_label(...)`.
"""


# ---------------------------------------------------------------------------
# Lexicons (lowercased where noted). Extend freely.
# ---------------------------------------------------------------------------
ACTION_VERBS = {
    "allow","perform","process","use","share","disclose","collect","receive","provide",
    "transfer","transmit","send","store","retain","delete","destroy","anonymize","pseudonymize",
    "aggregate","publish","access","view","monitor","profile","sell","reidentify","export","import","distribute"
}
ASSET_HEADS = {
    "data","information","record","records","content","document","documents","file","files","dataset","datasets",
    "metadata","log","logs","metric","metrics","identifier","identifiers","email","emails","name","names",
    "address","addresses","phone","phones","location","locations","biometric","financial","health","phi","pii",
    "account","accounts","credential","credentials","password","passwords","token","tokens","ip",
    "cookie","cookies","device","devices","telemetry","analytics","personal","special","category", "report","reports","confidential","report"
}
DUTY_VERBS = {
    "ensure","implement","maintain","provide","notify","report","cooperate","obtain","document",
    "permit","audit","comply","protect","encrypt","pseudonymize","anonymize","delete","return","certify",
    "segregate","restrict","access","log","train"
}
JURISDICTIONS = {"EU","EEA","UK","United","States","US","California","Switzerland","Canada","Brazil","China","India"}

# Party role lexicons (all lowercase for phrase matching)
ASSIGNEE_PHRASES = [
    "the data recipient","data recipient","recipient",
    "the third party","third party","the third parties","third parties",
    "both parties","each party","either party","the parties","parties",
    "the licensee","licensee",
    "the end user","end user",
    "the authorized user","authorized user",
    "the authorized users","authorized users",
    "the users","users","the user","user",
    "the data processor","data processor","processor",
    "the service provider","service provider","service providers",
    "the vendor","vendor","vendors",
    "the contractor","contractor","contractors",
    "the researcher","researcher","researchers",
    "the requester","requester","requestor","requestors",
    "the applicant","applicant","applicants",
    "the partner","partner","partners",
    "the data importer","data importer",
    "the receiving party","receiving party",
    "the data receiver","data receiver","receiver",
]
ASSIGNER_PHRASES = [
    "the data provider","data provider",
    "the data owner","data owner","owner",
    "the licensor","licensor",
    "the discloser","discloser",
    "the disclosing party","disclosing party",
    "the data discloser","data discloser",
    "the data controller","data controller","controller",
    "the authority","authority","the authorities","authorities",
    "the competent authority","competent authority",
    "the competent authorities","competent authorities",
    "the government department","government department",
    "the government departments","government departments",
    "the public authority","public authority",
    "the public authorities","public authorities",
    "the police","police",
    "the probation services","probation services",
    "the youth offending team","youth offending team",
    "the youth offending teams","youth offending teams",
    "the london fire brigade","london fire brigade",
    "the public protection desk","public protection desk","ppd",
    "the data exporter","data exporter","exporter",
]
SINGLE_ASSIGNEE = {
    "recipient","licensee","processor","researcher","contractor","vendor",
    "user","users","requester","requestors","requestor","applicant","partner","partners",
    "importer","receiver"
}
SINGLE_ASSIGNER = {
    "provider","owner","licensor","discloser","controller","exporter",
    "authority","authorities","commission","commissioner","council","agency","agencies",
    "police","department","departments","brigade"
}

# CONSEQUENCE lexicons (all lowercase)
TRIGGER_WORDS = {
    "breach","breaches","breached","breaching",
    "violation","violations","violate","violates","violated","violating",
    "non-compliance","noncompliance","noncompliant","non-compliant",
    "default","defaults","defaulted","defaulting",
    "fail","fails","failed","failing","failure",
    "nonperformance","non-performance",
}
TRIGGER_PHRASES = [
    "material breach","in the event of","in case of","in the case of",
    "upon breach","upon any breach","on breach","on any breach",
    "failure to comply","breach of this agreement","breach of confidentiality",
    "breach of contract","breach of the agreement"
]
SANCTION_VERBS = {
    "terminate","terminates","terminated","terminating",
    "suspend","suspends","suspended","suspending",
    "revoke","revokes","revoked","revoking",
    "withdraw","withdraws","withdrew","withdrawn","withdrawing",
    "disable","disables","disabled","disabling",
    "block","blocks","blocked","blocking",
    "forfeit","forfeits","forfeited","forfeiting",
    "fine","fines","fined","fining",
    "penalize","penalise","penalizes","penalises","penalized","penalised","penalizing","penalising",
    "sanction","sanctions","sanctioned","sanctioning",
    "remedy","remedies","remedied","remedying",
    "cure","cures","cured","curing",
    "indemnify","indemnifies","indemnified","indemnifying",
    "refund","refunds","refunded","refunding",
    "cease","ceases","ceased","ceasing",
}
SANCTION_PARTICIPLES = {
    "terminated","suspended","revoked","withdrawn","disabled","blocked","forfeited","fined","penalized","penalised"
}
SANCTION_NOUNS = {
    "termination","suspension","revocation","withdrawal","forfeiture",
    "damages","penalty","penalties","fine","fines","sanctions",
    "liability","indemnity","indemnification","remedy","cure","cure period"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_special(toks: Sequence[str]) -> List[str]:
    return [t for t in toks if t not in ("[CLS]","[SEP]")]


# ---------------------------------------------------------------------------
# Rule bias builder
# ---------------------------------------------------------------------------

def build_rule_bias_for_tokens(
    tokens: Sequence[str],
    label2id: Dict[str,int],
    alpha_pos: float = 3.0,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Return bias tensor [T_no_special, C] with +alpha on suggested tags.
    PURPOSE cues are folded into CONSTRAINT (no PURPOSE tag in BIO_HEAD).
    If num_classes is provided, indices beyond it are skipped safely.
    """
    toks = _strip_special(tokens)
    T = len(toks)
    C = num_classes if num_classes is not None else max(label2id.values())+1
    bias = torch.zeros(T, C)
    low = [t.lower() for t in toks]
    txt = " ".join(low)

    def _idx(tag: str) -> int | None:
        i = label2id.get(tag)
        if i is None: return None
        if i >= C:    return None
        return i

    def add_span(start: int, end: int, btag: str, itag: str):
        if start < 0 or start >= T: return
        bi = _idx(btag); ii = _idx(itag)
        if bi is None or ii is None: return
        bias[start, bi] += alpha_pos
        for k in range(start+1, min(end, T)):
            bias[k, ii] += alpha_pos

    # RULE_TYPE cues ---------------------------------------------------------
    for m in re.finditer(r"\b(shall not|must not|may not|prohibited from)\b", txt):
        s = txt[:m.start()].count(" ")
        l = len(m.group(1).split())
        add_span(s, s+l, "B-RULE_TYPE", "I-RULE_TYPE")
    for m in re.finditer(r"\b(may|can|permitted to|authorized to|shall|must|is required to|will ensure)\b", txt):
        s = txt[:m.start()].count(" ")
        l = len(m.group(1).split())
        add_span(s, s+l, "B-RULE_TYPE", "I-RULE_TYPE")

    # PURPOSE -> CONSTRAINT --------------------------------------------------
    for m in re.finditer(r"\bin order to\b", txt):
        s = txt[:m.start()].count(" ")
        add_span(s, s+3, "B-CONSTRAINT", "I-CONSTRAINT")
    for i, w in enumerate(low):
        if w == "to" and i+1 < T and re.match(r"^[a-z]+$", low[i+1]):
            end = i+2
            while end < T and toks[end] not in {",",";","."}:
                end += 1
            add_span(i, end, "B-CONSTRAINT", "I-CONSTRAINT")

    # Temporal constraints ---------------------------------------------------
    for m in re.finditer(r"\b(within|no\s+later\s+than|at\s+least|for|up\s+to|prior\s+to|after|before)\b\s+\d{1,4}\s+(day|days|week|weeks|month|months|year|years|hour|hours)\b", txt):
        s = txt[:m.start()].count(" ")
        l = len(m.group(0).split())
        add_span(s, s+l, "B-CONSTRAINT", "I-CONSTRAINT")

    # Location constraints ---------------------------------------------------
    for i, w in enumerate(low):
        if w in {"outside","within","in"}:
            end = i+1; found = False
            while end < min(T, i+8):
                if toks[end] in JURISDICTIONS: found = True
                if toks[end] in {",",";","."}: break
                end += 1
            if found:
                add_span(i, end, "B-CONSTRAINT", "I-CONSTRAINT")

    # DUTY cues --------------------------------------------------------------
    for m in re.finditer(r"\b(provided\s+that|so\s+long\s+as|on\s+condition\s+that)\b", txt):
        s = txt[:m.start()].count(" ")
        l = len(m.group(1).split())
        add_span(s, s+l, "B-DUTY", "I-DUTY")
    for i, w in enumerate(low):
        if w in DUTY_VERBS:
            end = i+1
            while end < T and toks[end] not in {",",";","."}:
                end += 1
            add_span(i, end, "B-DUTY", "I-DUTY")

    # CONSEQUENCE (trigger → sanction on right) ------------------------------
    HAS_CONSEQ = ("B-CONSEQUENCE" in label2id) and ("I-CONSEQUENCE" in label2id)
    if HAS_CONSEQ:
        trigger_pos = set(i for i, w in enumerate(low) if w in TRIGGER_WORDS)
        for phrase in TRIGGER_PHRASES:
            ph = phrase.split(); Lp = len(ph)
            if Lp and Lp <= T:
                for i in range(0, T - Lp + 1):
                    if low[i:i+Lp] == ph:
                        trigger_pos.add(i + Lp - 1)

        def has_trigger_within(i: int, back: int = 12) -> bool:
            left = max(0, i - back)
            return any(t >= left and t <= i for t in trigger_pos)

        # sanction verbs
        for i, w in enumerate(low):
            if w in SANCTION_VERBS and has_trigger_within(i):
                print(w)
                end = i + 1
                while end < T and toks[end] not in {",",";","."}:
                    end += 1
                add_span(i, end, "B-CONSEQUENCE", "I-CONSEQUENCE")

        # passive participles: "be terminated" etc.
        BE = {"be","is","are","was","were","been","being","shall","will","may","can","must"}
        for i in range(0, max(0, T-1)):
            if low[i] in BE and low[i+1] in SANCTION_PARTICIPLES and has_trigger_within(i+1):
                start = i+1; end = start+1
                while end < T and toks[end] not in {",",";","."}:
                    end += 1
                add_span(start, end, "B-CONSEQUENCE", "I-CONSEQUENCE")

        # sanction nouns: "termination of ..."
        for i, w in enumerate(low):
            if w in SANCTION_NOUNS and has_trigger_within(i):
                start = i; end = i+1
                if end < T and low[end] == "of":
                    end += 1
                    while end < T and toks[end] not in {",",";","."}:
                        end += 1
                add_span(start, end, "B-CONSEQUENCE", "I-CONSEQUENCE")

    # PARTY roles ------------------------------------------------------------
    dets = {"the","a","an"}
    for phrase in ASSIGNEE_PHRASES:
        ph = phrase.split(); Lp = len(ph)
        if not Lp or Lp > T: continue
        for i in range(0, T - Lp + 1):
            if low[i:i+Lp] == ph:
                off = 1 if ph[0] in dets and Lp > 1 else 0
                add_span(i+off, i+Lp, "B-PARTY_ASSIGNEE", "I-PARTY_ASSIGNEE")
    for phrase in ASSIGNER_PHRASES:
        ph = phrase.split(); Lp = len(ph)
        if not Lp or Lp > T: continue
        for i in range(0, T - Lp + 1):
            if low[i:i+Lp] == ph:
                off = 1 if ph[0] in dets and Lp > 1 else 0
                add_span(i+off, i+Lp, "B-PARTY_ASSIGNER", "I-PARTY_ASSIGNER")

    for i, w in enumerate(low):
        if w in SINGLE_ASSIGNEE:
            add_span(i, i+1, "B-PARTY_ASSIGNEE", "I-PARTY_ASSIGNEE")
        if w in SINGLE_ASSIGNER:
            add_span(i, i+1, "B-PARTY_ASSIGNER", "I-PARTY_ASSIGNER")

    # Light ASSET head prior -------------------------------------------------
    for i, w in enumerate(low):
        if w in {"data","information","records","dataset","datasets","report","reports"}:
            bi = _idx("B-ASSET")
            if bi is not None:
                bias[i, bi] += alpha_pos * 0.8

    return bias


def build_batch_rule_bias(
    features: List[Any],
    attn_mask: torch.Tensor,
    label2id: Dict[str,int],
    device: torch.device,
    alpha_pos: float = 3.0,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Return bias tensor [B, T, C] aligned to batched inputs (with specials).
    If num_classes is provided, bias width is C and tags with index >= C are ignored.
    """
    B, T = attn_mask.shape
    C = num_classes if num_classes is not None else max(label2id.values())+1
    bias = torch.zeros(B, T, C, device=device)

    act = attn_mask.bool()
    for b in range(B):
        active_idx = act[b].nonzero(as_tuple=False).squeeze(1)
        if active_idx.numel() < 3:
            continue
        start = int(active_idx[1].item())     # skip CLS
        end   = int(active_idx[-2].item())+1  # include last real token; skip SEP
        toks = features[b].text if isinstance(features[b].text, list) else str(features[b].text).split()
        per = build_rule_bias_for_tokens(toks, label2id, alpha_pos=alpha_pos, num_classes=C).to(device)
        L = min(per.shape[0], end-start)
        bias[b, start:start+L, :C] += per[:L, :C]
    return bias

# ---------------------------------------------------------------------------
# Emissions getter & decode
# ---------------------------------------------------------------------------

def _get_emissions(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Obtain emissions [B,T,C] robustly from your model.
    Order of preference:
      1) compute_emissions(input_ids, attention_mask)
      2) forward(..., return_emissions=True)
      3) forward(...) then read model.emissions (side effect)
      4) get_output_score(input_ids, attention_mask)
    """
    # 1) Dedicated method
    if hasattr(model, "compute_emissions"):
        try:
            return model.compute_emissions(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            return model.compute_emissions(input_ids, attention_mask)

    # 2) Forward with return_emissions
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_emissions=True)
        if hasattr(out, "emissions"):
            return out.emissions
        if isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
            return out[1]
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            return out
    except TypeError:
        try:
            out = model(input_ids, attention_mask, return_emissions=True)
            if hasattr(out, "emissions"):
                return out.emissions
            if isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], torch.Tensor):
                return out[1]
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                return out
        except TypeError:
            pass

    # 3) Side‑effect attribute
    try:
        _ = model(input_ids, attention_mask)
        if hasattr(model, "emissions") and isinstance(model.emissions, torch.Tensor):
            return model.emissions
    except Exception:
        pass

    # 4) Model‑specific helper
    if hasattr(model, "get_output_score"):
        try:
            return model.get_output_score(input_ids, attention_mask=attention_mask)
        except TypeError:
            return model.get_output_score(input_ids, attention_mask)

    raise RuntimeError(
        "Could not obtain emissions from model. Expose compute_emissions(...) or forward(..., return_emissions=True)."
    )


def decode_with_rule_bias(
    model: Any,
    batch: Sequence[torch.Tensor],
    batch_features: List[Any],
    device,
    label2id: Dict[str,int],
    alpha_pos: float = 3.0,
) -> List[List[int]]:
    """Return CRF‑decoded label IDs with rule bias added to emissions.

    batch: (input_ids, attention_mask, [labels])
    batch_features: matching InputFeatures for this batch (same order)
    label2id: mapping like cfg.l2i
    alpha_pos: strength of the rule bias (+logit)
    """
    # input_ids, attention_mask = batch[0], batch[1]
    # device = input_ids.device

    input_ids, attention_mask = batch[0], batch[1]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Base emissions
    emissions = _get_emissions(model, input_ids, attention_mask)  # [B,T,C]

    # Build aligned bias (match emission width)
    num_classes = emissions.size(-1)
    bias = build_batch_rule_bias(batch_features, attention_mask, label2id, device, alpha_pos=alpha_pos, num_classes=num_classes)

    # Decode with CRF
    emissions = emissions + bias
    if not hasattr(model, "crf"):
        raise RuntimeError("Model has no `.crf` for decode().")
    if hasattr(model.crf, "decode"):
        pred_ids = model.crf.decode(emissions, mask=attention_mask.bool())
    else:
        _, pred_ids = model.crf._viterbi_decode(emissions, attention_mask.bool())

    # Sanitize: map ids outside your label map (e.g., CRF START/STOP) to 'O'
    C_map = len(label2id)
    o_idx = int(label2id.get("O", 0))
    clean_pred_ids: List[List[int]] = []
    for seq in pred_ids:
        clean_seq = []
        for tid in seq:
            t = int(tid) if not isinstance(tid, int) else tid
            if t < 0 or t >= C_map:
                t = o_idx
            clean_seq.append(t)
        clean_pred_ids.append(clean_seq)

    return clean_pred_ids

# ---------------------------------------------------------------------------
# Tiny self‑check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class _Feat:  # mimic InputFeatures
        def __init__(self, text): self.text = text

    toks = ["[CLS]","If","a","material","breach","occurs",",","the","Provider","may","terminate","this","Agreement",".","[SEP]"]
    feats = [_Feat(toks)]
    mask = torch.tensor([[1]*len(toks)])
    l2i = {k:i for i,k in enumerate([
        "<pad>","<start>","<eos>","O",
        "B-ACTION","I-ACTION","B-ASSET","I-ASSET",
        "B-PARTY_ASSIGNER","I-PARTY_ASSIGNER","B-PARTY_ASSIGNEE","I-PARTY_ASSIGNEE",
        "B-RULE_TYPE","I-RULE_TYPE","B-CONSTRAINT","I-CONSTRAINT","B-DUTY","I-DUTY",
        "B-CONSEQUENCE","I-CONSEQUENCE",
    ])}
    B,T,C = 1,len(toks),len(l2i)
    dummy_emissions = torch.zeros(B,T,C)
    bias = build_batch_rule_bias(feats, mask, l2i, device=torch.device("cpu"), alpha_pos=3.0, num_classes=C)
    print("bias shape:", bias.shape)  # [1, T, C]
