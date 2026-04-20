import argparse
import torch
import json
import pathlib
import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader


def print_and_save_params(config, filepath):
    exclude_keys = {'tokenizer', 'model', 'id2rel', 'rel_vocab'}
    params = {str(key): (vars(value) if hasattr(value, '__dict__') else value) for key, value in config.__dict__.items() if key not in exclude_keys}


    # Convert non-serializable objects to strings
    for key, value in params.items():
        if isinstance(value, torch.device):
            params[key] = str(value)
        elif isinstance(value, pathlib.Path):
            params[key] = str(value)
        elif isinstance(value, set):
            params[key] = list(value)
        elif callable(value):
            params[key] = str(value)
        elif isinstance(value, object) and not isinstance(value, (str, int, float, list, dict)):
            params[key] = str(value)

    # Save parameters to file
    try:
        # filename = (
        #     f"{config.model_name}_ep{config.num_epochs}_bz{config.batch_size}"
        #     f"_lr{config.learning_rate}_ml{config.max_length}_lt{config.label_type}_num_layers{config.rnn_layers}_parameters.json"
        # )

        filename = (
            f"{config.save_path}_parameters.json"
        )



        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
        print("params:", params)
        print(f"Parameters saved to {filepath}")
    except TypeError as e:
        print(f"Error saving parameters: {e}")


def get_ner_BIO(label_list):
    """
    Extract entity spans from BIO tags.
    Output format matches your BMES extractor: 'TYPE[start,end]'.
    Robust to broken BIO (e.g., I- without preceding B-).
    """
    stand_matrix = []
    cur_type = None
    start = None

    for i, lab in enumerate(label_list):
        if lab == 'O' or lab == '<pad>' or lab == '<start>' or lab == '<eos>':
            if cur_type is not None:
                stand_matrix.append(f"{cur_type}[{start},{i-1}]")
                cur_type, start = None, None
            continue

        if lab.startswith('B-'):
            # close previous
            if cur_type is not None:
                stand_matrix.append(f"{cur_type}[{start},{i-1}]")
            cur_type = lab[2:]
            start = i

        elif lab.startswith('I-'):
            typ = lab[2:]
            if cur_type is None:
                # ill-formed I-: start a new span
                cur_type, start = typ, i
            elif typ != cur_type:
                # type switched: close previous, start new
                stand_matrix.append(f"{cur_type}[{start},{i-1}]")
                cur_type, start = typ, i
        # else: ignore unknown tags safely

    if cur_type is not None:
        stand_matrix.append(f"{cur_type}[{start},{len(label_list)-1}]")
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position: input_len] + input_string[0: target_position]
    return output_string


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def bio_to_bioes(label_seq):
    """
    Convert a sequence of BIO tags into BIOES tags.
    - 'O', '<pad>', '<start>', '<eos>' are kept as-is.
    - Robust to a bit of broken BIO (I- without preceding B- etc.).
    - Does NOT modify tags that are already non-B/I (e.g. S-/E-/M-), so you can call it safely once.
    """
    specials = {'O', '<pad>', '<start>', '<eos>'}
    n = len(label_seq)
    bioes = ['O'] * n
    i = 0

    while i < n:
        lab = label_seq[i]
        # keep non-entity & special tokens
        if lab in specials or not (lab.startswith('B-') or lab.startswith('I-')):
            bioes[i] = lab
            i += 1
            continue

        # start of a span (treat both B- and broken I- as a new span)
        typ = lab[2:]
        j = i + 1
        while j < n:
            nxt = label_seq[j]
            if nxt.startswith('I-') and nxt[2:] == typ:
                j += 1
                continue
            # either different type, 'O', or a new B- => end of this span
            break

        span_len = j - i
        if span_len == 1:
            bioes[i] = f"S-{typ}"
        else:
            bioes[i] = f"B-{typ}"
            for k in range(i + 1, j - 1):
                bioes[k] = f"I-{typ}"
            bioes[j - 1] = f"E-{typ}"

        i = j

    return bioes


def spans_to_bioes_over_words(text, entities, tokenizer):
    """
    Convert span entities → BIOES per word.
    This is a thin wrapper around spans_to_bio_over_words + bio_to_bioes.
    """
    words, bio_labels = spans_to_bio_over_words(text, entities, tokenizer)
    bioes_labels = bio_to_bioes(bio_labels)
    return words, bioes_labels


# 利用标签字典恢复真实的预测标签
def recover_label(pred_var, gold_var, l2i_dic, i2l_dic):
    assert len(pred_var) == len(gold_var)
    pred_variable = []
    gold_variable = []

    for i in range(len(gold_var)):
        start_index = gold_var[i].index(l2i_dic['<start>'])
        end_index = gold_var[i].index(l2i_dic['<eos>'])
        pred_variable.append(pred_var[i][start_index:end_index])
        gold_variable.append(gold_var[i][start_index:end_index])

    pred_label = []
    gold_label = []
    for j in range(len(gold_variable)):
        pred_label.append([i2l_dic[t] for t in pred_variable[j]])
        gold_label.append([i2l_dic[t] for t in gold_variable[j]])

    return pred_label, gold_label


# 计算NER的关键指标
def get_ner_fmeasure(golden_lists, predict_lists, label_type='BIO'):
    sent_num = len(golden_lists)
    golden_full, predict_full, right_full = [], [], []
    right_tag, all_tag = 0, 0

    for idx in range(sent_num):
        gold = golden_lists[idx]
        pred = predict_lists[idx]

        # token accuracy
        for j in range(len(gold)):
            if gold[j] == pred[j]:
                right_tag += 1
        all_tag += len(gold)

        if label_type.upper() in ('BMES', 'BIOES'):
            gold_spans = get_ner_BMES(gold)
            pred_spans = get_ner_BMES(pred)
        else:  # BIO
            gold_spans = get_ner_BIO(gold)
            pred_spans = get_ner_BIO(pred)

        right = list(set(gold_spans).intersection(set(pred_spans)))
        golden_full += gold_spans
        predict_full += pred_spans
        right_full += right

    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)

    precision = -1 if predict_num == 0 else right_num / float(predict_num)
    recall    = -1 if golden_num  == 0 else right_num / float(golden_num)
    f_measure = -1 if (precision == -1 or recall == -1 or precision + recall <= 0) \
                    else 2 * precision * recall / (precision + recall)
    accuracy  = right_tag / float(all_tag) if all_tag > 0 else 0.0
    print('gold_num = ', golden_num, ' pred_num = ', predict_num, ' right_num = ', right_num)
    return accuracy, precision, recall, f_measure



# metrics_per_tag_and_type.py
import pandas as pd
from collections import defaultdict

def _bio_spans(labels):
    """
    Convert a BIO tag sequence into spans: list of (TYPE, start, end) inclusive.
    Robust to broken BIO (I- without B-, or type switches).
    """
    spans = []
    cur_type, start = None, None
    for i, lab in enumerate(labels):
        if lab == 'O' or lab in ('<pad>', '<start>', '<eos>'):
            if cur_type is not None:
                spans.append((cur_type, start, i-1))
                cur_type, start = None, None
            continue
        if lab.startswith('B-'):
            if cur_type is not None:
                spans.append((cur_type, start, i-1))
            cur_type, start = lab[2:], i
        elif lab.startswith('I-'):
            typ = lab[2:]
            if cur_type is None:
                cur_type, start = typ, i
            elif typ != cur_type:
                spans.append((cur_type, start, i-1))
                cur_type, start = typ, i
    if cur_type is not None:
        spans.append((cur_type, start, len(labels)-1))
    return spans


def _bioes_spans(labels):
    """
    Convert a BIOES (or BMES) tag sequence into spans: list of (TYPE, start, end) inclusive.
    Robust to broken sequences and fallback cases.
    """
    spans = []
    cur_type, start = None, None

    def close_span(end_idx):
        nonlocal cur_type, start
        if cur_type is not None and start is not None:
            spans.append((cur_type, start, end_idx))
        cur_type, start = None, None

    for i, lab in enumerate(labels):
        if lab in ('O', '<pad>', '<start>', '<eos>'):
            # close any ongoing span
            if cur_type is not None:
                close_span(i - 1)
            continue

        up = lab.upper()

        if up.startswith('S-'):
            # single-token span
            typ = up[2:]
            # close existing and add single span
            if cur_type is not None:
                close_span(i - 1)
            spans.append((typ, i, i))

        elif up.startswith('B-'):
            typ = up[2:]
            # close previous and start new
            if cur_type is not None:
                close_span(i - 1)
            cur_type, start = typ, i

        elif up.startswith('I-'):
            typ = up[2:]
            if cur_type is None:
                # broken sequence, start new span
                cur_type, start = typ, i
            elif typ != cur_type:
                # type switch, close previous and start new
                close_span(i - 1)
                cur_type, start = typ, i

        elif up.startswith('E-'):
            typ = up[2:]
            if cur_type is None or typ != cur_type:
                # inconsistent E-, treat as single-token span
                if cur_type is not None:
                    close_span(i - 1)
                spans.append((typ, i, i))
                cur_type, start = None, None
            else:
                # normal end
                close_span(i)

        # Other prefixes are ignored safely

    # close tail span if still open
    if cur_type is not None:
        close_span(len(labels) - 1)

    return spans


def compute_entity_metrics_by_type_bioes(gold_seqs, pred_seqs):
    """
    Entity-level (span) metrics per TYPE when labels are BIOES/BMES.
    Uses _bioes_spans instead of _bio_spans.
    """
    from collections import defaultdict
    import pandas as pd

    gold_sets = defaultdict(set)   # type -> set of (sent_idx, start, end)
    pred_sets = defaultdict(set)

    for si, (g, p) in enumerate(zip(gold_seqs, pred_seqs)):
        for typ, s, e in _bioes_spans(g):
            gold_sets[typ].add((si, s, e))
        for typ, s, e in _bioes_spans(p):
            pred_sets[typ].add((si, s, e))

    types = sorted(set(gold_sets.keys()) | set(pred_sets.keys()))
    rows = []
    for t in types:
        gset, pset = gold_sets[t], pred_sets[t]
        tp = len(gset & pset)
        fp = len(pset - gset)
        fn = len(gset - pset)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "type": t, "support": len(gset),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        })
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return df


def compute_entity_metrics_by_type(gold_seqs, pred_seqs):
    """
    Entity-level (span) metrics per TYPE (ASSET, ACTION, ...), excluding 'O'.
    Returns a pandas DataFrame with columns: type, support, tp, fp, fn, precision, recall, f1.
    """
    gold_sets = defaultdict(set)   # type -> set of (sent_idx, start, end)
    pred_sets = defaultdict(set)

    for si, (g, p) in enumerate(zip(gold_seqs, pred_seqs)):
        for typ, s, e in _bio_spans(g):
            gold_sets[typ].add((si, s, e))
        for typ, s, e in _bio_spans(p):
            pred_sets[typ].add((si, s, e))

    types = sorted(set(gold_sets.keys()) | set(pred_sets.keys()))
    rows = []
    for t in types:
        gset, pset = gold_sets[t], pred_sets[t]
        tp = len(gset & pset)
        fp = len(pset - gset)
        fn = len(gset - pset)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "type": t, "support": len(gset),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        })
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return df


def compute_token_metrics_by_tag(gold_seqs, pred_seqs, label_list):
    """
    Token-level metrics per TAG (B-ASSET, I-ASSET, ..., O).
    Returns a pandas DataFrame with columns: tag, support, tp, fp, fn, precision, recall, f1.
    """
    counts = {lbl: {"tp":0,"fp":0,"fn":0,"gold":0,"pred":0} for lbl in label_list}

    for g_seq, p_seq in zip(gold_seqs, pred_seqs):
        if len(g_seq) != len(p_seq):
            min_len = min(len(g_seq), len(p_seq))
            if min_len == 0:
                continue
            g_seq = g_seq[:min_len]
            p_seq = p_seq[:min_len]
        for g, p in zip(g_seq, p_seq):
            if g not in counts: counts[g] = {"tp":0,"fp":0,"fn":0,"gold":0,"pred":0}
            if p not in counts: counts[p] = {"tp":0,"fp":0,"fn":0,"gold":0,"pred":0}
            counts[g]["gold"] += 1
            counts[p]["pred"] += 1
            if p == g:
                counts[g]["tp"] += 1
            else:
                counts[p]["fp"] += 1
                counts[g]["fn"] += 1

    rows = []
    for tag, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "tag": tag,
            "support": c["gold"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        })
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return df


# ===================== BIOES span utils + heuristics =====================

def spans_to_bioes_tags(num_tokens, spans):
    """
    Build a BIOES label sequence from spans = [(TYPE, start, end), ...].
    Assumes spans do not overlap.
    """
    tags = ['O'] * num_tokens
    # sort by (start, end) to have a deterministic order
    spans = sorted(spans, key=lambda x: (x[1], x[2], x[0]))
    for typ, s, e in spans:
        length = e - s + 1
        if length <= 0:
            continue
        if length == 1:
            tags[s] = f"S-{typ}"
        else:
            tags[s] = f"B-{typ}"
            for i in range(s + 1, e):
                tags[i] = f"I-{typ}"
            tags[e] = f"E-{typ}"
    return tags

def _adjust_spans_for_type(spans, tokens, entity_type, occupied,
                           max_gap=2, max_extend=15):
    """
    Heuristic refinement for spans of a single entity type
    (PURPOSE or CONSTRAINT).

    spans: list of (TYPE, start, end) for this entity_type only.
    tokens: list of tokens (same length as label sequence).
    occupied: list of types or None, from the ORIGINAL spans, used to avoid
              crossing other entity types.
    """
    if not spans:
        return []

    # -------- 1) Merge close spans separated by small connectors --------
    allowed_gap_tokens = {",", ";", "and", "or", "and/or", "/", "(", ")"}

    spans_sorted = sorted(spans, key=lambda x: x[1])
    merged = []
    i = 0
    while i < len(spans_sorted):
        _, cur_start, cur_end = spans_sorted[i]
        j = i + 1
        while j < len(spans_sorted):
            _, next_start, next_end = spans_sorted[j]

            # overlapping or touching directly -> just merge
            if next_start <= cur_end + 1:
                cur_end = max(cur_end, next_end)
                j += 1
                continue

            gap_start = cur_end + 1
            gap_end = next_start - 1
            gap_len = max(0, gap_end - gap_start + 1)
            if gap_len > max_gap:
                break

            # Check gap content: must be all allowed_connector tokens, and not
            # occupied by other entity types.
            ok = True
            for k in range(gap_start, next_start):
                if occupied[k] is not None and occupied[k] != entity_type:
                    ok = False
                    break
                if tokens[k].lower() not in allowed_gap_tokens:
                    ok = False
                    break

            if not ok:
                break

            # merge
            cur_end = next_end
            j += 1

        merged.append((entity_type, cur_start, cur_end))
        i = j

    # -------- 2) Extend spans to clause boundaries (conservatively) --------
    n = len(tokens)
    extended = []

    # trigger words to decide when to extend
    if entity_type == "PURPOSE":
        trigger_tokens = {"to", "for", "in", "so"}
        trigger_prev = {"for", "to"}
    else:  # CONSTRAINT
        trigger_tokens = {"subject", "under", "unless", "provided", "outside",
                          "prior", "except", "notwithstanding", "in"}
        trigger_prev = {"subject", "under", "in"}

    strong_boundaries = {".", ";", "?", "!"}

    for _, start, end in merged:
        new_end = end
        start_tok = tokens[start].lower()
        prev_tok = tokens[start - 1].lower() if start > 0 else None

        extend_span = False

        # Decide whether this looks like a clause that should be extended
        if start_tok in trigger_tokens:
            extend_span = True
        elif prev_tok in trigger_prev:
            extend_span = True
        # small safety net: very short PURPOSE/CONSTRAINT (likely truncated)
        span_len = end - start + 1
        if span_len <= 3:
            extend_span = True

        if extend_span:
            steps = 0
            k = end + 1
            while k < n and steps < max_extend:
                tok = tokens[k]
                # Stop at strong punctuation
                if tok in strong_boundaries:
                    break
                # do not invade other entity types
                if occupied[k] is not None and occupied[k] != entity_type:
                    break
                # we only extend over tokens that the model thought were 'O'
                # (otherwise we'd be overriding another entity label)
                # caller is responsible for checking original labels if needed.

                new_end = k
                steps += 1
                k += 1

        extended.append((entity_type, start, new_end))

    return extended


def heuristic_postprocess_purpose_constraint(tokens, labels):
    """
    Main entry point for the heuristic layer.

    Args:
        tokens: list of tokens for this sentence (same length as labels).
        labels: list of BIOES tags for this sentence (strings).

    Returns:
        new_labels: list of BIOES tags after heuristic refinement
                    for PURPOSE and CONSTRAINT (other types unchanged).
    """
    # 1) extract spans from BIOES labels
    spans = _bioes_spans(labels)
    n = len(tokens)

    # 2) build occupancy map from ORIGINAL spans
    occupied = [None] * n
    for typ, s, e in spans:
        for i in range(s, e + 1):
            if 0 <= i < n:
                occupied[i] = typ

    # 3) group spans by type
    by_type = {}
    for typ, s, e in spans:
        by_type.setdefault(typ, []).append((typ, s, e))

    final_spans = []

    for typ, span_list in by_type.items():
        if typ in ("PURPOSE", "CONSTRAINT"):
            refined = _adjust_spans_for_type(span_list, tokens, typ, occupied)
            final_spans.extend(refined)
        else:
            final_spans.extend(span_list)

    # 4) rebuild BIOES tags from refined spans
    new_labels = spans_to_bioes_tags(n, final_spans)
    return new_labels










# Reuse your InputFeatures class from your file
class InputFeatures(object):
    def __init__(self, text, label, input_id, label_id, input_mask):
        self.text = text
        self.label = label
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

# ---------- SPAN → BIO conversion helpers ----------

LABEL_PRIORITY = [
    # Highest priority first (later labels overwrite earlier char mask positions)
    "ASSET",
    "ACTION",
    "RULE_TYPE-PROHIBITION",
    "RULE_TYPE-OBLIGATION",
    "RULE_TYPE-PERMISSION",
    "ASSIGNEE",
    "CONSTRAINT",
    "PURPOSE",
]
_PRIO = {lab: i for i, lab in enumerate(LABEL_PRIORITY)}

def _word_pretok_with_offsets(text, tokenizer):
    """
    Produce (words, spans) pairs using the SAME logic as _words_from_text(),
    but also return the character (start, end) offsets for each word.

    We try tokenizer pretokenizer if available; otherwise fallback to whitespace (\S+).
    """
    # 1) fast tokenizer backend pre-tokenizer
    if getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "backend_tokenizer") and tokenizer.backend_tokenizer is not None:
        pre = tokenizer.backend_tokenizer.pre_tokenizer
        if pre is not None:
            pairs = pre.pre_tokenize_str(text)  # [(token, (start,end)), ...]
            words = [tok for tok, _ in pairs]
            spans = [span for _, span in pairs]
            return words, spans

    # 2) basic_tokenizer (slow) — offsets are not given, so we fall back to regex to recover spans
    if hasattr(tokenizer, "basic_tokenizer") and tokenizer.basic_tokenizer is not None:
        # reproduce same tokens
        toks = tokenizer.basic_tokenizer.tokenize(text, never_split=getattr(tokenizer, "all_special_tokens", []))
        # recover spans with a scan
        spans = []
        idx = 0
        for tok in toks:
            pos = text.find(tok, idx)
            if pos == -1:
                # fallback: skip, mark as zero-length (won't be labeled)
                spans.append((idx, idx))
            else:
                spans.append((pos, pos + len(tok)))
                idx = pos + len(tok)
        return toks, spans

    # 3) fallback: whitespace (\S+) tokens with offsets
    import re as _re
    words = []
    spans = []
    for m in _re.finditer(r"\S+", text):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans

def _build_char_mask(text, entities):
    """
    Build a char-wise label mask from span entities.
    Overlaps resolved deterministically by LABEL_PRIORITY (later in list wins).
    """
    char = ["O"] * len(text)
    # sort entities so higher priority writes last (overwrite)
    ents = sorted(entities, key=lambda e: _PRIO.get(e["label"], 999))
    for ent in ents:
        s = max(0, min(int(ent["start"]), len(text)))
        e = max(0, min(int(ent["end"]),   len(text)))
        lab = ent["label"]
        for i in range(s, e):
            char[i] = lab
    return char

def spans_to_bio_over_words(text, entities, tokenizer):
    """
    Convert span entities → BIO per WORD (your training format),
    using tokenizer pretokenization (or whitespace) as word boundaries.
    Returns: words, labels (same length)
    """
    words, word_spans = _word_pretok_with_offsets(text, tokenizer)
    char_mask = _build_char_mask(text, entities)

    labels = []
    prev_lab = "O"
    for (ws, we) in word_spans:
        lab = "O"
        # inspect any char inside this word span
        for j in range(ws, we):
            if 0 <= j < len(char_mask) and char_mask[j] != "O":
                lab = char_mask[j]
                break
        if lab == "O":
            labels.append("O")
            prev_lab = "O"
        else:
            prefix = "B-" if prev_lab != lab else "I-"
            labels.append(prefix + lab)
            prev_lab = lab
    return words, labels


def _read_jsonl(path, tokenizer=None):
    """
    Read a JSONL file and return a list of (text, label_seq).
    Accepts two formats:
      A) {"text": "...", "label": [...]}  # token-aligned BIO (existing)
      B) {"text": "...", "entities": [{"start":..,"end":..,"label":"..."}]}  # span format
         -> will be converted to BIO aligned to your pre-tokenization.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "label" in obj:
                # already token-aligned
                samples.append((obj["text"], obj["label"]))
                continue

            if "entities" in obj:
                if tokenizer is None:
                    raise ValueError(
                        f"{path} line {ln}: found 'entities' but no tokenizer was provided to _read_jsonl."
                    )
                text = obj["text"]
                ents = obj.get("entities", [])
                # validate entity keys
                for e in ents:
                    if not all(k in e for k in ("start", "end", "label")):
                        raise ValueError(f"{path} line {ln}: malformed entity {e}")
                # convert
                words, bio = spans_to_bio_over_words(text, ents, tokenizer)
                # We return (text, labels). Downstream will re-tokenize words again;
                # since spans_to_bio used the same pretokenization, the length will match.
                samples.append((text, bio))
                continue

            raise ValueError(f"{path} line {ln}: expected 'label' or 'entities' keys.")
    return samples


def _read_jsonl_bioes(path, tokenizer=None):
    """
    Wrapper around _read_jsonl that converts BIO -> BIOES.
    Keeps the original _read_jsonl unchanged and reuses all its logic.
    """
    bio_samples = _read_jsonl(path, tokenizer=tokenizer)
    bioes_samples = []
    for text, label_seq in bio_samples:
        bioes_labels = bio_to_bioes(label_seq)
        bioes_samples.append((text, bioes_labels))
    return bioes_samples



# def _words_from_text(text):
#     """Whitespace tokenization that matches your labels."""
#     # If you later change how labels are generated, adjust this splitter accordingly.
#     return text.split()

# put near the top of get_data.py
import re

def _words_from_text(text, tokenizer):
    """
    Use the tokenizer's pre-tokenization to split into 'words' that match your labels
    (punctuation and hyphens become separate tokens). Falls back to a regex splitter.
    """
    # 1) BertTokenizer (slow) exposes basic_tokenizer
    if hasattr(tokenizer, "basic_tokenizer") and tokenizer.basic_tokenizer is not None:
        # never_split keeps special tokens intact if present
        return tokenizer.basic_tokenizer.tokenize(text, never_split=getattr(tokenizer, "all_special_tokens", []))

    # 2) Fast tokenizers expose the underlying pre-tokenizer
    if getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "backend_tokenizer"):
        pre = tokenizer.backend_tokenizer.pre_tokenizer
        if pre is not None:
            # returns list of (token, (start, end))
            return [tok for tok, _ in pre.pre_tokenize_str(text)]

    # 3) Fallback: punctuation-aware regex
    # return re.findall(r"\w+|[^\w\s]", text)
    return text.split()  # whitespace split only (aligns with typical BIO labeling)

def encode_and_align_labels(words, labels, tokenizer, label2id):
    # Optional: truncate words/labels BEFORE tokenization to avoid partial truncation
    # that would otherwise drop tokens without dropping labels.
    # Keep at most max_length - specials (2 for RoBERTa/DeBERTa)
    max_words = tokenizer.model_max_length - 2 if tokenizer.model_max_length and tokenizer.model_max_length != int(1e30) else None
    if max_words is not None and len(words) > max_words:
        words  = words[:max_words]
        labels = labels[:max_words]

    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        # choose one padding strategy; for dataloaders batching is easier with padding
        padding=False,                 # or "max_length" during batching
        max_length=tokenizer.model_max_length,
        return_attention_mask=True,
    )

    word_ids = enc.word_ids()
    aligned = []
    prev_wid = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)       # special tokens -> ignored by loss
        else:
            # First subword gets the word label; subsequent subwords get -100
            if wid != prev_wid:
                aligned.append(label2id[labels[wid]])
                prev_wid = wid
            else:
                aligned.append(-100)   # mask out continuation pieces
    enc["labels"] = aligned
    return enc


def encode_with_alignment(text, labels, tokenizer, l2i, max_length):
    # 1) Preserve the dataset’s word boundaries (whitespace) -> matches label count
    words = text.split()
    assert len(words) == len(labels), f"mismatch: {len(words)} vs {len(labels)}"

    # 2) Tokenize with alignment to words
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    word_ids  = enc.word_ids()  # list of word index or None for specials

    # 3) Build CRF labels/mask: first subword gets the label; others are ignored
    label_ids = [l2i["<pad>"]] * len(input_ids)
    crf_mask  = [0] * len(input_ids)
    prev_wi = None
    for i, wi in enumerate(word_ids):
        if wi is None:
            # CLS/SEP -> <start>/<eos>
            if i == 0:
                label_ids[i] = l2i["<start>"]; crf_mask[i] = 1
            elif i == len(word_ids) - 1:
                label_ids[i] = l2i["<eos>"];   crf_mask[i] = 1
            continue
        if wi != prev_wi:  # first subword of this word
            label_ids[i] = l2i[labels[wi]]
            crf_mask[i]  = 1
        # else: subsequent subword stays <pad> with mask 0
        prev_wi = wi

    return input_ids, attn_mask, label_ids, crf_mask


def _encode_sample(words, labels, tokenizer, l2i, max_length, cls_id, sep_id, pad_id):
    """
    Build InputFeatures for one sample:
    - keep only the FIRST subword of each word (to preserve 1:1 labels)
    - add [CLS]/[SEP] (or BOS/EOS fallback) and <start>/<eos> labels
    - pad/truncate to max_length
    """
    assert len(words) == len(labels), f"Words and labels length mismatch: {len(words)} vs {len(labels)}"

    # room for special tokens
    max_words = max_length - 2
    words = words[:max_words]
    labels = labels[:max_words]

    # print("before:", max_words, words, labels, sep="\n")

    # Special-token aware build
    input_ids = [cls_id]
    input_mask = [1]
    label_ids = [l2i["<start>"]]

    kept_tokens = ["[CLS]"]  # purely for debug readability

    for w, lab in zip(words, labels):
        # Tokenize word -> subwords; keep only first subword to align labels
        sub_tokens = tokenizer.tokenize(w)
        if not sub_tokens:
            sub_tokens = [tokenizer.unk_token] if tokenizer.unk_token is not None else ["[UNK]"]
        first_piece = sub_tokens[0]
        tok_id = tokenizer.convert_tokens_to_ids(first_piece)
        if tok_id is None:
            tok_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100
        input_ids.append(tok_id)
        input_mask.append(1)
        # label for the word
        if lab not in l2i:
            raise KeyError(f"Label '{lab}' not found in label map.")
        label_ids.append(l2i[lab])
        kept_tokens.append(first_piece)



    # Append [SEP] and <eos>
    input_ids.append(sep_id)
    input_mask.append(1)
    label_ids.append(l2i["<eos>"])
    kept_tokens.append("[SEP]")

    # Pad
    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)
        label_ids.append(pad_id)

    # Truncate hard if necessary (shouldn't happen due to max_words)
    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    label_ids = label_ids[:max_length]

    # print("\n kept_tokens, input_ids, label_ids, input_mask", kept_tokens, input_ids, label_ids, input_mask)

    return InputFeatures(text=kept_tokens, label=None, input_id=input_ids, label_id=label_ids, input_mask=input_mask)


def _build_features(jsonl_path, tokenizer, l2i, max_length, label_type):
    """Read a JSONL file and convert to a list of InputFeatures."""

    if label_type.upper() == "BIOES":
        samples = _read_jsonl_bioes(jsonl_path, tokenizer=tokenizer)
    else:
        samples = _read_jsonl(jsonl_path, tokenizer=tokenizer)

    # Resolve special token ids with safe fallbacks
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 101  # BERT [CLS]
    )
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 102  # BERT [SEP]
    )
    pad_label_id = l2i["<pad>"]  # from your Config

    features = []
    n_max_length = 0
    for text, label_seq in samples:

        # words = _words_from_text(text)
        words = _words_from_text(text, tokenizer)
        if len(words) > max_length:
            # tiny improvement: keep the global max, not just the last long one
            n_max_length = max(n_max_length, len(words))

        if len(words) != len(label_seq):
            raise ValueError(
                f"Label alignment error in {jsonl_path}:\n"
                f"  words: {len(words)} != labels: {len(label_seq)}\n"
                f"  text snippet: {text[:120]}..."
            )

        feat = _encode_sample(
            words,
            label_seq,          # already BIOES
            tokenizer,
            l2i,
            max_length,
            cls_id,
            sep_id,
            pad_label_id,
        )
        features.append(feat)

    print("maximum length of sentence:", n_max_length)
    return features



def get_data(cfg):
    """
    Returns: train_loader, dev_loader, test_loader
    Each batch is (input_ids, attention_mask, label_ids) as LongTensors.
    """
    tokenizer = cfg.tokenizer
    l2i_dic = cfg.l2i

    max_length = cfg.max_length  # ← was cfg.pad_size
    batch_size = cfg.batch_size

    # Resolve file paths from cfg (supports either dev.txt or valid.txt)
    train_file = getattr(cfg, "train_path", os.path.join(cfg.data_path, "train.txt"))
    dev_file = getattr(cfg, "dev_path", None)
    if dev_file is None or (not os.path.isfile(dev_file) and os.path.basename(dev_file) == "dev.txt"):
        alt = os.path.join(cfg.data_path, "valid.txt")
        dev_file = alt if os.path.isfile(alt) else os.path.join(cfg.data_path, "dev.txt")
    test_file = getattr(cfg, "test_path", os.path.join(cfg.data_path, "test.txt"))

    # Build features
    print("Loading & encoding train/dev/test ...")
    train_feats = _build_features(train_file, tokenizer, l2i_dic, max_length, cfg.label_type)
    dev_feats   = _build_features(dev_file,   tokenizer, l2i_dic, max_length, cfg.label_type)
    test_feats  = _build_features(test_file,  tokenizer, l2i_dic, max_length, cfg.label_type)

    # Tensors
    def to_tensors(features):
        ids   = torch.LongTensor([f.input_id  for f in features])
        masks = torch.LongTensor([f.input_mask for f in features])
        tags  = torch.LongTensor([f.label_id  for f in features])
        return ids, masks, tags

    train_ids, train_masks, train_tags = to_tensors(train_feats)
    dev_ids,   dev_masks,   dev_tags   = to_tensors(dev_feats)
    test_ids,  test_masks,  test_tags  = to_tensors(test_feats)

    # Datasets / loaders
    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    dev_dataset   = TensorDataset(dev_ids,   dev_masks,   dev_tags)
    test_dataset  = TensorDataset(test_ids,  test_masks,  test_tags)

    # For dev/test, shuffle=False is typical for evaluation; change if you prefer.
    train_loader = DataLoader(train_dataset, shuffle=True,  batch_size=batch_size)
    dev_loader   = DataLoader(dev_dataset,   shuffle=False, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  shuffle=False, batch_size=batch_size)

    return train_loader, dev_loader, test_loader, dev_feats, test_feats



if __name__ == '__main__':
    pass
