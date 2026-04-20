import json
import re
from pathlib import Path


# --------- CONFIG: tweak these if needed ---------
INPUT_PATH = Path("./labelled_data/odrl_simodrl_spans.jsonl")
OUTPUT_PATH = Path("./labelled_data/odrl_simodrl_spans_constraint_splitted2.jsonl")

# minimum characters for left/right side of "and"/"or" to be considered a real clause split
MIN_LEFT_CHARS = 40
MIN_RIGHT_CHARS = 25
# minimum length of a resulting segment to keep it as a constraint
MIN_SEG_CHARS = 5


def find_constraint_splits(span_text: str):
    """
    Find candidate split positions (local char indices) inside a constraint span.

    Returns a sorted list of positions in [0, len(span_text)] where we
    consider splitting BETWEEN characters.
    """
    splits = set()

    # 1) Split after semicolons
    for i, ch in enumerate(span_text):
        if ch == ";":
            # split AFTER this character
            splits.add(i + 1)

    # 2) Split before 'and' / 'or' when they likely signal separate clauses
    #    We keep the conjunction with the RIGHT segment (split at its start).
    for match in re.finditer(r"\b(and|or)\b", span_text, flags=re.IGNORECASE):
        conj_start = match.start()
        left = span_text[:conj_start]
        right = span_text[conj_start:]

        # crude but effective clause-length heuristic
        if len(left.strip()) >= MIN_LEFT_CHARS and len(right.strip()) >= MIN_RIGHT_CHARS:
            splits.add(conj_start)

    # Always include start and end
    splits.add(0)
    splits.add(len(span_text))

    split_positions = sorted(splits)
    # deduplicate adjacent splits (e.g., ";;" or weird patterns)
    dedup_positions = [split_positions[0]]
    for pos in split_positions[1:]:
        if pos != dedup_positions[-1]:
            dedup_positions.append(pos)

    return dedup_positions


def split_constraint_span(text: str, start: int, end: int):
    """
    Given the full text and the char-span [start, end) of a CONSTRAINT,
    return a list of (new_start, new_end) tuples for the split constraint spans.

    If no meaningful split is found, return an empty list (caller can keep the original).
    """
    span_text = text[start:end]
    if not span_text.strip():
        return []

    split_positions = find_constraint_splits(span_text)
    segments = []

    for i in range(len(split_positions) - 1):
        local_s = split_positions[i]
        local_e = split_positions[i + 1]
        if local_e <= local_s:
            continue

        # raw segment text
        raw_seg = span_text[local_s:local_e]

        # 1) trim whitespace and simple punctuation on both sides
        s = local_s
        e = local_e

        while s < e and span_text[s] in " \t\r\n,;:":
            s += 1
        while e > s and span_text[e - 1] in " \t\r\n,;:":
            e -= 1

        if e <= s:
            continue

        # 2) drop leading "and" / "or" / "and/or" if they start the segment
        seg_sub = span_text[s:e]
        # strip leading whitespace in subsegment, but track shift
        leading_ws = len(seg_sub) - len(seg_sub.lstrip())
        s += leading_ws
        seg_sub = span_text[s:e]
        lower_sub = seg_sub.lower()

        for pref in ("and/or ", "and ", "or "):
            if lower_sub.startswith(pref):
                s += len(pref)
                seg_sub = span_text[s:e]
                lower_sub = seg_sub.lower()
                break

        # 3) drop trailing " and"/" or" if they end the segment
        # (less common but can happen)
        while e > s:
            tail = span_text[s:e]
            lower_tail = tail.lower().rstrip()
            trimmed_len = len(lower_tail)
            if tail.lower().endswith(" and") and trimmed_len >= MIN_SEG_CHARS:
                e = s + trimmed_len - len(" and")
            elif tail.lower().endswith(" or") and trimmed_len >= MIN_SEG_CHARS:
                e = s + trimmed_len - len(" or")
            else:
                break

        if e <= s:
            continue

        final_seg = span_text[s:e]
        if len(final_seg.strip()) < MIN_SEG_CHARS:
            continue

        new_start = start + s
        new_end = start + e
        segments.append((new_start, new_end))

    # If we didn't produce anything useful, signal to keep original
    return segments


def process_file(input_path: Path, output_path: Path):
    total_lines = 0
    total_entities_before = 0
    total_entities_after = 0
    total_constraints_before = 0
    total_constraints_after = 0
    total_constraints_splitted = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            obj = json.loads(line)

            text = obj.get("text", "")
            entities = obj.get("entities", []) or obj.get("spans", [])

            total_entities_before += len(entities)

            new_entities = []

            for ent in entities:
                start = ent["start"]
                end = ent["end"]
                label = ent["label"]

                # 1) Merge PURPOSE into CONSTRAINT
                if label == "PURPOSE":
                    label = "CONSTRAINT"

                if label != "CONSTRAINT":
                    # keep other labels as-is
                    new_entities.append({"start": start, "end": end, "label": label})
                    continue

                # Here: label == CONSTRAINT (merged)
                total_constraints_before += 1

                segments = split_constraint_span(text, start, end)

                if not segments:
                    # no split -> keep original
                    new_entities.append({"start": start, "end": end, "label": "CONSTRAINT"})
                    total_constraints_after += 1
                else:
                    # we split this constraint into multiple sub-constraints
                    total_constraints_splitted += 1
                    for (ns, ne) in segments:
                        new_entities.append({"start": ns, "end": ne, "label": "CONSTRAINT"})
                        total_constraints_after += 1

            total_entities_after += len(new_entities)

            obj["entities"] = new_entities
            # if original used "spans" instead of "entities", you can also set that:
            # obj["spans"] = new_entities

            fout.write(json.dumps(obj, ensure_ascii=False))
            fout.write("\n")

    print("Done.")
    print(f"Lines (rules): {total_lines}")
    print(f"Entities before: {total_entities_before}")
    print(f"Entities after:  {total_entities_after}")
    print(f"CONSTRAINT before (after merge): {total_constraints_before}")
    print(f"CONSTRAINT after  (splitted):    {total_constraints_after}")
    print(f"Constraints that were actually split: {total_constraints_splitted}")


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    process_file(INPUT_PATH, OUTPUT_PATH)
