import json
import csv
import random
from pathlib import Path
from typing import List, Tuple, Union


def _read_jsonl(path: Union[str, Path]) -> List[Tuple[str, List[str]]]:
    """Read a JSON-lines file where each line has keys 'text' (str) and 'label' (List[str])."""
    samples: List[Tuple[str, List[str]]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {p} line {ln}: {e}")
            if "text" not in obj or "label" not in obj:
                raise ValueError(f"Expected keys 'text' and 'label' in {p} line {ln}")
            samples.append((obj["text"], obj["label"]))
    return samples


def _read_txt(path: Union[str, Path]) -> List[str]:
    """Read a UTF-8 text file and return a list of lines (without trailing newlines)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    data = p.read_text(encoding="utf-8", errors="replace")
    # Strip potential UTF-8 BOM
    if data.startswith("\ufeff"):
        data = data.lstrip("\ufeff")

    return data.splitlines()

def _oneline(s: str) -> str:
    """Normalize to a single line of text."""
    return s.replace("\r", " ").replace("\n", " ").strip() if isinstance(s, str) else ""

if __name__ == "__main__":

    # Input paths (adjust if needed)
    odrl_set_path = './data_sharing_agreement_mt20_labels_o4-mini-more-tags-seeds-v3adv.jsonl'
    non_oderl_path = './non_odrl/non_odrl_v2.txt'  # (variable name has 'oderl' but path points to non_odrl)

    # Read inputs
    odrl_set = _read_jsonl(odrl_set_path)
    non_odrl_set = _read_txt(non_oderl_path)

    print("Read labeled ODRL JSONL samples:", len(odrl_set))
    print("Read non-ODRL text lines:", len(non_odrl_set))

    # Split into positives/negatives; add O-only JSONL rows into the negative set
    odrl_set_adv: List[str] = []
    added_from_odrl_O_only = 0

    for text, labels in odrl_set:
        if set(labels) == {"O"}:
            non_odrl_set.append(text)
            added_from_odrl_O_only += 1
        else:
            odrl_set_adv.append(text)

    print("Appended O-only rows from JSONL into negatives:", added_from_odrl_O_only)
    # ----- Compose final data (dicts) -----
    all_rows: List[dict] = []

    # Positives = label 1
    for t in odrl_set_adv:
        line = _oneline(t)
        if line:
            all_rows.append({"text": line, "label": 1})

    # Negatives = label 0
    for t in non_odrl_set:
        line = _oneline(t)
        if line:
            all_rows.append({"text": line, "label": 0})

    # ----- Shuffle deterministically -----
    rng = random.Random(14)
    rng.shuffle(all_rows)

    # ----- Write JSON Lines -----
    out_path = Path('odrl_classification_dataset.jsonl')  # output file
    with out_path.open('w', encoding='utf-8') as f:
        for obj in all_rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Wrote:", out_path.resolve())
    print("Total rows:", len(all_rows))
    print("Positives (label=1):", len(odrl_set_adv))
    print("Negatives (label=0):", len(all_rows) - len(odrl_set_adv))