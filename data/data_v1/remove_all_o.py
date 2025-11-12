import json
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


def _write_jsonl(path: Union[str, Path], samples: List[Tuple[str, List[str]]]) -> None:
    """Write samples back to JSONL with keys 'text' and 'label'."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for text, labels in samples:
            json.dump({"text": text, "label": labels}, f, ensure_ascii=False)
            f.write("\n")


def _is_all_o(labels: List[str]) -> bool:
    """True if labels list is non-empty and every entry is 'O' (case-insensitive)."""
    return bool(labels) and all(str(l).strip().upper() == "O" for l in labels)


if __name__ == "__main__":

    # Input path (adjust if needed)
    odrl_set_path = "./data_sharing_agreement_mt20_labels_o4-mini-adv-version-v1.jsonl"

    # Read inputs
    odrl_set = _read_jsonl(odrl_set_path)
    print("Read labeled ODRL JSONL samples:", len(odrl_set))

    # Filter out samples with labels all 'O'
    kept: List[Tuple[str, List[str]]] = []
    removed_count = 0

    for text, labels in odrl_set:
        if _is_all_o(labels):
            removed_count += 1
            continue
        kept.append((text, labels))

    # Derive output path and write
    in_path = Path(odrl_set_path)
    out_path = in_path.with_name(in_path.stem + ".non_o.jsonl")
    _write_jsonl(out_path, kept)

    print(f"Kept: {len(kept)} | Removed (all-O): {removed_count}")
    print(f"Wrote filtered file to: {out_path}")
