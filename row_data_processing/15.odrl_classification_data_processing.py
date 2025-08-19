import json
from pathlib import Path
from typing import List, Union

def _read_jsonl(path):
    """Read a JSON-lines file: each line has keys 'text' (string) and 'label' (list of str)."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} line {ln}: {e}")
            if "text" not in obj or "label" not in obj:
                raise ValueError(f"Expected keys 'text' and 'label' in {path} line {ln}")
            samples.append((obj["text"], obj["label"]))
    return samples


def _read_txt(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    data = p.read_text(encoding="utf-8", errors="replace")
    # Strip a potential UTF-8 BOM
    if data.startswith("\ufeff"):
        data = data.lstrip("\ufeff")

    return data.splitlines()

if __name__ == "__main__":

    odrl_set_path = './data_sharing_agreement_mt20_labels_o4-mini-more-tags-seeds-v3adv.jsonl'
    non_oderl_path = './non_odrl/non_odrl_v2.txt'

    odrl_set = _read_jsonl(odrl_set_path)
    non_odrl_set = _read_txt(non_oderl_path)

    print("Reading odrl set...", non_odrl_set)


    print("Reading odrl set...", odrl_set)
    for sample in odrl_set:
        print(sample[0], sample[1])

