import os
import re
from statistics import mean
from typing import List, Dict


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer using regex to split on word boundaries."""
    return re.findall(r"\b\w+\b", text)


def list_text_files(folder: str) -> List[str]:
    """Return a sorted list of .txt file paths in the folder."""
    names = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]
    names.sort()
    return [os.path.join(folder, f) for f in names]


def read_text(path: str) -> str:
    """Read a text file with robust UTF-8 handling."""
    # utf-8-sig handles possible BOM; errors='replace' avoids crashes on odd bytes
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def merge_files(input_folder: str, output_file: str) -> None:
    """
    Merge all .txt files from input_folder into output_file.
    Skips files with no tokens (truly empty/whitespace).
    Adds a single newline between files; preserves original line breaks within files.
    """
    files = list_text_files(input_folder)

    with open(output_file, "w", encoding="utf-8") as out:
        for path in files:
            content = read_text(path).strip()
            # Skip if there are no tokens at all
            if not tokenize(content):
                continue
            out.write(content.rstrip() + "\n")



def analyze_file(path: str) -> Dict[str, float]:
    """
    Treat each non-empty line as a sentence.
    Compute line count and token stats (avg, max, min).
    Prints debug info for lines with a single token.
    Returns a dict of stats.
    """
    line_lengths: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        line_no = 0
        for raw in f:
            line_no += 1
            line = raw.strip()
            if not line:
                continue
            # If your merged file contains headers like "=== FILE: ... ===", skip them:
            if line.startswith("===") and line.endswith("==="):
                continue
            tokens = tokenize(line)
            n = len(tokens)
            if n == 0:
                continue
            line_lengths.append(n)
            if n == 1:
                print(f"Debug: single-token line at #{line_no}: tokens={tokens}")

    if not line_lengths:
        return {
            "num_lines": 0,
            "avg_tokens": 0.0,
            "max_tokens": 0,
            "min_tokens": 0,
        }

    return {
        "num_lines": len(line_lengths),
        "avg_tokens": float(f"{mean(line_lengths):.2f}"),
        "max_tokens": max(line_lengths),
        "min_tokens": min(line_lengths),
    }


def main() -> None:
    # Configure paths
    input_folder = "output_after_process_script1"
    output_folder = "output_after_merge_process_script2"
    output_file = os.path.join(output_folder, "all_agreements.txt")

    ensure_dir(output_folder)
    merge_files(input_folder, output_file)
    print(f"✅ Merged all agreements into: {output_file}")

    stats = analyze_file(output_file)

    if stats["num_lines"] == 0:
        print("⚠️ No valid lines found in the merged file.")
        return

    print("\n📊 Statistical Summary:")
    print(f"Total number of lines (sentences): {stats['num_lines']}")
    print(f"Average token count per line: {stats['avg_tokens']}")
    print(f"Maximum token count in a line: {stats['max_tokens']}")
    print(f"Minimum token count in a line: {stats['min_tokens']}")


if __name__ == "__main__":
    main()


