#!/usr/bin/env python3
"""
Clean all extracted agreement .txt files in a folder and
write them, one per line, into agreement_text__row_clean.txt.
"""
import os
import re
import glob

def clean_text(text: str) -> str:
    # 1. Remove TOC entries
    text = re.sub(r'^\d+(\.\d+)*\s+.+\s+\d+$', '', text, flags=re.MULTILINE)
    # 2. Remove boilerplate headings/footers (all-caps lines, page numbers)
    text = re.sub(r'^[A-Z ]{5,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # 3. Kill hyphenated breaks
    text = re.sub(r'-\s*\n\s*', '', text)
    # 4. Drop template notes/instructions
    text = re.sub(r'(?ms)^Notes? for.*?(?=\n[A-Z])', '', text)
    # 5. Remove placeholders [..], <..>, {..}
    text = re.sub(r'\[.*?\]|<.*?>|\{.*?\}', '', text)
    # 6. Strip URLs, emails, bracketed refs, SI cites
    text = re.sub(r'https?://\S+|\S+@\S+|\[\d+\]|\(SI\s*\d+/\d+\)', '', text)
    # 7. Drop bullets and list markers
    text = re.sub(r'^[•\-\*\#]\s*', '', text, flags=re.MULTILINE)
    # 8. Remove leading section numbers
    text = re.sub(r'^\d+(\.\d+)*\s*', '', text, flags=re.MULTILINE)
    # 9. Cull signature & appendix blocks
    text = re.sub(r'(?ms)^(Signature:|Signed by|Appendix\b).*$', '', text)
    # 10. Normalize whitespace & flatten
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r' {2,}', ' ', text).strip()


    return text

def main(input_dir: str, output_file: str):
    cleaned_lines = []
    for path in glob.glob(os.path.join(input_dir, '*.txt')):
        with open(path, encoding='utf-8') as f:
            raw = f.read()
        cleaned = clean_text(raw)
        if cleaned:
            cleaned_lines.append(cleaned)
    # Write one cleaned agreement per line
    with open(output_file, 'a', encoding='utf-8') as out:
        out.write('\n'.join(cleaned_lines))

if __name__ == '__main__':
    INPUT_DIR = ['data_sharing_agreement_1_text', 'data_sharing_agreement_2_text']
    # your folder of extracted .txt files
    OUTPUT_FILE = 'agreement_text_row_clean.txt'

    for _input_folder in INPUT_DIR:
        main(_input_folder, OUTPUT_FILE)
