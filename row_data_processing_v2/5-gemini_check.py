import json
import re
import os

# Input and Output filenames
input_filename = './labelled_data/odrl_simodrl_spans_checked_adv_v2.jsonl'
output_filename = './labelled_data/odrl_simodrl_spans_checked_adv_gemini_v3.jsonl'


def fix_record(record):
    text = record.get('text', '')
    entities = record.get('entities', [])

    if not text or not entities:
        return record

    new_entities = []
    # Sort entities to ensure we process them in order (though not strictly necessary for this logic)
    entities.sort(key=lambda x: x['start'])

    for ent in entities:
        label = ent['label']
        start = ent['start']
        end = ent['end']
        span_text = text[start:end]

        # --- CORRECTION 1: Fix Label Typos ---
        if "TULE_TYPE" in label:
            label = label.replace("TULE_TYPE", "RULE_TYPE")

        # --- CORRECTION 2: Expand Truncated ASSIGNEE Spans ---
        if label == "ASSIGNEE":
            # Fix "Public" -> "Public Health England"
            if span_text == "Public" and text[end:].startswith(" Health England"):
                end += len(" Health England")

            # Fix "London" -> "London Borough of [Name]"
            elif span_text == "London" and text[end:].startswith(" Borough of"):
                # Regex to find the full borough name (e.g., "London Borough of Camden")
                # We look for " Borough of " followed by a word starting with a capital letter
                match = re.match(r" Borough of [A-Z][a-z]+", text[end:])
                if match:
                    end += len(match.group(0))

            # Fix "Greater" -> "Greater London Authority"
            elif span_text == "Greater" and text[end:].startswith(" London Authority"):
                end += len(" London Authority")

            # Fix "Information" -> "Information Commissioner's Office"
            elif span_text == "Information" and text[end:].startswith(" Commissioner"):
                # Handle cases where it might just be "Information Commissioner" or the full "Office"
                if text[end:].startswith(" Commissioner's Office"):
                    end += len(" Commissioner's Office")
                elif text[end:].startswith(" Commissioner"):
                    end += len(" Commissioner")

            # Fix "Department" -> "Department of [Name]" (Generic handler)
            elif span_text == "Department" and text[end:].startswith(" of"):
                # Look for "Department of Health and Social Care" specific case or generic
                if text[end:].startswith(" of Health and Social Care"):
                    end += len(" of Health and Social Care")
                else:
                    match = re.match(r" of [A-Z][a-z]+", text[end:])
                    if match:
                        end += len(match.group(0))

        ent['label'] = label
        ent['end'] = end
        new_entities.append(ent)

    record['entities'] = new_entities
    return record


def main():
    if not os.path.exists(input_filename):
        print(f"Error: Could not find '{input_filename}'. Please ensure it is in this directory.")
        return

    print(f"Processing {input_filename}...")
    count = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
                open(output_filename, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        fixed_data = fix_record(data)
                        outfile.write(json.dumps(fixed_data) + '\n')
                        count += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line at line {count + 1}")

        print(f"Success! Processed {count} records.")
        print(f"File saved as: {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()