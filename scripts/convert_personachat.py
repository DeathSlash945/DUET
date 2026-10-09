
#!/usr/bin/env python3

import json
import re
from pathlib import Path
from collections import Counter
import time

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "parlai_data" / "Persona-Chat" / "personachat"
OUTPUT = ROOT / "data"

SPLITS = {
    "train": SOURCE / "train_both_original.txt",
    "valid": SOURCE / "valid_both_original.txt",
    "test": SOURCE / "test_both_original.txt",
}

# Words too common to be useful for persona matching.
STOPWORDS = {
    "i", "me", "my", "mine", "we", "our", "you", "your",
    "he", "she", "they", "it", "a", "an", "the", "is",
    "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "and",
    "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "as", "this", "that", "there", "here", "what",
    "who", "how", "when", "where", "which", "like", "just",
    "really", "very", "also", "not", "don't", "dont",
    "can", "could", "would", "will", "about", "if",
}


def strip_turn_number(text):
    """Remove a leading turn number, e.g. '2 hello' -> 'hello'."""
    return re.sub(r"^\s*\d+\s+", "", text).strip()




def parse_episodes(path):
    """Parse PersonaChat episodes using numbered lines."""
    episodes = []
    current = []
    total_bytes = path.stat().st_size
    bytes_read = 0
    last_percent = -1

    with path.open("rb") as f:
        for raw in f:
            bytes_read += len(raw)
            line = raw.decode("utf-8").rstrip("\r\n")

            # PersonaChat episodes restart at turn number 1.
            match = re.match(r"^\s*(\d+)\s+", line)

            if match and int(match.group(1)) == 1 and current:
                episodes.append(current)
                current = []

            if line.strip():
                current.append(line)

            percent = int(100 * bytes_read / max(total_bytes, 1))

            if percent != last_percent and (
                percent % 2 == 0 or percent == 100
            ):
                print(
                    f"\rReading {path.name}: {percent:3d}% | "
                    f"Episodes: {len(episodes)} | "
                    f"{bytes_read / (1024 * 1024):.1f}/"
                    f"{total_bytes / (1024 * 1024):.1f} MB",
                    end="",
                    flush=True,
                )
                last_percent = percent

    if current:
        episodes.append(current)

    print(f"\nFinished reading {path.name}: {len(episodes)} episodes")
    return episodes



def parse_episode(lines):
    """
    Convert one PersonaChat episode into PAL examples.

    Each dialogue record contains:
        field 0: current input utterance
        field 1: target response
        field 2: unused
        field 3: distractor candidates

    PAL expects each example as:
        [persona_list, history, response, persona_label]

    History contains alternating utterances, ending with the
    current input utterance. The target response is NOT in history.
    """
    personas = []
    dialogue_records = []

    for line in lines:
        fields = line.rstrip("\r\n").split("\t")
        first = fields[0].strip() if fields else ""

        # Remove the record number.
        first = re.sub(r"^\s*\d+\s+", "", first).strip()
        lower = first.lower()

        if lower.startswith("your persona:"):
            statement = first[len("your persona:"):].strip()
            if statement:
                personas.append(statement)
            continue

        if lower.startswith("partner's persona:"):
            # PAL conditions on the target speaker's persona.
            continue

        # Dialogue records need both an input and target response.
        if len(fields) < 2:
            continue

        context = re.sub(
            r"^\s*\d+\s+", "", fields[0]
        ).strip()
        response = fields[1].strip()

        if context and response:
            dialogue_records.append((context, response))

    if not personas:
        return []

    examples = []
    history = []

    for context, response in dialogue_records:
        # Add the current input utterance to the cumulative history.
        history.append(context)

        # Weak supervision: this is a heuristic, not a gold label.
        label = assign_persona_label(personas, response)

        examples.append([
            personas.copy(),
            history.copy(),
            response,
            label,
        ])

        # The target becomes part of the history for the NEXT example.
        history.append(response)

    return examples


def content_words(text):
    """Basic lexical normalization for heuristic persona matching."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def assign_persona_label(personas, response):
    """
    Return [best_persona_index], or [-1] if no lexical overlap exists.

    This is a weak heuristic, not a human-annotated relevance label.
    """
    response_words = content_words(response)

    if not response_words:
        return [-1]

    scores = [
        len(response_words & content_words(persona))
        for persona in personas
    ]

    if not scores or max(scores) == 0:
        return [-1]

    return [scores.index(max(scores))]



def convert_split(path):
    examples = []
    skipped = 0

    episodes = parse_episodes(path)
    total = len(episodes)

    print(f"\nProcessing {path.name}: {total} episodes")

    for i, lines in enumerate(episodes, start=1):
        episode_examples = parse_episode(lines)

        if episode_examples:
            examples.extend(episode_examples)
        else:
            skipped += 1

        if i % 100 == 0 or i == total:
            print(
                f"\rProcessing {path.name}: "
                f"{i}/{total} episodes "
                f"({100 * i / max(total, 1):.1f}%) | "
                f"Examples: {len(examples)} | "
                f"Skipped: {skipped}",
                end="",
                flush=True,
            )

    print()
    print(
        f"{path.name}: episodes={total}, "
        f"examples={len(examples)}, skipped={skipped}"
    )

    return examples


def write_json(path, examples):
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def validate_file(path):
    with path.open(encoding="utf-8") as f:
        examples = json.load(f)

    assert isinstance(examples, list), f"{path}: expected a JSON list"

    for i, example in enumerate(examples):
        assert len(example) == 4, f"{path}: example {i} has wrong length"

        personas, history, response, label = example

        assert isinstance(personas, list) and personas
        assert isinstance(history, list)
        assert isinstance(response, str) and response.strip()
        assert isinstance(label, list) and len(label) == 1

        idx = label[0]
        assert isinstance(idx, int)
        assert idx == -1 or 0 <= idx < len(personas)

    return len(examples)


def main():
    for name, path in SPLITS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    train = convert_split(SPLITS["train"])
    valid = convert_split(SPLITS["valid"])
    test = convert_split(SPLITS["test"])

    if not train or not valid or not test:
        raise RuntimeError(
            "At least one split is empty. Inspect the raw data format "
            "before using the generated files."
        )

    # The loader uses these exact filenames for English data.
    outputs = {
        "train_origin.txt": train,
        "valid_origin.txt": valid,
        "valid#1.txt": valid[:len(valid) // 2],
        "valid#2.txt": valid[len(valid) // 2:],
        "test.txt": test,
    }

    for filename, examples in outputs.items():
        write_json(OUTPUT / filename, examples)

    print("\nValidation:")
    for filename in outputs:
        path = OUTPUT / filename
        count = validate_file(path)
        print(f"  OK: {path.relative_to(ROOT)} ({count} examples)")

    print("\nConversion complete.")


if __name__ == "__main__":
    main()