"""
PERSONA-CHAT data preprocessing pipeline for DUET.

Loads the OFFICIAL ParlAI PERSONA-CHAT "both" files (fbdialog format) for the
Original and Revised versions and converts every labelled turn into a
ProcessedRecord, preserving BOTH speakers' profiles on every record:

    profile_self    <- lines prefixed "your persona:"
    profile_partner <- lines prefixed "partner's persona:"

Upstream layout (verified against ParlAI's task code and a byte-level sample of
the released files; the official archive is SHA256-pinned by ParlAI):

    data/raw/personachat/{train,valid,test}_both_original.txt
    data/raw/personachat/{train,valid,test}_both_revised.txt

fbdialog line format (parsed exactly like ParlAI's FbDeprecatedDialogTeacher):

    <conv_id> <text>[<TAB>labels[<TAB>reward[<TAB>candidates]]]

    * conv_id restarts at 1 for every episode -> episode boundary.
    * Literal "\\n" sequences inside a line unescape to newlines.
    * Persona sentences are their own context lines at the start of an episode.

Supported Formats:
  - Official ParlAI fbdialog "both" files (see reader docstring).
  - Standard dictionary records with explicit profile keys.

Dataset Versions:
  - original (both_original)
  - revised (both_revised)

Output Directory Structure:
  data/processed/{dataset_version}/{split}.jsonl
"""

import os
import json
import hashlib
import socket
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from duet.data.schemas import ProcessedRecord, DatasetVersion

PARLAI_TGZ_URL = "http://parl.ai/downloads/personachat/personachat.tgz"
# Checksum pinned by ParlAI in parlai/tasks/personachat/build.py.
PARLAI_TGZ_SHA256 = "507cf8641d333240654798870ea584d854ab5261071c5e3521c20d8fa41d5622"
# Fallback mirror with the individual *_both_*.txt files (parl.ai answers 403
# on some networks; same files, used by scripts/01b_build_dataset_both.sh).
HF_MIRROR_BASE = "https://huggingface.co/datasets/awsaf49/persona-chat/resolve/main"
# Official file names, one tuple of splits per dataset version.
BOTH_FILENAMES = {
    "original": ("train_both_original.txt", "valid_both_original.txt", "test_both_original.txt"),
    "revised": ("train_both_revised.txt", "valid_both_revised.txt", "test_both_revised.txt"),
}


def parse_parlai_episode(
    episode_turns: List[Dict[str, Any]],
    episode_index: int,
    dataset_version: DatasetVersion,
    split: str
) -> List[ProcessedRecord]:
    """
    Parses a single multi-turn ParlAI episode into ProcessedRecord instances.

    Leading context lines carry the personas (official fbdialog "both" files use
    one persona sentence per numbered line):
      - 'your persona:' -> profile_self
      - 'partner's persona:' -> profile_partner
    Both profiles must be complete before the first record is emitted, and a
    record is only created for turns that have a target label.

    Args:
        episode_turns: List of turn dicts for one full episode.
        episode_index: Index of the episode.
        dataset_version: 'original' or 'revised'.
        split: Split name ('train', 'valid', 'test').

    Returns:
        List of ProcessedRecord instances (one per dialogue turn with target response).
    """
    if not episode_turns:
        return []

    profile_self: List[str] = []
    profile_partner: List[str] = []
    history: List[str] = []
    processed_records: List[ProcessedRecord] = []

    for turn_idx, turn in enumerate(episode_turns):
        text_content = turn.get("text", "")
        labels = turn.get("labels") or turn.get("eval_labels") or []
        target = labels[0] if labels and isinstance(labels, list) else turn.get("target", "")

        # Split text into lines to process persona prefixes vs user utterances
        lines = [line.strip() for line in text_content.split("\n") if line.strip()]
        utterance_lines: List[str] = []

        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("your persona:"):
                sent = line[len("your persona:"):].strip()
                if sent and sent not in profile_self:
                    profile_self.append(sent)
            elif line_lower.startswith("partner's persona:") or line_lower.startswith("partners persona:"):
                prefix = "partner's persona:" if line_lower.startswith("partner's persona:") else "partners persona:"
                sent = line[len(prefix):].strip()
                if sent and sent not in profile_partner:
                    profile_partner.append(sent)
            else:
                utterance_lines.append(line)

        # Keep every non-persona line, in order: fbdialog context and unescaped
        # "\n" sequences can put more than one line in a single turn.
        dialogue_utterance = "\n".join(utterance_lines) if utterance_lines else None

        if dialogue_utterance:
            # Current turn user utterance is added to history after creating the record (if target exists)
            current_history = list(history)
            current_history.append(dialogue_utterance)

            if target:
                # Both profiles must be complete before any record is emitted
                # (official files deliver personas as separate context lines).
                if not profile_self or not profile_partner:
                    raise ValueError(
                        f"Missing speaker profile in episode {episode_index}! "
                        f"profile_self={len(profile_self)}, profile_partner={len(profile_partner)}"
                    )
                record_id = f"{dataset_version}_{split}_ep{episode_index:05d}_t{turn_idx:02d}"
                record = ProcessedRecord(
                    id=record_id,
                    history=current_history,
                    profile_self=list(profile_self),
                    profile_partner=list(profile_partner),
                    target=str(target).strip(),
                    persona_labels=[],  # Populated if explicit persona labels are provided
                    dataset_version=dataset_version,
                )
                processed_records.append(record)

            # Update history for next turns
            history.append(dialogue_utterance)
            if target:
                history.append(str(target).strip())

    return processed_records


def parse_raw_personachat_sample(
    raw_sample: Union[Dict[str, Any], List[Dict[str, Any]]],
    sample_index: int,
    dataset_version: DatasetVersion,
    split: str
) -> List[ProcessedRecord]:
    """
    Parses a single sample (either a ParlAI episode list or a raw dictionary) into ProcessedRecords.
    """
    if isinstance(raw_sample, list):
        return parse_parlai_episode(raw_sample, sample_index, dataset_version, split)

    if not isinstance(raw_sample, dict):
        raise TypeError(f"Sample at index {sample_index} must be a dictionary or list of dicts.")

    # ParlAI single turn dictionary check
    if "text" in raw_sample and ("your persona:" in raw_sample["text"].lower() or "partner's persona:" in raw_sample["text"].lower()):
        return parse_parlai_episode([raw_sample], sample_index, dataset_version, split)

    # Standard explicit dictionary format check
    profile_self = (
        raw_sample.get("profile_self")
        or raw_sample.get("personality")
        or raw_sample.get("your_persona")
        or []
    )
    profile_partner = (
        raw_sample.get("profile_partner")
        or raw_sample.get("partner_persona")
        or raw_sample.get("their_persona")
        or []
    )

    if isinstance(profile_self, str):
        profile_self = [profile_self]
    if isinstance(profile_partner, str):
        profile_partner = [profile_partner]

    profile_self = [str(p).strip() for p in profile_self if str(p).strip()]
    profile_partner = [str(p).strip() for p in profile_partner if str(p).strip()]

    if not profile_self or not profile_partner:
        raise ValueError(f"Missing speaker profile in sample {sample_index}! Self: {len(profile_self)}, Partner: {len(profile_partner)}")

    raw_history = raw_sample.get("history") or raw_sample.get("dialogue") or []
    target = raw_sample.get("target") or raw_sample.get("response") or ""
    persona_labels = raw_sample.get("persona_labels") or []

    history_turns = [str(h).strip() for h in raw_history if str(h).strip()]
    labels = [int(lbl) for lbl in persona_labels if isinstance(lbl, (int, str)) and str(lbl).isdigit()]

    record_id = raw_sample.get("id") or f"{dataset_version}_{split}_{sample_index:06d}"

    record = ProcessedRecord(
        id=str(record_id),
        history=history_turns,
        profile_self=profile_self,
        profile_partner=profile_partner,
        target=str(target).strip(),
        persona_labels=labels,
        dataset_version=dataset_version,
    )
    return [record]


def read_parlai_txt_file(file_path: Union[str, Path]) -> List[List[Dict[str, Any]]]:
    """
    Reads an official ParlAI fbdialog file (e.g. train_both_original.txt) into episodes.

    Parsing mirrors ParlAI's FbDeprecatedDialogTeacher.setup_data:

      - Each physical line is "<conv_id> <text>", optionally followed by
        tab-separated columns: labels, reward, label candidates.
      - conv_id restarts at 1 for every episode; an episode boundary is detected
        when conv_id does not increase (ParlAI's "conv_id <= last_conv_id" rule).
      - Literal "\\n" sequences inside a line are unescaped to real newlines.
      - Lines without a labels column (persona sentences, context utterances)
        are kept as turns with empty labels so nothing is silently dropped;
        such turns never become ProcessedRecords downstream.

    Returns:
        List of episodes; each episode is a list of turn dicts
        {"text": str, "labels": List[str]} in file order.
    """
    episodes: List[List[Dict[str, Any]]] = []
    current_episode: List[Dict[str, Any]] = []
    last_conv_id: Optional[int] = None
    skipped_lines = 0

    def _flush_episode() -> None:
        nonlocal current_episode, last_conv_id
        if current_episode:
            episodes.append(current_episode)
            current_episode = []
        last_conv_id = None

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip().replace("\\n", "\n")
            if not line:
                continue

            space_idx = line.find(" ")
            if space_idx <= 0:
                skipped_lines += 1
                continue
            try:
                conv_id = int(line[:space_idx])
            except ValueError:
                skipped_lines += 1
                continue

            if last_conv_id is not None and conv_id <= last_conv_id:
                _flush_episode()
            last_conv_id = conv_id

            # Columns: text <TAB> labels <TAB> reward <TAB> label_candidates.
            columns = [column.strip() for column in line[space_idx + 1:].split("\t")]
            text = columns[0]
            labels: List[str] = []
            if len(columns) > 1 and columns[1]:
                labels = [label.strip() for label in columns[1].split("|") if label.strip()]
            current_episode.append({"text": text, "labels": labels})

    _flush_episode()

    if skipped_lines:
        print(f"[persona_chat] {file_path}: skipped {skipped_lines} malformed line(s)")
    return episodes


def process_dataset_split(
    dataset_split: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    dataset_version: DatasetVersion,
    split_name: str,
    output_dir: Union[str, Path]
) -> str:
    """
    Processes dataset records and outputs a JSONL file.
    """
    output_path = Path(output_dir) / dataset_version / f"{split_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset_split):
            records = parse_raw_personachat_sample(
                raw_sample=sample,
                sample_index=idx,
                dataset_version=dataset_version,
                split=split_name
            )
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                processed_count += 1

    print(f"[{dataset_version.upper()} - {split_name}] Processed {processed_count} records -> {output_path}")
    return str(output_path)


def _sha256_hex(file_path: Path) -> str:
    """Hex SHA256 of a file (used to pin the official PERSONA-CHAT archive)."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract_parlai_data(
    download_dir: Union[str, Path] = "data/raw",
    dataset_version: Optional[DatasetVersion] = None,
) -> Path:
    """
    Materialises the official PERSONA-CHAT "both" files under {download_dir}/personachat.

    Primary source: ParlAI's official archive (URL + SHA256 taken from ParlAI's
    parlai/tasks/personachat/build.py, checksum verified before extraction).
    Fallback: the Hugging Face mirror awsaf49/persona-chat, fetching the
    individual *_both_*.txt files (parl.ai answers 403 on some networks).

    Idempotent: returns immediately when the expected files already exist.
    Args:
        download_dir: Raw-data directory (default data/raw).
        dataset_version: Limit to one version's files ('original' | 'revised');
            None expects both versions.
    Returns:
        Path of the personachat/ directory holding the txt files.
    """
    # Guard against a hanging mirror/CDN: fail fast instead of stalling the pipeline.
    socket.setdefaulttimeout(60)

    download_dir = Path(download_dir)
    target_dir = download_dir / "personachat"
    expected_names = (
        list(BOTH_FILENAMES[dataset_version])
        if dataset_version in ("original", "revised")
        else [name for names in BOTH_FILENAMES.values() for name in names]
    )

    if target_dir.is_dir() and any((target_dir / name).exists() for name in expected_names):
        return target_dir

    download_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "personachat.tgz"

    try:
        if not archive_path.exists():
            print(f"[persona_chat] downloading official archive: {PARLAI_TGZ_URL}")
            urllib.request.urlretrieve(PARLAI_TGZ_URL, archive_path)

        archive_sha = _sha256_hex(archive_path)
        if archive_sha != PARLAI_TGZ_SHA256:
            # Corrupt/partial download artefact: remove it (regenerable cache)
            # and fall through to the mirror instead of trusting bad bytes.
            archive_path.unlink()
            raise ValueError(f"SHA256 mismatch (got {archive_sha})")

        print("[persona_chat] extracting personachat.tgz")
        with tarfile.open(archive_path, "r:gz") as tar:
            try:
                tar.extractall(path=download_dir, filter="data")
            except TypeError:  # Python without PEP 706 tar filters
                tar.extractall(path=download_dir)

        if not any((target_dir / name).exists() for name in expected_names):
            raise FileNotFoundError("archive extracted but *_both_*.txt files not found")
    except Exception as exc:
        print(f"[persona_chat] official archive unavailable ({exc}); falling back to HF mirror")
        for name in expected_names:
            dest = target_dir / name
            if dest.exists():
                continue
            print(f"[persona_chat]   downloading {name}")
            # Write to a temp file first so an interrupted run never leaves a
            # truncated file that the idempotency check would then trust.
            partial = dest.with_name(dest.name + ".part")
            urllib.request.urlretrieve(f"{HF_MIRROR_BASE}/{name}", partial)
            partial.replace(dest)

    return target_dir


def process_parlai_personachat(
    dataset_version: DatasetVersion = "original",
    raw_dir: Union[str, Path] = "data/raw",
    output_dir: Union[str, Path] = "data/processed"
) -> List[str]:
    """
    Processes the official ParlAI PERSONA-CHAT both-format files for one dataset version.

    Reads {raw_dir}/personachat/{split}_both_{dataset_version}.txt for each split
    ("train", "valid", "test"), writes one JSONL per existing split to
    {output_dir}/{dataset_version}/{split}.jsonl, and logs per-split episode and
    record counts. Missing splits are reported and skipped; a version with no
    files at all raises FileNotFoundError instead of silently producing nothing.
    Downloads the data first when the raw directory is absent.

    Returns:
        List of written JSONL file paths.
    """
    if dataset_version not in ("original", "revised"):
        raise ValueError("dataset_version must be 'original' or 'revised'.")

    raw_path = Path(raw_dir) / "personachat"
    if not raw_path.exists():
        raw_path = download_and_extract_parlai_data(raw_dir, dataset_version=dataset_version)

    expected_names = BOTH_FILENAMES[dataset_version]
    if not any((raw_path / name).exists() for name in expected_names):
        raise FileNotFoundError(
            f"No {dataset_version} both-format files in {raw_path}. "
            f"Expected one of: {', '.join(expected_names)}"
        )

    written: List[str] = []
    for split_name in ("train", "valid", "test"):
        file_path = raw_path / f"{split_name}_both_{dataset_version}.txt"
        if not file_path.exists():
            print(f"[persona_chat] {dataset_version}/{split_name}: missing {file_path.name}, skipping")
            continue
        episodes = read_parlai_txt_file(file_path)
        print(f"[persona_chat] {dataset_version}/{split_name}: {len(episodes)} episodes from {file_path.name}")
        output_path = process_dataset_split(
            dataset_split=episodes,
            dataset_version=dataset_version,
            split_name=split_name,
            output_dir=output_dir
        )
        written.append(output_path)

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process ParlAI PERSONA-CHAT dataset for DUET.")
    parser.add_argument(
        "--version", type=str, choices=["original", "revised", "both"], default="original",
        help="Dataset version to process; 'both' runs Original then Revised."
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")

    args = parser.parse_args()

    versions = ["original", "revised"] if args.version == "both" else [args.version]
    for version in versions:
        process_parlai_personachat(
            dataset_version=version,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir
        )
