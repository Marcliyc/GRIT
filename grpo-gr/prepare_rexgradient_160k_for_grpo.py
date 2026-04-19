#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import jsonlines


QUESTION_FALLBACK = (
    "Review this chest X-ray and provide your diagnostic conclusion. "
    "Return the final diagnosis after <answer>."
)


def _pick_first(row: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _load_bbox_map(path: Optional[Path]) -> Dict[str, List[List[float]]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, List[List[float]]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key)] = value
    return out


def _resolve_image_path(raw_image: str, image_root: Optional[Path]) -> str:
    if image_root is None:
        return raw_image
    candidate = image_root / raw_image
    if candidate.exists():
        return raw_image
    # Keep original relative path even when local existence check fails.
    return raw_image


def convert_split(
    metadata_csv: Path,
    output_jsonl: Path,
    split: str,
    image_root: Optional[Path],
    bbox_map: Dict[str, List[List[float]]],
    default_question: str,
) -> int:
    image_candidates = ("image", "image_path", "path", "file_path", "jpg_path", "png_path", "dicom_path")
    question_candidates = ("question", "prompt", "instruction", "query")
    answer_candidates = ("answer", "report", "impression", "findings", "response", "target")
    bbox_key_candidates = ("image", "image_path", "path", "dicom_id", "image_id", "id")

    count = 0
    with metadata_csv.open("r", encoding="utf-8") as f, jsonlines.open(output_jsonl, mode="w") as writer:
        reader = csv.DictReader(f)
        for row in reader:
            image = _pick_first(row, image_candidates)
            answer = _pick_first(row, answer_candidates)
            if image is None or answer is None:
                continue

            question = _pick_first(row, question_candidates) or default_question
            bbox_key = _pick_first(row, bbox_key_candidates)
            bboxs = bbox_map.get(bbox_key, []) if bbox_key else []

            writer.write(
                {
                    "question": question,
                    "answer": answer,
                    "image": _resolve_image_path(image, image_root),
                    "bboxs": bboxs,
                    "dataset": "rexgradient-160k",
                    "split": split,
                }
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ReXGradient-160K metadata CSV files into GRPO-GR-Med jsonl format."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Path to ReXGradient-160K root folder.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for generated jsonl files.")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help="Optional metadata folder. Defaults to <data-root>/metadata.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional image root folder used with --train-image-folder-path/--eval-image-folder-path.",
    )
    parser.add_argument(
        "--bbox-json",
        type=Path,
        default=None,
        help="Optional bbox json file (e.g. interstitial_pattern_bbox.json).",
    )
    parser.add_argument("--default-question", type=str, default=QUESTION_FALLBACK)
    args = parser.parse_args()

    metadata_dir = args.metadata_dir or (args.data_root / "metadata")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_json = args.bbox_json or (metadata_dir / "interstitial_pattern_bbox.json")
    bbox_map = _load_bbox_map(bbox_json)

    splits = {
        "train": metadata_dir / "train_metadata.csv",
        "valid": metadata_dir / "valid_metadata.csv",
        "test": metadata_dir / "test_metadata.csv",
    }

    for split, metadata_csv in splits.items():
        if not metadata_csv.exists():
            print(f"Skipping {split}: {metadata_csv} not found.")
            continue
        output_jsonl = output_dir / f"{split}.jsonl"
        converted = convert_split(
            metadata_csv=metadata_csv,
            output_jsonl=output_jsonl,
            split=split,
            image_root=args.image_root,
            bbox_map=bbox_map,
            default_question=args.default_question,
        )
        print(f"Wrote {converted} rows to {output_jsonl}.")


if __name__ == "__main__":
    main()
