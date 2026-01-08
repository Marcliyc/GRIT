import os
import re
import json
import argparse
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import jsonlines
from shapely.geometry import box
from shapely.ops import unary_union

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_gt(gt_root: str) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """Load ground-truth annotations keyed by a unique name."""
    gt_data_dict: Dict[str, dict] = {}
    gt_data_count: Dict[str, int] = defaultdict(int)
    for fname in os.listdir(gt_root):
        path = os.path.join(gt_root, fname)
        if not fname.endswith("_val.jsonl") or os.path.isdir(path):
            continue
        entries = list(jsonlines.open(path))
        for entry in entries:
            ds = entry.get("dataset", "")
            question_raw = entry.get("question", "")
            question_part = "" if ds in ("mathvista_mini", "mme_position") else question_raw.split("?", 1)[0].strip()
            name = f"{ds}_{os.path.basename(entry.get('image', ''))}_{question_part}_{entry.get('answer', '')}"
            if name in gt_data_dict:
                continue
            gt_data_dict[name] = entry
            gt_data_count[ds] += 1
    return gt_data_dict, gt_data_count


def map_single_gt_set(pred_boxes: List[List[float]], gt_box: List[float], iou_thresholds=np.arange(0.50, 0.96, 0.05)) -> float:
    """mAP for a single GT bbox without scores (set precision)."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    gt_box = np.asarray(gt_box, dtype=np.float64).reshape(4)

    def iou(box):
        x1, y1 = np.maximum(box[:2], gt_box[:2])
        x2, y2 = np.minimum(box[2:], gt_box[2:])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        if inter == 0.0:
            return 0.0
        area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
        area_b = np.maximum(0.0, gt_box[2] - gt_box[0]) * np.maximum(0.0, gt_box[3] - gt_box[1])
        return inter / (area_a + area_b - inter)

    if len(pred_boxes) == 0:
        return 0.0

    aps = []
    for thr in iou_thresholds:
        hits = any(iou(box) >= thr for box in pred_boxes)
        aps.append((1.0 / len(pred_boxes)) if hits else 0.0)
    return float(np.mean(aps))


def iou_of_bbox_groups(bboxes1: List[List[float]], bboxes2: List[List[float]]) -> float:
    polys1 = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in bboxes1]
    polys2 = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in bboxes2]
    union1 = unary_union(polys1)
    union2 = unary_union(polys2)
    inter = union1.intersection(union2)
    uni = union1.union(union2)
    if uni.area == 0:
        return 0.0
    return inter.area / uni.area


def parse_bbox(bbox_str: str) -> List[int]:
    return list(map(int, bbox_str.replace("[", "").replace("]", "").split(",")))


def load_data_from_dir(dirpath: str) -> List[dict]:
    data: List[dict] = []
    for fname in os.listdir(dirpath):
        if fname.endswith('.json'):
            with open(os.path.join(dirpath, fname), 'r') as f:
                data.extend(json.load(f))
    return data


def compute_metrics(data: List[dict], gt_lookup: Dict[str, dict], relative_bbox: bool = False) -> Tuple[Dict[str, float], Dict[str, float]]:
    acc_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    iou_dict: Dict[str, List[float]] = defaultdict(list)

    for entry in data:
        io = entry.get("input_output_conv", {})
        ds = io.get("dataset", "")
        msg = io.get("message", [])
        question_text = msg[0].get("content", [])[-1].get('text', '') if msg else ''
        question_short = question_text.split('Question: ', 1)[-1].split(' First, think between ')[0].split('?', 1)[0].strip()
        name = f"{ds}_{os.path.basename(io.get('image', ''))}_{question_short}_{io.get('gt_answer', '')}"

        reward_name = entry.get('reward_name', [])
        reward_list = json.loads(entry.get('reward_list', '[]')) if 'reward_list' in entry else []
        for r_name, r_val in zip(reward_name, reward_list):
            if 'gpt_score_reward' in r_name or 'single_gt_grounding_answer_reward' in r_name:
                acc_dict[ds][name] = r_val
                if 'ovd' in ds:
                    acc_dict[ds.split('_')[0]][name] = r_val
                break

        gt_matches = re.findall(r'\[\b\d+,\s*\d+,\s*\d+,\s*\d+\b\]', io.get('gt_answer', ''))
        bboxes_gt = [parse_bbox(m) for m in gt_matches]

        response = msg[1].get('content', [])[-1].get('text', '') if len(msg) > 1 else ''
        pred_matches = re.findall(r'\b\d+,\s*\d+,\s*\d+,\s*\d+\b', response)
        bboxes_pred = [parse_bbox(m) for m in pred_matches]

        if bboxes_gt:
            mAP = map_single_gt_set(bboxes_pred, bboxes_gt[0]) if bboxes_pred else 0.0
            acc_dict[ds][name] = mAP
            if 'ovd' in ds:
                acc_dict[ds.split('_')[0]][name] = mAP
            width, height = io.get('width', 1), io.get('height', 1)
        else:
            if name not in gt_lookup:
                continue
            gt_entry = gt_lookup[name]
            bboxes_gt = gt_entry.get('bboxs', [])
            if not bboxes_gt:
                continue
            width, height = gt_entry.get('width', 1), gt_entry.get('height', 1)

        if not bboxes_pred:
            iou_dict[ds].append(0.0)
            if 'ovd' in ds:
                iou_dict[ds.split('_')[0]].append(0.0)
            continue

        if relative_bbox:
            bboxes_pred = [
                (
                    int(x1 / 1000 * width),
                    int(y1 / 1000 * height),
                    int(x2 / 1000 * width),
                    int(y2 / 1000 * height),
                )
                for x1, y1, x2, y2 in bboxes_pred
            ]

        iou = iou_of_bbox_groups(bboxes_pred, bboxes_gt)
        iou_dict[ds].append(iou)
        if 'ovd' in ds:
            iou_dict[ds.split('_')[0]].append(iou)

    avg_acc = {ds: (sum(acc_dict[ds].values()) / len(acc_dict[ds])) for ds in acc_dict if acc_dict[ds]}
    avg_iou = {ds: (sum(iou_dict[ds]) / len(iou_dict[ds])) for ds in iou_dict if iou_dict[ds]}
    return avg_acc, avg_iou


def build_md(table: OrderedDict, title: str, dataset_order: List[str]) -> str:
    extras = sorted({k for metrics in table.values() for k in metrics if k not in dataset_order})
    headers = ["Experiment"] + dataset_order + extras
    lines = [f"## {title}", '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row_key, metrics in table.items():
        cells = [row_key] + [f"{metrics.get(ds, ''):.4f}" if ds in metrics else '' for ds in headers[1:]]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export ACC/IoU metrics from outputs.")
    parser.add_argument("--base-output", default="./output", help="Root directory containing experiment subfolders.")
    parser.add_argument("--gt-path", default="./mm-cot-data", help="Directory with *_val.jsonl ground truth files.")
    parser.add_argument("--out-dir", default="./eval_resuts", help="Where to write markdown/json outputs.")
    parser.add_argument("--dataset-order", nargs="*", default=[
        "vsr", "tallyqa", "gqa", "mathvista_mini",
        "mme_count", "mme_position", "mme_existence", "mme_color",
        "ovd_position"
    ], help="Preferred dataset ordering for the markdown tables.")
    args = parser.parse_args()

    gt_lookup, _ = load_gt(args.gt_path)

    acc_table: OrderedDict[str, Dict[str, float]] = OrderedDict()
    iou_table: OrderedDict[str, Dict[str, float]] = OrderedDict()

    for exp in sorted(os.listdir(args.base_output)):
        exp_path = os.path.join(args.base_output, exp)
        if not os.path.isdir(exp_path):
            continue
        for sub in sorted(os.listdir(exp_path)):
            if not sub.startswith("local_log_step_"):
                continue
            step = sub.split("_")[-1]
            key = f"{exp}_step{step}"
            data_dir = os.path.join(exp_path, sub)
            data = load_data_from_dir(data_dir)

            seen = set()
            deduped = []
            for item in data:
                io = item.get("input_output_conv", {})
                msg = io.get("message", [])
                q_text = msg[0].get("content", [])[-1].get('text', '') if msg else ''
                sig = (io.get("dataset", ""), q_text, io.get("image", ""), io.get("gt_answer", ""))
                if sig in seen:
                    continue
                seen.add(sig)
                deduped.append(item)

            avg_acc, avg_iou = compute_metrics(deduped, gt_lookup, relative_bbox=("internvl" in exp.lower()))
            acc_table[key] = avg_acc
            iou_table[key] = avg_iou

    md_acc = build_md(acc_table, "ACC Results", args.dataset_order)
    md_iou = build_md(iou_table, "IoU Results", args.dataset_order)

    os.makedirs(args.out_dir, exist_ok=True)
    acc_md_path = os.path.join(args.out_dir, "metrics_acc.md")
    iou_md_path = os.path.join(args.out_dir, "metrics_giou.md")
    acc_json_path = os.path.join(args.out_dir, "metrics_acc.json")
    iou_json_path = os.path.join(args.out_dir, "metrics_giou.json")

    with open(acc_md_path, "w", encoding="utf-8") as f:
        f.write(md_acc + "\n")
    with open(iou_md_path, "w", encoding="utf-8") as f:
        f.write(md_iou + "\n")
    with open(acc_json_path, "w", encoding="utf-8") as f:
        json.dump(acc_table, f, indent=2)
    with open(iou_json_path, "w", encoding="utf-8") as f:
        json.dump(iou_table, f, indent=2)

    print(f"Wrote {acc_md_path}\nWrote {iou_md_path}\nWrote {acc_json_path}\nWrote {iou_json_path}")


if __name__ == "__main__":
    main()
