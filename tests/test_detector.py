import sys
import json
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

from perception.detector import Detector, HybridDetector


def load_clevr_samples(jsonl_path: Path, limit: int = 10, seed: int | None = None):
    samples_by_id = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            img_id = data["image_id"]

            if img_id in samples_by_id:
                continue

            num_objects = len(data["scene_graph"]["objects"])
            samples_by_id[img_id] = {
                "image_id": img_id,
                "image_path": data["image_path"],
                "num_objects": num_objects,
                "objects": data["scene_graph"]["objects"],
            }

    img_ids = list(samples_by_id.keys())

    if not img_ids:
        return []

    if seed is not None:
        random.seed(seed)

    chosen_ids = random.sample(img_ids, k=min(limit, len(img_ids)))
    return [samples_by_id[i] for i in chosen_ids]


def draw_detections(image_path: str, detections: list, gt_objects: list, output_path: Path):
    img = cv2.imread(str(image_path))

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, det in enumerate(detections):
        bbox = det["bbox"]

        x1, y1, x2, y2 = [int(x) for x in bbox]
        conf = det["confidence"]
        class_name = det["class_name"]

        color = colors[i % len(colors)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), img)


def run_single_detector(
    detector_name: str,
    detector,
    samples: list,
    output_root: Path,
    conf_threshold: float = 0.05,
    save_images: bool = True,
    verbose: bool = False,
):
    output_dir = output_root / detector_name
    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for sample in samples:
        img_path = project_root / sample["image_path"]

        if not img_path.exists():
            continue

        gt_objects = sample["objects"]
        gt_count = len(gt_objects)

        start = time.perf_counter()
        detections = detector.detect(str(img_path), conf_threshold=conf_threshold)
        elapsed = time.perf_counter() - start
        det_count = len(detections)

        result = {
            "image_id": sample["image_id"],
            "gt_count": gt_count,
            "detected_count": det_count,
            "gt_objects": gt_objects,
            "detections": detections,
            "missed": gt_count - det_count if det_count < gt_count else 0,
            "extra": det_count - gt_count if det_count > gt_count else 0,
            "time_sec": elapsed,
        }
        results.append(result)

        if save_images:
            output_path = output_dir / f"{sample['image_id']}.png"
            draw_detections(img_path, detections, gt_objects, output_path)

        if verbose:
            gt_desc = [f"{o['shape']}/{o['color']}" for o in gt_objects]
            det_desc = [f"{d['class_name']}({d['confidence']:.2f})" for d in detections]
            print(f"[{detector_name}] {sample['image_id']}: GT={gt_count}, Det={det_count}")
            print(f"  GT: {', '.join(gt_desc)}")
            print(f"  Det: {', '.join(det_desc) if det_desc else 'NONE'}")

    return results


def compute_stats(results: list) -> dict:
    if not results:
        return {
            "total_gt": 0,
            "total_det": 0,
            "total_missed": 0,
            "detection_rate": 0.0,
            "avg_gt": 0.0,
            "avg_det": 0.0,
            "avg_missed": 0.0,
            "avg_time": 0.0,
        }

    total_gt = sum(r["gt_count"] for r in results)
    total_det = sum(r["detected_count"] for r in results)
    total_missed = sum(r["missed"] for r in results)
    n = len(results)

    total_time = sum(r.get("time_sec", 0.0) for r in results)
    avg_time = total_time / n if n > 0 else 0.0

    detection_rate = total_det / total_gt if total_gt > 0 else 0.0
    avg_gt = total_gt / n
    avg_det = total_det / n
    avg_missed = total_missed / n

    return {
        "total_gt": total_gt,
        "total_det": total_det,
        "total_missed": total_missed,
        "detection_rate": detection_rate,
        "avg_gt": avg_gt,
        "avg_det": avg_det,
        "avg_missed": avg_missed,
        "avg_time": avg_time,
    }


def print_stats(detector_name: str, results: list, stats: dict):
    if not results:
        print(f"{detector_name}: no results to display")
        return

    print()
    print(f"{detector_name} detector results")
    print()
    print(f"{'Image':<25} {'GT':<6} {'Det':<6} {'Missed':<8} {'Time [ms]':<10}")
    print()

    for r in results:
        status = "OK" if r["missed"] == 0 else f"-{r['missed']}"
        time_ms = r.get("time_sec", 0.0) * 1000.0
        print(
            f"{r['image_id']:<25} {r['gt_count']:<6} {r['detected_count']:<6} "
            f"{status:<8} {time_ms:>9.1f}"
        )

    print()
    print(
        f"{'TOTAL':<25} {stats['total_gt']:<6} "
        f"{stats['total_det']:<6} {stats['total_missed']:<8}"
    )
    print()
    print(f"Detection rate: {stats['detection_rate']:.1%}")
    print(f"Avg GT per image: {stats['avg_gt']:.1f}")
    print(f"Avg detections per image: {stats['avg_det']:.1f}")
    print(f"Avg missed per image: {stats['avg_missed']:.1f}")
    print(f"Avg time per image: {stats['avg_time']*1000.0:.1f} ms")


def plot_detection_rate(detector_stats: list, output_root: Path):
    names = [name for name, _ in detector_stats]
    rates = [s["detection_rate"] for _, s in detector_stats]

    plt.figure(figsize=(6, 4))
    x = np.arange(len(names))
    plt.bar(x, rates)
    plt.xticks(x, names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Detection rate")
    plt.title("Detection rate comparison")
    for i, v in enumerate(rates):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    out_path = output_root / "detection_rate_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_avg_counts(detector_stats: list, output_root: Path):
    names = [name for name, _ in detector_stats]
    avg_gt = [s["avg_gt"] for _, s in detector_stats]
    avg_det = [s["avg_det"] for _, s in detector_stats]
    avg_missed = [s["avg_missed"] for _, s in detector_stats]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(8, 4))
    plt.bar(x - width, avg_gt, width, label="Avg GT")
    plt.bar(x, avg_det, width, label="Avg detections")
    plt.bar(x + width, avg_missed, width, label="Avg missed")

    plt.xticks(x, names)
    plt.ylabel("Count per image")
    plt.title("Average counts per image")
    plt.legend()
    for i in range(len(names)):
        plt.text(x[i] - width, avg_gt[i] + 0.05, f"{avg_gt[i]:.1f}", ha="center", va="bottom", fontsize=8)
        plt.text(x[i], avg_det[i] + 0.05, f"{avg_det[i]:.1f}", ha="center", va="bottom", fontsize=8)
        plt.text(x[i] + width, avg_missed[i] + 0.05, f"{avg_missed[i]:.1f}", ha="center", va="bottom", fontsize=8)

    out_path = output_root / "avg_counts_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_runtime(detector_stats: list, output_root: Path):
    names = [name for name, _ in detector_stats]
    avg_time_ms = [s["avg_time"] * 1000.0 for _, s in detector_stats]

    plt.figure(figsize=(6, 4))
    x = np.arange(len(names))
    plt.bar(x, avg_time_ms)
    plt.xticks(x, names)
    plt.ylabel("Average inference time per image [ms]")
    plt.title("Runtime comparison")
    for i, v in enumerate(avg_time_ms):
        plt.text(i, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    out_path = output_root / "runtime_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



if __name__ == "__main__":
    num_samples = 10
    conf_threshold = 0.05
    save_images = True
    verbose = False

    clevr_jsonl = project_root / "datasets" / "converted" / "clevr" / "clevr_val.jsonl"
    output_root = project_root / "tests" / "detector_output"

    if save_images:
        output_root.mkdir(parents=True, exist_ok=True)

    if not clevr_jsonl.exists():
        print(f"[ERROR] CLEVR dataset not found at: {clevr_jsonl}")
        sys.exit(1)

    samples = load_clevr_samples(clevr_jsonl, limit=num_samples, seed=42)

    yolo_detector = Detector(model_path="yolov8n.pt", device="cpu")
    hybrid_detector = HybridDetector(
        yolo_model_path="yolov8n.pt",
        sam_model_path="FastSAM-s.pt",
        device="cpu",
    )

    yolo_results = run_single_detector(
        detector_name="yolo",
        detector=yolo_detector,
        samples=samples,
        output_root=output_root,
        conf_threshold=conf_threshold,
        save_images=save_images,
        verbose=verbose,
    )

    hybrid_results = run_single_detector(
        detector_name="hybrid",
        detector=hybrid_detector,
        samples=samples,
        output_root=output_root,
        conf_threshold=conf_threshold,
        save_images=save_images,
        verbose=verbose,
    )

    yolo_stats = compute_stats(yolo_results)
    hybrid_stats = compute_stats(hybrid_results)

    print_stats("YOLO", yolo_results, yolo_stats)
    print_stats("Hybrid", hybrid_results, hybrid_stats)

    detector_stats = [("YOLO", yolo_stats), ("Hybrid", hybrid_stats)]
    plot_detection_rate(detector_stats, output_root)
    plot_avg_counts(detector_stats, output_root)
    plot_runtime(detector_stats, output_root)
