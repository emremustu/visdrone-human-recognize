"""
evaluation/evaluate.py — Model Değerlendirme Runner

Eğitilmiş YOLOv8 modelini VisDrone test seti üzerinde değerlendirir.
Ground truth YOLO label dosyaları ile prediction'ları karşılaştırarak
mAP@0.5 ve mAP@0.75 hesaplar.

Kullanım:
    python evaluation/evaluate.py \\
        --model runs/train/visdrone_human/weights/best.pt \\
        --data  data/ \\
        --split test \\
        --sahi
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from inference.detector  import HumanDetector
from evaluation.metrics  import DetectionMetrics, compute_ap
from utils.config        import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, EVAL_IOU_THRESHOLDS


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> List[List[float]]:
    """
    YOLO formatındaki label dosyasını piksel bbox listesine dönüştürür.

    Returns:
        [[x1, y1, x2, y2], ...]
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def run_evaluation(
    model_path : str,
    data_dir   : Path,
    split      : str,
    use_sahi   : bool = False,
    conf       : float = CONFIDENCE_THRESHOLD,
    iou        : float = IOU_THRESHOLD,
    device     : str = "cpu",
    max_images : int = -1,
    save_json  : bool = False,
    out_dir    : Path = Path("reports"),
) -> Dict:
    """
    Ana değerlendirme fonksiyonu.

    Returns:
        {
          "results_by_iou": {0.5: {...}, 0.75: {...}},
          "per_image"     : List[Dict],
        }
    """
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split

    img_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )
    if max_images > 0:
        img_files = img_files[:max_images]

    if not img_files:
        raise FileNotFoundError(f"Görüntü bulunamadı: {img_dir}")

    detector = HumanDetector(model_path=model_path, conf=conf, iou=iou, device=device)

    all_pred_boxes  : List[List[List[float]]] = []
    all_pred_confs  : List[List[float]]       = []
    all_gt_boxes    : List[List[List[float]]] = []
    per_image       : List[Dict]              = []

    print(f"\n{'='*60}")
    print(f"  Değerlendirme: {split.upper()} | {len(img_files)} görüntü")
    print(f"  Model: {model_path} | SAHI: {use_sahi}")
    print(f"{'='*60}\n")

    for img_path in tqdm(img_files, desc="Inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Inference
        dets = detector.detect_sahi(img) if use_sahi else detector.detect(img)

        pred_boxes = [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets]
        pred_confs = [d["conf"] for d in dets]

        # Ground truth
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(lbl_path, img_w, img_h)

        all_pred_boxes.append(pred_boxes)
        all_pred_confs.append(pred_confs)
        all_gt_boxes.append(gt_boxes)

        per_image.append({
            "filename"  : img_path.name,
            "n_pred"    : len(pred_boxes),
            "n_gt"      : len(gt_boxes),
        })

    # Metrik hesaplama — her IoU eşiği için
    results_by_iou = {}
    for thresh in EVAL_IOU_THRESHOLDS:
        m = DetectionMetrics(iou_threshold=thresh)
        r = m.evaluate_dataset(all_pred_boxes, all_pred_confs, all_gt_boxes)
        results_by_iou[thresh] = r

    # mAP@0.5:0.95 (COCO tarzı ortalama)
    iou_range = np.arange(0.50, 1.00, 0.05).tolist()
    aps = []
    for thresh in iou_range:
        m  = DetectionMetrics(iou_threshold=thresh)
        pr = m.pr_curve(all_pred_boxes, all_pred_confs, all_gt_boxes)
        aps.append(compute_ap(pr[0], pr[1]))
    map_coco = float(np.mean(aps))

    # Özet yazdır
    print(f"\n{'─'*55}")
    print(f"  Değerlendirme Sonuçları ({split.upper()})")
    print(f"{'─'*55}")
    for thresh, r in results_by_iou.items():
        print(f"\n  IoU@{thresh:.2f}:")
        print(f"    mAP       : {r['mAP']:.4f}")
        print(f"    Precision : {r['precision']:.4f}")
        print(f"    Recall    : {r['recall']:.4f}")
        print(f"    F1        : {r['f1']:.4f}")
        print(f"    TP/FP/FN  : {r['tp']}/{r['fp']}/{r['fn']}")
    print(f"\n  mAP@0.5:0.95 (COCO) : {map_coco:.4f}")
    print(f"  Toplam görüntü       : {len(img_files)}")
    print(f"  Toplam GT bbox       : {results_by_iou[0.50]['n_gt']}")
    print(f"{'─'*55}")

    # Görselleştirme: PR eğrisi
    try:
        import matplotlib.pyplot as plt
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors     = ["#2196F3", "#4CAF50"]
        for (thresh, r), color in zip(results_by_iou.items(), colors):
            m   = DetectionMetrics(iou_threshold=thresh)
            pr, rec = m.pr_curve(all_pred_boxes, all_pred_confs, all_gt_boxes)
            ap  = compute_ap(pr, rec)
            ax.plot(rec, pr, color=color, linewidth=2,
                    label=f"AP@{thresh:.2f} = {ap:.3f}")
            ax.fill_between(rec, pr, alpha=0.1, color=color)

        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim([0, 1]);    ax.set_ylim([0, 1.05])
        ax.set_title(f"Precision-Recall — {split.upper()} (mAP@0.5:0.95={map_coco:.3f})")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        plt.tight_layout()

        pr_out = out_dir / f"eval_pr_{split}.png"
        plt.savefig(pr_out, dpi=130)
        plt.close()
        print(f"[Eval] PR eğrisi kaydedildi → {pr_out}")
    except Exception as e:
        print(f"[UYARI] Grafik üretilemedi: {e}")

    output = {
        "results_by_iou": {str(k): v for k, v in results_by_iou.items()},
        "mAP_coco"      : map_coco,
        "split"         : split,
        "per_image"     : per_image,
    }

    if save_json:
        json_out = out_dir / f"eval_{split}.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[Eval] JSON kaydedildi → {json_out}")

    return output


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 değerlendirme runner")
    p.add_argument("--model",  required=True, help="YOLOv8 ağırlık dosyası (.pt)")
    p.add_argument("--data",   default="data", help="YOLO dataset dizini")
    p.add_argument("--split",  default="test", choices=["train", "val", "test"])
    p.add_argument("--sahi",   action="store_true", help="SAHI kullan")
    p.add_argument("--conf",   type=float,default=CONFIDENCE_THRESHOLD)
    p.add_argument("--iou",    type=float, default=IOU_THRESHOLD)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max",    type=int,   default=-1, help="Maksimum görüntü sayısı")
    p.add_argument("--json",   action="store_true")
    p.add_argument("--out",    default="reports")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_path = args.model,
        data_dir   = Path(args.data),
        split      = args.split,
        use_sahi   = args.sahi,
        conf       = args.conf,
        iou        = args.iou,
        device     = args.device,
        max_images = args.max,
        save_json  = args.json,
        out_dir    = Path(args.out),
    )
