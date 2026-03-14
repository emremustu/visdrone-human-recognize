"""
evaluation/metrics.py — Nesne Tespiti Değerlendirme Metrikleri

mAP, Precision, Recall ve F1 hesaplama.
COCO standardındaki IoU eşikleri kullanılır (0.50 ve 0.50:0.95).

Terminoloji:
  - TP (True Positive):  Gerçek nesneyle eşleşen doğru tespit
  - FP (False Positive): Gerçek nesne olmadan yapılan hatalı tespit
  - FN (False Negative): Tespit edilemeyen gerçek nesne
  - IoU:  Intersection over Union = kesişim / birleşim

Kullanım:
    from evaluation.metrics import DetectionMetrics

    metrics = DetectionMetrics(iou_threshold=0.50)
    result  = metrics.evaluate(predictions, ground_truths)
    print(result)

Demo kullanım:
    python evaluation/metrics.py --demo
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import EVAL_IOU_THRESHOLDS


# ─── IoU ─────────────────────────────────────────────────────────────────────

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    İki bounding box arasındaki IoU'yu hesaplar.

    Args:
        box_a, box_b: [x1, y1, x2, y2] piksel formatı

    Returns:
        IoU değeri [0.0 – 1.0]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1);  inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2);  inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(preds: List[List[float]], gts: List[List[float]]) -> np.ndarray:
    """
    N preds × M gts IoU matrisi hesaplar.

    Returns:
        shape=(N, M) float64 array
    """
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    mat = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            mat[i, j] = compute_iou(p, g)
    return mat


# ─── Precision-Recall ─────────────────────────────────────────────────────────

def match_predictions(
    pred_boxes  : List[List[float]],
    pred_confs  : List[float],
    gt_boxes    : List[List[float]],
    iou_thresh  : float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Güven skoru sırasına göre predictionları GT'lere eşleştirir.
    Greedy matching: en yüksek IoU'lu GT'yi alır (her GT yalnızca bir kez).

    Args:
        pred_boxes:  [[x1,y1,x2,y2], ...]
        pred_confs:  Her prediction'ın güven skoru
        gt_boxes:    [[x1,y1,x2,y2], ...]
        iou_thresh:  Eşleşme için minimum IoU

    Returns:
        (tp_array, fp_array)  — her eleman 0 veya 1, confidence sırasına göre
    """
    n_pred = len(pred_boxes)
    n_gt   = len(gt_boxes)

    if n_pred == 0:
        return np.array([]), np.array([])

    # Güven skoruna göre azalan sırala
    order     = np.argsort(pred_confs)[::-1]
    tp        = np.zeros(n_pred)
    fp        = np.zeros(n_pred)
    matched   = set()   # Eşleşen GT indeksleri

    if n_gt > 0:
        iou_mat = iou_matrix(pred_boxes, gt_boxes)

    for rank, idx in enumerate(order):
        if n_gt == 0:
            fp[rank] = 1
            continue

        iou_row   = iou_mat[idx]
        best_gt   = int(np.argmax(iou_row))
        best_iou  = iou_row[best_gt]

        if best_iou >= iou_thresh and best_gt not in matched:
            tp[rank]   = 1
            matched.add(best_gt)
        else:
            fp[rank] = 1

    return tp, fp


def precision_recall_curve(
    tp: np.ndarray,
    fp: np.ndarray,
    n_gt: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Birikimli TP/FP'den precision-recall eğrisi hesaplar.

    Returns:
        (precision_array, recall_array)
    """
    if n_gt == 0:
        return np.array([1.0]), np.array([0.0])

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    recall    = cum_tp / (n_gt + 1e-9)

    return precision, recall


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    11-point interpolation yöntemiyle Average Precision hesaplar.
    (Pascal VOC standardı, COCO ile yakın sonuç verir)

    Returns:
        AP float değeri [0.0 – 1.0]
    """
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precision[recall >= thr]
        ap += prec_at_thr.max() if prec_at_thr.size > 0 else 0.0
    return ap / 11.0


# ─── Ana Metrik Sınıfı ────────────────────────────────────────────────────────

class DetectionMetrics:
    """
    Nesne tespiti değerlendirme metrikleri.

    Args:
        iou_threshold: TP kabul eşiği (varsayılan 0.50)
    """

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    def evaluate_image(
        self,
        pred_boxes : List[List[float]],
        pred_confs : List[float],
        gt_boxes   : List[List[float]],
    ) -> Dict:
        """
        Tek görüntü için TP/FP/FN sayar.

        Returns:
            {"tp", "fp", "fn", "precision", "recall", "f1"}
        """
        tp_arr, fp_arr = match_predictions(
            pred_boxes, pred_confs, gt_boxes, self.iou_threshold
        )
        tp = int(tp_arr.sum())
        fp = int(fp_arr.sum())
        fn = max(0, len(gt_boxes) - tp)

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        return {
            "tp"       : tp,
            "fp"       : fp,
            "fn"       : fn,
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "f1"       : round(f1, 4),
        }

    def evaluate_dataset(
        self,
        all_pred_boxes  : List[List[List[float]]],
        all_pred_confs  : List[List[float]],
        all_gt_boxes    : List[List[List[float]]],
    ) -> Dict:
        """
        Veri seti genelinde mAP, precision, recall hesaplar.

        Args:
            all_pred_boxes: Her görüntü için prediction bbox listesi
            all_pred_confs: Her görüntü için confidence listesi
            all_gt_boxes:   Her görüntü için GT bbox listesi

        Returns:
            {
              "mAP"        : float,
              "AP"         : float,    (@ iou_threshold)
              "precision"  : float,
              "recall"     : float,
              "f1"         : float,
              "n_images"   : int,
              "n_gt"       : int,
              "n_pred"     : int,
              "tp"         : int,
              "fp"         : int,
              "fn"         : int,
            }
        """
        all_tp     : List[float] = []
        all_fp     : List[float] = []
        all_confs  : List[float] = []
        total_gt   = 0
        total_pred = 0

        for preds, confs, gts in zip(all_pred_boxes, all_pred_confs, all_gt_boxes):
            tp_arr, fp_arr = match_predictions(preds, confs, gts, self.iou_threshold)
            all_tp.extend(tp_arr.tolist())
            all_fp.extend(fp_arr.tolist())
            all_confs.extend(confs)
            total_gt   += len(gts)
            total_pred += len(preds)

        if not all_confs:
            return {"mAP": 0, "AP": 0, "precision": 0, "recall": 0,
                    "f1": 0, "n_images": len(all_gt_boxes), "n_gt": total_gt,
                    "n_pred": 0, "tp": 0, "fp": 0, "fn": total_gt}

        # Tüm datasetteki conf sırasına yeniden sırala
        order   = np.argsort(all_confs)[::-1]
        tp_arr  = np.array(all_tp)[order]
        fp_arr  = np.array(all_fp)[order]

        precision, recall = precision_recall_curve(tp_arr, fp_arr, total_gt)
        ap     = compute_ap(precision, recall)

        tp_tot = int(tp_arr.sum())
        fp_tot = int(fp_arr.sum())
        fn_tot = total_gt - tp_tot

        prec = tp_tot / (tp_tot + fp_tot + 1e-9)
        rec  = tp_tot / (total_gt + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        return {
            "mAP"      : round(ap, 4),   # Dataset genelinde tek eşik AP
            "AP"       : round(ap, 4),
            "precision": round(prec, 4),
            "recall"   : round(rec,  4),
            "f1"       : round(f1,   4),
            "n_images" : len(all_gt_boxes),
            "n_gt"     : total_gt,
            "n_pred"   : total_pred,
            "tp"       : tp_tot,
            "fp"       : fp_tot,
            "fn"       : fn_tot,
        }

    def pr_curve(
        self,
        all_pred_boxes  : List[List[List[float]]],
        all_pred_confs  : List[List[float]],
        all_gt_boxes    : List[List[List[float]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precision-Recall eğrisi noktalarını döner (görselleştirme için)."""
        all_tp, all_fp, all_confs = [], [], []
        total_gt = 0

        for preds, confs, gts in zip(all_pred_boxes, all_pred_confs, all_gt_boxes):
            tp_arr, fp_arr = match_predictions(preds, confs, gts, self.iou_threshold)
            all_tp.extend(tp_arr.tolist())
            all_fp.extend(fp_arr.tolist())
            all_confs.extend(confs)
            total_gt += len(gts)

        if not all_confs:
            return np.array([1.0, 0.0]), np.array([0.0, 1.0])

        order = np.argsort(all_confs)[::-1]
        return precision_recall_curve(
            np.array(all_tp)[order], np.array(all_fp)[order], total_gt
        )


# ─── Demo ─────────────────────────────────────────────────────────────────────

def _demo():
    import matplotlib.pyplot as plt

    print("[Demo] Sentetik prediction / ground-truth verisi oluşturuluyor...")
    rng = np.random.default_rng(1)

    N_IMGS = 50
    all_pred_boxes_50, all_pred_confs, all_gt_boxes = [], [], []
    all_pred_boxes_75 = []

    for _ in range(N_IMGS):
        n_gt   = rng.integers(2, 20)
        n_pred = rng.integers(int(n_gt * 0.6), int(n_gt * 1.4))

        # GT
        gts = []
        for _ in range(n_gt):
            x1, y1 = rng.integers(0, 1200), rng.integers(0, 680)
            w,  h  = rng.integers(10, 40),  rng.integers(12, 50)
            gts.append([float(x1), float(y1), float(x1+w), float(y1+h)])

        # Predictions: bazıları GT'ye yakın, bazıları uzak
        preds, confs = [], []
        for i in range(n_pred):
            if i < n_gt and rng.random() > 0.25:  # TP adayı
                g = gts[i % len(gts)]
                jitter = rng.integers(-8, 8, size=4)
                preds.append([max(0, g[0]+jitter[0]), max(0, g[1]+jitter[1]),
                               g[2]+jitter[2], g[3]+jitter[3]])
            else:  # FP adayı
                x1, y1 = rng.integers(0, 1200), rng.integers(0, 680)
                w,  h  = rng.integers(10, 40),  rng.integers(12, 50)
                preds.append([float(x1), float(y1), float(x1+w), float(y1+h)])
            confs.append(float(rng.uniform(0.2, 0.99)))

        all_gt_boxes.append(gts)
        all_pred_boxes_50.append(preds)
        all_pred_boxes_75.append(preds)
        all_pred_confs.append(confs)

    for thresh, preds in [(0.50, all_pred_boxes_50), (0.75, all_pred_boxes_75)]:
        m = DetectionMetrics(iou_threshold=thresh)
        r = m.evaluate_dataset(preds, all_pred_confs, all_gt_boxes)
        print(f"\n── IoU@{thresh} ──────────────────────")
        print(f"   mAP       : {r['mAP']:.4f}")
        print(f"   Precision : {r['precision']:.4f}")
        print(f"   Recall    : {r['recall']:.4f}")
        print(f"   F1        : {r['f1']:.4f}")
        print(f"   TP/FP/FN  : {r['tp']}/{r['fp']}/{r['fn']}")

    # PR Curve görselleştirme
    m  = DetectionMetrics(iou_threshold=0.50)
    pr, rec = m.pr_curve(all_pred_boxes_50, all_pred_confs, all_gt_boxes)
    ap = compute_ap(pr, rec)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, pr, color="#2196F3", linewidth=2, label=f"AP@0.50 = {ap:.3f}")
    ax.fill_between(rec, pr, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Recall");    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1]);        ax.set_ylim([0, 1.05])
    ax.set_title("Precision-Recall Eğrisi (Demo)")
    ax.legend(fontsize=10);     ax.grid(alpha=0.3)
    plt.tight_layout()
    out = Path("reports/demo_pr_curve.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"\n[Demo] PR eğrisi kaydedildi → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo:
        _demo()
