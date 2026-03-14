"""
analysis/counter.py — Bounding Box'tan İnsan Sayımı

Bu modül tespit edilen bounding box'lardan:
- Toplam kişi sayısı
- Bölge (ROI) bazlı sayım
- Zaman serisi / frame bazlı sayım istatistikleri
sağlar.

Demo kullanım:
    python analysis/counter.py --demo
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import CONFIDENCE_THRESHOLD


class PersonCounter:
    """
    Tespit edilen kişi bounding box'larını sayar ve analiz eder.

    Args:
        conf_threshold: Kabul eşiği (bu değerin altındaki tespitler sayılmaz)
    """

    def __init__(self, conf_threshold: float = CONFIDENCE_THRESHOLD):
        self.conf_threshold = conf_threshold

    def filter(self, detections: List[Dict]) -> List[Dict]:
        """Güven skoru eşiğinin altındaki tespitleri filtreler."""
        return [d for d in detections if d.get("conf", 1.0) >= self.conf_threshold]

    def count(self, detections: List[Dict]) -> int:
        """
        Toplam kişi sayısını döner.

        Args:
            detections: HumanDetector.detect() çıktısı

        Returns:
            Güven eşiğini geçen tespit sayısı
        """
        return len(self.filter(detections))

    def count_in_roi(
        self,
        detections: List[Dict],
        roi: Tuple[int, int, int, int],
    ) -> int:
        """
        Belirtilen dikdörtgen bölge (ROI) içindeki kişi sayısını döner.

        Kriter: bbox merkezinin ROI içinde olması.

        Args:
            detections: Tespit listesi
            roi:        (x1, y1, x2, y2) piksel koordinatları

        Returns:
            ROI içindeki tespit sayısı
        """
        rx1, ry1, rx2, ry2 = roi
        count = 0
        for det in self.filter(detections):
            cx = det.get("cx", (det["x1"] + det["x2"]) / 2)
            cy = det.get("cy", (det["y1"] + det["y2"]) / 2)
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                count += 1
        return count

    def count_in_grid(
        self,
        detections: List[Dict],
        img_w: int,
        img_h: int,
        rows: int = 4,
        cols: int = 4,
    ) -> np.ndarray:
        """
        Görüntüyü eşit hücrelere böler ve her hücredeki kişi sayısını döner.

        Args:
            detections: Tespit listesi
            img_w, img_h: Görüntü boyutları
            rows, cols:   Izgara boyutları

        Returns:
            (rows x cols) boyutunda int32 array
        """
        grid   = np.zeros((rows, cols), dtype=np.int32)
        cell_w = img_w / cols
        cell_h = img_h / rows

        for det in self.filter(detections):
            cx = det.get("cx", (det["x1"] + det["x2"]) / 2)
            cy = det.get("cy", (det["y1"] + det["y2"]) / 2)
            col = min(int(cx // cell_w), cols - 1)
            row = min(int(cy // cell_h), rows - 1)
            grid[row, col] += 1

        return grid

    def density_stats(self, detections: List[Dict]) -> Dict:
        """
        Tespit listesine ait temel yoğunluk istatistikleri.

        Returns:
            {"count", "avg_bbox_area", "min_conf", "max_conf", "avg_conf"}
        """
        filtered = self.filter(detections)
        if not filtered:
            return {"count": 0, "avg_bbox_area": 0,
                    "min_conf": 0, "max_conf": 0, "avg_conf": 0}

        areas = [(d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) for d in filtered]
        confs = [d["conf"] for d in filtered]
        return {
            "count"        : len(filtered),
            "avg_bbox_area": float(np.mean(areas)),
            "min_conf"     : float(np.min(confs)),
            "max_conf"     : float(np.max(confs)),
            "avg_conf"     : float(np.mean(confs)),
        }


# ─── Demo ─────────────────────────────────────────────────────────────────────

def _demo():
    import matplotlib.pyplot as plt

    print("[Demo] Sentetik tespit verisi oluşturuluyor...")

    rng = np.random.default_rng(42)
    N   = 50  # Sahte tespit sayısı
    IMG_W, IMG_H = 1280, 720

    fake_dets = []
    for _ in range(N):
        x1 = rng.integers(0, IMG_W - 40)
        y1 = rng.integers(0, IMG_H - 40)
        w  = rng.integers(10, 40)
        h  = rng.integers(10, 50)
        fake_dets.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x1 + w), "y2": float(y1 + h),
            "cx": float(x1 + w / 2), "cy": float(y1 + h / 2),
            "conf": float(rng.uniform(0.25, 0.99)),
            "cls": 0,
        })

    counter = PersonCounter(conf_threshold=0.30)

    print(f"\n[Sayım] Toplam tespit (ham)          : {N}")
    print(f"[Sayım] Güven filtresi sonrası       : {counter.count(fake_dets)}")
    print(f"[Sayım] Sol yarı ROI (x<640)         : {counter.count_in_roi(fake_dets, (0, 0, 640, IMG_H))}")
    print(f"[Sayım] Sağ yarı ROI (x>640)         : {counter.count_in_roi(fake_dets, (640, 0, IMG_W, IMG_H))}")

    stats = counter.density_stats(fake_dets)
    print(f"\n[İstatistik] {stats}")

    # Izgara sayımı
    grid = counter.count_in_grid(fake_dets, IMG_W, IMG_H, rows=4, cols=4)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Kişi Sayısı")
    ax.set_title("Hücre Bazlı Kişi Yoğunluğu (4×4 Izgara)")
    ax.set_xlabel("Sütun"); ax.set_ylabel("Satır")
    for r in range(4):
        for c in range(4):
            ax.text(c, r, str(grid[r, c]), ha="center", va="center",
                    color="white" if grid[r, c] > grid.max() * 0.5 else "black",
                    fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = Path("reports/demo_counter_grid.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n[Demo] Izgara haritası kaydedildi → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo:
        _demo()
