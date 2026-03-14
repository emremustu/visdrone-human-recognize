"""
analysis/clustering.py — DBSCAN ile İnsan Grubu Tespiti

Tespit edilen insanların bounding box merkezlerini DBSCAN ile kümelere ayırır.
Her küme bir "insan grubunu" temsil eder.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
  - Parametresiz küme sayısı (önceden K belirlemeye gerek yok)
  - Gürültü/izole kişileri -1 etiketi ile işaretler
  - Keyfi şekil kümeleri bulabilir
  - Drone görüntülerindeki kalabalıklar için idealdir

Demo kullanım:
    python analysis/clustering.py --demo
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, CONFIDENCE_THRESHOLD


class GroupAnalyzer:
    """
    Tespit edilen insanları DBSCAN ile gruplar (kümeler).

    Args:
        eps:         Komşuluk yarıçapı (piksel). Drone yüksekliğine göre ayarlanmalı.
                     Alçak irtifa → küçük eps, yüksek irtifa → büyük eps.
        min_samples: Bir küme kurmak için gereken minimum nokta sayısı.
        conf_threshold: Kümeleme öncesi uygulanan güven eşiği.
    """

    def __init__(
        self,
        eps: float             = DBSCAN_EPS,
        min_samples: int       = DBSCAN_MIN_SAMPLES,
        conf_threshold: float  = CONFIDENCE_THRESHOLD,
    ):
        self.eps            = eps
        self.min_samples    = min_samples
        self.conf_threshold = conf_threshold

    def _centers(self, detections: List[Dict]) -> np.ndarray:
        """Tespitlerden (cx, cy) merkez koordinat array'i oluşturur."""
        filtered = [d for d in detections if d.get("conf", 1.0) >= self.conf_threshold]
        if not filtered:
            return np.empty((0, 2))
        return np.array([[d["cx"], d["cy"]] for d in filtered], dtype=np.float32)

    def cluster(
        self,
        detections: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        DBSCAN kümeleme yapar.

        Args:
            detections: HumanDetector.detect() çıktısı

        Returns:
            (labels, filtered_detections)
            - labels: Her tespit için küme etiketi (-1 = gürültü)
            - filtered_detections: Güven eşiğini geçen tespitler (labels ile eş boyutlu)
        """
        filtered = [d for d in detections if d.get("conf", 1.0) >= self.conf_threshold]
        if not filtered:
            return np.array([], dtype=np.int32), []

        centers = np.array([[d["cx"], d["cy"]] for d in filtered])

        db     = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean")
        labels = db.fit_predict(centers)

        return labels, filtered

    def group_stats(
        self,
        labels: np.ndarray,
        detections: List[Dict],
    ) -> List[Dict]:
        """
        Her küme için istatistik hesaplar.

        Args:
            labels:     cluster() metodunun döndürdüğü etiket dizisi
            detections: cluster() metodunun döndürdüğü filtered_detections

        Returns:
            Her küme için:
            {
              "cluster_id"  : int,       # -1 = gürültü
              "count"       : int,       # Kümedeki kişi sayısı
              "center_x"    : float,     # Küme merkezi X
              "center_y"    : float,     # Küme merkezi Y
              "bbox"        : [x1,y1,x2,y2],  # Kümeyi kapsayan sınır kutusu
              "avg_conf"    : float,
              "is_noise"    : bool
            }
        """
        if len(labels) == 0:
            return []

        unique_labels = sorted(set(labels))
        stats = []

        for lbl in unique_labels:
            mask = labels == lbl
            group_dets = [d for d, m in zip(detections, mask) if m]

            xs1 = [d["x1"] for d in group_dets]
            ys1 = [d["y1"] for d in group_dets]
            xs2 = [d["x2"] for d in group_dets]
            ys2 = [d["y2"] for d in group_dets]
            cxs = [d["cx"] for d in group_dets]
            cys = [d["cy"] for d in group_dets]

            stats.append({
                "cluster_id" : int(lbl),
                "count"      : len(group_dets),
                "center_x"   : float(np.mean(cxs)),
                "center_y"   : float(np.mean(cys)),
                "bbox"       : [float(min(xs1)), float(min(ys1)),
                                float(max(xs2)), float(max(ys2))],
                "avg_conf"   : float(np.mean([d["conf"] for d in group_dets])),
                "is_noise"   : lbl == -1,
            })

        return sorted(stats, key=lambda x: x["count"], reverse=True)

    def analyze(self, detections: List[Dict]) -> dict:
        """
        Tek çağrıda hem cluster hem de group_stats döner.

        Returns:
            {
              "labels"      : np.ndarray,
              "filtered"    : List[Dict],
              "groups"      : List[Dict],
              "n_groups"    : int,
              "n_noise"     : int,
              "largest_group": int  (en kalabalık grubun kişi sayısı)
            }
        """
        labels, filtered = self.cluster(detections)
        groups           = self.group_stats(labels, filtered)

        n_groups = sum(1 for g in groups if not g["is_noise"])
        n_noise  = sum(g["count"] for g in groups if g["is_noise"])
        largest  = max((g["count"] for g in groups if not g["is_noise"]), default=0)

        return {
            "labels"       : labels,
            "filtered"     : filtered,
            "groups"       : groups,
            "n_groups"     : n_groups,
            "n_noise"      : n_noise,
            "largest_group": largest,
        }

    def print_report(self, result: dict) -> None:
        """Kümeleme sonuçlarını konsola yaz."""
        print(f"\n{'─'*55}")
        print(f"  DBSCAN Kümeleme Raporu")
        print(f"  eps={self.eps}  min_samples={self.min_samples}")
        print(f"{'─'*55}")
        print(f"  Toplam tespit     : {len(result['filtered'])}")
        print(f"  Küme sayısı       : {result['n_groups']}")
        print(f"  Gürültü kişi      : {result['n_noise']}")
        print(f"  En büyük grup     : {result['largest_group']} kişi")
        print(f"{'─'*55}")

        for g in result["groups"]:
            tag = f"Küme {g['cluster_id']}" if not g["is_noise"] else "Gürültü"
            print(f"  [{tag:12s}] {g['count']:>3} kişi | "
                  f"merkez=({g['center_x']:.0f},{g['center_y']:.0f}) | "
                  f"bbox=[{int(g['bbox'][0])},{int(g['bbox'][1])},{int(g['bbox'][2])},{int(g['bbox'][3])}]")
        print()


# ─── Demo ─────────────────────────────────────────────────────────────────────

def _demo():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    print("[Demo] Sentetik tespit verisi oluşturuluyor (3 grup + gürültü)...")
    rng = np.random.default_rng(0)

    # 3 grup oluştur
    group_centers   = [(200, 200), (700, 400), (1100, 600)]
    group_sizes     = [20, 35, 12]
    fake_dets       = []

    for (gcx, gcy), gsz in zip(group_centers, group_sizes):
        for _ in range(gsz):
            cx = float(gcx + rng.normal(0, 30))
            cy = float(gcy + rng.normal(0, 30))
            w, h = float(rng.integers(8, 25)), float(rng.integers(12, 30))
            fake_dets.append({
                "x1": cx - w/2, "y1": cy - h/2,
                "x2": cx + w/2, "y2": cy + h/2,
                "cx": cx, "cy": cy, "conf": float(rng.uniform(0.4, 0.99)), "cls": 0,
            })

    # Gürültü ekle
    for _ in range(8):
        cx = float(rng.integers(0, 1280))
        cy = float(rng.integers(0, 720))
        fake_dets.append({
            "x1": cx-8, "y1": cy-10, "x2": cx+8, "y2": cy+10,
            "cx": cx, "cy": cy, "conf": float(rng.uniform(0.35, 0.80)), "cls": 0,
        })

    analyzer = GroupAnalyzer(eps=70, min_samples=3)
    result   = analyzer.analyze(fake_dets)
    analyzer.print_report(result)

    # Görselleştir
    from utils.visualizer import CLUSTER_PALETTE, NOISE_COLOR
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1280); ax.set_ylim(720, 0)
    ax.set_facecolor("#1a1a2e")
    ax.set_title("DBSCAN Kümeleme — Demo", color="white", fontsize=13)
    ax.tick_params(colors="white")

    labels   = result["labels"]
    filtered = result["filtered"]

    for det, lbl in zip(filtered, labels):
        color = (np.array(NOISE_COLOR) / 255 if lbl == -1
                 else np.array(CLUSTER_PALETTE[int(lbl) % len(CLUSTER_PALETTE)]) / 255)
        rect = patches.Rectangle(
            (det["x1"], det["y1"]), det["x2"]-det["x1"], det["y2"]-det["y1"],
            linewidth=1.5, edgecolor=color, facecolor=(*color, 0.15)
        )
        ax.add_patch(rect)

    for g in result["groups"]:
        if g["is_noise"]:
            continue
        lbl   = g["cluster_id"]
        color = np.array(CLUSTER_PALETTE[lbl % len(CLUSTER_PALETTE)]) / 255
        ax.annotate(
            f"Grup {lbl}\n{g['count']} kişi",
            xy=(g["center_x"], g["center_y"]),
            color="white", fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
        )

    out = Path("reports/demo_clustering.png")
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, facecolor="#1a1a2e")
    plt.close()
    print(f"[Demo] Kümeleme grafiği → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--eps",         type=float, default=DBSCAN_EPS)
    p.add_argument("--min-samples", type=int,   default=DBSCAN_MIN_SAMPLES)
    args = p.parse_args()
    if args.demo:
        _demo()
