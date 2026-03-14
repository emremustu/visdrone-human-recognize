"""
utils/visualizer.py — Görselleştirme yardımcı fonksiyonları

Bounding box çizimi, küme renklendirme ve heatmap overlay işlemlerini sağlar.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple


# Küme renkleri (DBSCAN çıktısı için farklı kümeler farklı renlerde gösterilir)
CLUSTER_PALETTE = [
    (255, 87,  34),   # turuncu
    (33,  150, 243),  # mavi
    (76,  175, 80),   # yeşil
    (156, 39,  176),  # mor
    (255, 193, 7),    # sarı
    (0,   188, 212),  # cyan
    (244, 67,  54),   # kırmızı
    (63,  81,  181),  # indigo
]
NOISE_COLOR = (150, 150, 150)  # DBSCAN gürültü noktaları için gri


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_conf: bool = True,
) -> np.ndarray:
    """
    Görüntü üzerine tespit edilen bounding box'ları çizer.

    Args:
        image:      BGR formatında NumPy görüntüsü
        detections: [{"x1":int,"y1":int,"x2":int,"y2":int,"conf":float}, ...] listesi
        color:      BGR renk tuple'ı
        thickness:  Çizgi kalınlığı
        show_conf:  Güven skorunu göster

    Returns:
        Çizilmiş görüntü (orijinal değiştirilmez)
    """
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        conf = det.get("conf", 0.0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if show_conf:
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                img, label, (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )
    return img


def draw_clusters(
    image: np.ndarray,
    detections: List[Dict],
    labels: np.ndarray,
    draw_convex_hull: bool = True,
) -> np.ndarray:
    """
    DBSCAN kümelerini farklı renklerle çizer ve isteğe bağlı convex hull gösterir.

    Args:
        image:            BGR görüntüsü
        detections:       Tespit listesi (her elemanın cx/cy alanı olmalı)
        labels:           DBSCAN etiket dizisi (detections ile eş boyutlu)
        draw_convex_hull: Her kümenin dış sınırını çiz

    Returns:
        Renklendirilmiş görüntü
    """
    img = image.copy()
    unique_labels = set(labels)

    # Her küme için nokta grupları
    clusters: Dict[int, List] = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(detections[i])

    for lbl, dets in clusters.items():
        if lbl == -1:
            color = NOISE_COLOR
        else:
            color = CLUSTER_PALETTE[lbl % len(CLUSTER_PALETTE)]

        centers = []
        for det in dets:
            cx = int(det.get("cx", (det["x1"] + det["x2"]) / 2))
            cy = int(det.get("cy", (det["y1"] + det["y2"]) / 2))
            x1, y1 = int(det["x1"]), int(det["y1"])
            x2, y2 = int(det["x2"]), int(det["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (cx, cy), 4, color, -1)
            centers.append([cx, cy])

        # Convex hull
        if draw_convex_hull and lbl != -1 and len(centers) >= 3:
            pts = np.array(centers, dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.polylines(img, [hull], True, color, 1, cv2.LINE_AA)

        # Küme etiketi
        if centers:
            mx = int(np.mean([c[0] for c in centers]))
            my = int(np.mean([c[1] for c in centers])) - 10
            tag = f"G{lbl}" if lbl != -1 else "noise"
            cv2.putText(
                img, f"{tag} ({len(dets)})", (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

    return img


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Isı haritasını orijinal görüntü üzerine bindirerek döndürür.

    Args:
        image:    BGR görüntüsü
        heatmap:  0-255 arası uint8 tek-kanal harita
        alpha:    Isı haritası opaklığı
        colormap: cv2 colormap sabiti

    Returns:
        Overlay uygulanmış BGR görüntüsü
    """
    h, w = image.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    colored    = cv2.applyColorMap(hm_resized, colormap)
    blended    = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return blended


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Matplotlib figürü diske kaydeder."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Kaydedildi → {path}")
