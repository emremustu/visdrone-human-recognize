"""
analysis/heatmap.py — İnsan Yoğunluk Isı Haritası Üretimi

Tespit edilen kişilerin bounding box merkezlerinden Gaussian kernel konvolüsyonu
ile yoğunluk haritası (density map) oluşturur.

Çalışma prensibi:
  1. Her kişi merkezine Gaussian blob yerleştir (sigma ile kontrol edilir)
  2. Tüm bloblarin toplamı density map'i oluşturur
  3. Min-max normalize et → 0-255 uint8'e ölçekle
  4. cv2 colormap uygula ve orijinal görüntü ile birleştir

Demo kullanım:
    python analysis/heatmap.py --demo
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import (
    HEATMAP_SIGMA, HEATMAP_ALPHA, HEATMAP_COLORMAP,
    CONFIDENCE_THRESHOLD
)


class HeatmapGenerator:
    """
    İnsan tespit sonuçlarından yoğunluk ısı haritası üretir.

    Args:
        sigma:          Gaussian kernel standart sapması (piksel).
                        Büyük sigma → daha geniş/bulanık blob
        alpha:          Görüntü üzerine overlay opaklığı [0..1]
        conf_threshold: Kullanılacak minimum güven skoru
    """

    def __init__(
        self,
        sigma          : float = HEATMAP_SIGMA,
        alpha          : float = HEATMAP_ALPHA,
        conf_threshold : float = CONFIDENCE_THRESHOLD,
    ):
        self.sigma          = sigma
        self.alpha          = alpha
        self.conf_threshold = conf_threshold

    # ─── Yardımcı ─────────────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """
        2D Gaussian kernel oluşturur.

        Args:
            size:  Kernel boyutu (tek sayı, örn. 61)
            sigma: Standart sapma

        Returns:
            (size x size) float32 array, toplam 1.0'a normalize
        """
        ax   = np.arange(size) - size // 2
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return (kernel / kernel.sum()).astype(np.float32)

    def _get_centers(self, detections: List[Dict]) -> List[Tuple[int, int]]:
        """Güven eşiğini geçen tespitlerin piksel merkezlerini döner."""
        centers = []
        for d in detections:
            if d.get("conf", 1.0) < self.conf_threshold:
                continue
            cx = int(d.get("cx", (d["x1"] + d["x2"]) / 2))
            cy = int(d.get("cy", (d["y1"] + d["y2"]) / 2))
            centers.append((cx, cy))
        return centers

    # ─── Ana Metodlar ──────────────────────────────────────────────────────────

    def generate(
        self,
        detections : List[Dict],
        img_h      : int,
        img_w      : int,
        normalize  : bool = True,
    ) -> np.ndarray:
        """
        Float32 density map oluşturur.

        Her tespit merkezi için Gaussian blob, float accumulator'a eklenir.
        Bu yöntem scipy.ndimage.gaussian_filter yerine manuel konvolüsyon kullanır
        (SAHI ile uyumluluk ve bağımlılıkları azaltmak için).

        Args:
            detections: Tespit listesi
            img_h, img_w: Çıktı haritası boyutları
            normalize: True → [0,1] aralığına normalize et

        Returns:
            float32 density map, shape=(img_h, img_w)
        """
        density = np.zeros((img_h, img_w), dtype=np.float32)
        centers = self._get_centers(detections)

        if not centers:
            return density

        # Kernel boyutunu sigma'ya göre belirle (sigma*6, tek sayı olsun)
        k_size = int(self.sigma * 6) | 1   # bitwise OR 1 → her zaman tek
        k_size = max(k_size, 3)
        half   = k_size // 2
        kernel = self._gaussian_kernel(k_size, self.sigma)

        for cx, cy in centers:
            # Görüntü sınırları içinde kırp
            x1 = max(cx - half, 0);    x2 = min(cx + half + 1, img_w)
            y1 = max(cy - half, 0);    y2 = min(cy + half + 1, img_h)

            # Kernel kırpma ofsetleri
            kx1 = half - (cx - x1);   kx2 = kx1 + (x2 - x1)
            ky1 = half - (cy - y1);   ky2 = ky1 + (y2 - y1)

            density[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

        if normalize and density.max() > 0:
            density = density / density.max()

        return density

    def to_colormap(
        self,
        density    : np.ndarray,
        colormap   : str = HEATMAP_COLORMAP,
    ) -> np.ndarray:
        """
        Float density map'i renkli BGR görüntüye dönüştürür.

        Args:
            density:  [0..1] float32 array
            colormap: Matplotlib colormap adı ("jet", "inferno", "hot", ...)

        Returns:
            BGR uint8, shape=(H, W, 3)
        """
        import matplotlib.pyplot as plt
        cmap   = plt.get_cmap(colormap)
        rgba   = (cmap(density) * 255).astype(np.uint8)   # RGBA
        rgb    = rgba[:, :, :3]
        bgr    = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def overlay(
        self,
        image    : np.ndarray,
        density  : np.ndarray,
        colormap : str  = HEATMAP_COLORMAP,
        alpha    : Optional[float] = None,
    ) -> np.ndarray:
        """
        Density map'i orijinal görüntü üzerine bindirir.

        Args:
            image:    BGR görüntüsü
            density:  generate() çıktısı
            colormap: Colormap adı
            alpha:    Overlay opaklığı (None → self.alpha)

        Returns:
            Blended BGR görüntüsü
        """
        alpha = alpha if alpha is not None else self.alpha
        h, w  = image.shape[:2]

        hm_resized = cv2.resize(density, (w, h))
        colored    = self.to_colormap(hm_resized, colormap)
        blended    = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
        return blended

    def save(
        self,
        density  : np.ndarray,
        path     : str,
        colormap : str = HEATMAP_COLORMAP,
    ) -> None:
        """Density map'i renkli png olarak kaydeder."""
        colored = self.to_colormap(density, colormap)
        # 0-255 uint8'e çevir
        uint8   = (density * 255).astype(np.uint8)
        cv2.imwrite(path, colored)
        print(f"[Heatmap] Kaydedildi → {path}")

    def process_sequence(
        self,
        frame_detections : List[List[Dict]],
        img_h            : int,
        img_w            : int,
    ) -> np.ndarray:
        """
        Video sekansı için birikimli density map üretir.
        Her frame'in density map'i toplanır (temporal heatmap).

        Args:
            frame_detections: Her frame için tespit listelerinin listesi
            img_h, img_w:     Frame boyutları

        Returns:
            Normalize edilmiş float32 cumulative density map
        """
        cumulative = np.zeros((img_h, img_w), dtype=np.float32)
        for dets in frame_detections:
            cumulative += self.generate(dets, img_h, img_w, normalize=False)

        if cumulative.max() > 0:
            cumulative /= cumulative.max()
        return cumulative


# ─── Demo ─────────────────────────────────────────────────────────────────────

def _demo():
    import matplotlib.pyplot as plt

    print("[Demo] Sentetik tespit verisi oluşturuluyor...")
    rng = np.random.default_rng(7)

    IMG_H, IMG_W = 720, 1280

    # Kalabalık bölgeler + dağınık bireyler
    centers_crowd = [
        (rng.normal(320, 40), rng.normal(250, 40))
        for _ in range(40)
    ] + [
        (rng.normal(900, 60), rng.normal(500, 50))
        for _ in range(60)
    ] + [
        (rng.uniform(0, IMG_W), rng.uniform(0, IMG_H))
        for _ in range(15)
    ]

    fake_dets = []
    for cx, cy in centers_crowd:
        cx = float(np.clip(cx, 5, IMG_W - 5))
        cy = float(np.clip(cy, 5, IMG_H - 5))
        fake_dets.append({
            "x1": cx - 8, "y1": cy - 10, "x2": cx + 8, "y2": cy + 10,
            "cx": cx, "cy": cy,
            "conf": float(rng.uniform(0.40, 0.99)), "cls": 0,
        })

    gen     = HeatmapGenerator(sigma=25, alpha=0.60)
    density = gen.generate(fake_dets, IMG_H, IMG_W)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Isı Haritası Demo", fontsize=13, fontweight="bold")

    # Ham density
    axes[0].imshow(density, cmap="jet", origin="upper")
    axes[0].set_title("Density Map (jet)")
    axes[0].axis("off")
    plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046)

    # Drone sahnesini simüle et (gri arka plan)
    fake_img = np.full((IMG_H, IMG_W, 3), 30, dtype=np.uint8)
    for det in fake_dets:
        cv2.rectangle(
            fake_img,
            (int(det["x1"]), int(det["y1"])),
            (int(det["x2"]), int(det["y2"])),
            (180, 180, 180), 1
        )

    # Overlay
    overlay = gen.overlay(fake_img, density, colormap="jet")
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Gradient Overlay (jet)")
    axes[1].axis("off")

    # Inferno colormap
    overlay_inf = gen.overlay(fake_img, density, colormap="inferno", alpha=0.7)
    axes[2].imshow(cv2.cvtColor(overlay_inf, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Gradient Overlay (inferno)")
    axes[2].axis("off")

    # Kişi noktaları
    for ax in axes:
        for det in fake_dets:
            ax.plot(det["cx"], det["cy"], "w.", markersize=2, alpha=0.5)

    plt.tight_layout()
    out = Path("reports/demo_heatmap.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[Demo] Isı haritası kaydedildi → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--demo",  action="store_true")
    p.add_argument("--sigma", type=float, default=HEATMAP_SIGMA)
    p.add_argument("--alpha", type=float, default=HEATMAP_ALPHA)
    args = p.parse_args()

    if args.demo:
        _demo()
