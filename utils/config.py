"""
utils/config.py — Merkezi konfigürasyon sabitleri

Bu dosya, proje genelinde kullanılan tüm sabit değerleri içerir.
Bir değeri değiştirmeniz gerektiğinde yalnızca bu dosyayı düzenlemeniz yeterlidir.
"""

# ─── Inference ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.30   # Minimum güven skoru (bounding box kabul eşiği)
IOU_THRESHOLD        = 0.45   # NMS IoU eşiği
MAX_DETECTIONS       = 1000   # Kare başına maksimum tespit sayısı
DEVICE               = "cpu"  # "cuda", "mps" veya "cpu"

# ─── SAHI (Sliced Inference) ───────────────────────────────────────────────────
SAHI_SLICE_HEIGHT    = 640    # Her dilim yüksekliği (piksel)
SAHI_SLICE_WIDTH     = 640    # Her dilim genişliği  (piksel)
SAHI_OVERLAP_RATIO   = 0.20   # Dilimler arası örtüşme oranı (0.0 – 1.0)

# ─── DBSCAN Clustering ────────────────────────────────────────────────────────
DBSCAN_EPS           = 50     # Komşuluk yarıçapı (piksel cinsinden merkez mesafesi)
DBSCAN_MIN_SAMPLES   = 2      # Bir küme oluşturmak için minimum nokta sayısı

# ─── Heatmap ──────────────────────────────────────────────────────────────────
HEATMAP_SIGMA        = 20     # Gaussian kernel standart sapması
HEATMAP_ALPHA        = 0.55   # Overlay şeffaflığı (0: tam transparan, 1: tam opak)
HEATMAP_COLORMAP     = "jet"  # Matplotlib colormap adı

# ─── VisDrone İnsan Kategorileri ─────────────────────────────────────────────
VISDRONE_PERSON_CATS = {1, 2}  # 1: pedestrian, 2: people (kalabalık)

# ─── Eğitim (Training) ────────────────────────────────────────────────────────
TRAIN_EPOCHS         = 100
TRAIN_IMAGE_SIZE     = 640
TRAIN_BATCH_SIZE     = 16
TRAIN_WORKERS        = 8
TRAIN_BASE_MODEL     = "yolov8n.pt"   # nano | small=yolov8s.pt | medium=yolov8m.pt
TRAIN_PROJECT_DIR    = "runs/train"
TRAIN_RUN_NAME       = "visdrone_human"

# ─── Değerlendirme (Evaluation) ───────────────────────────────────────────────
EVAL_IOU_THRESHOLDS  = [0.50, 0.75]   # AP hesaplama için IoU eşikleri
