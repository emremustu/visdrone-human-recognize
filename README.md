# UAV Human Detection System

VisDrone veri seti üzerinde YOLOv8 + SAHI kullanarak drone görüntülerinde insan tespiti, sayımı, grup analizi ve ısı haritası üretimi yapan modüler bir sistem.

## Özellikler

| Modül | Açıklama |
|---|---|
| YOLOv8 | Gerçek zamanlı nesne tespiti |
| SAHI | Küçük nesne iyileştirmesi (dilimlenmiş inference) |
| DBSCAN | Kalabalık grup tespiti |
| Heatmap | İnsan yoğunluk haritası |
| mAP Metrics | Precision / Recall / AP@0.5 / AP@0.75 |

---

## Kurulum

```bash
cd c:\Users\yusuf\Documents\visdrone-human-recognize

# Ortam oluştur (opsiyonel ama önerilir)
python -m venv .venv
.venv\Scripts\activate   # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

> **GPU için:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

## Proje Yapısı

```
visdrone-human-recognize/
├── configs/
│   └── visdrone.yaml           # YOLOv8 dataset config (nc=1, person)
├── dataset/
│   ├── visdrone_to_yolo.py     # VisDrone → YOLO format dönüştürücü
│   └── dataset_analyzer.py     # Dataset istatistik analizi
├── inference/
│   ├── detector.py             # HumanDetector sınıfı (YOLOv8 + SAHI)
│   └── predict.py              # CLI inference scripti
├── analysis/
│   ├── counter.py              # Kişi sayımı (toplam / ROI / ızgara)
│   ├── clustering.py           # DBSCAN grup tespiti
│   └── heatmap.py              # Gaussian yoğunluk haritası
├── evaluation/
│   ├── metrics.py              # IoU, AP, mAP hesaplama
│   └── evaluate.py             # Model değerlendirme runner
├── utils/
│   ├── config.py               # Merkezi sabitler
│   └── visualizer.py           # Görselleştirme araçları
├── train.py                    # YOLOv8 eğitim scripti
└── requirements.txt
```

---

## Hızlı Başlangıç

### 1. Dataset Hazırla

VisDrone'u [buradan](https://github.com/VisDrone/VisDrone-Dataset) indirin.

```bash
# VisDrone → YOLO formatına dönüştür (tüm split'ler)
python dataset/visdrone_to_yolo.py \
    --src  /path/to/VisDrone \
    --dst  data/ \
    --all \
    --verify

# Dataset istatistiklerini analiz et
python dataset/dataset_analyzer.py --data data/ --split train
```

`configs/visdrone.yaml` dosyasındaki `path` değerini `data/` dizinine göre ayarlayın.

### 2. Modeli Eğit

```bash
# Hızlı test (nano model)
python train.py --model yolov8n.pt --imgsz 640 --epochs 50 --device 0

# Üretim kalitesi (small model, yüksek çözünürlük)
python train.py --model yolov8s.pt --imgsz 1280 --epochs 150 --batch 8 --device 0

# CPU (yavaş, sadece test için)
python train.py --model yolov8n.pt --imgsz 640 --epochs 10 --device cpu
```

Eğitim çıktısı: `runs/train/visdrone_human/weights/best.pt`

### 3. Inference

```bash
# Tek görüntü — standart
python inference/predict.py \
    --source drone_image.jpg \
    --model  runs/train/visdrone_human/weights/best.pt \
    --save

# Tek görüntü — SAHI (küçük nesneler için)
python inference/predict.py \
    --source drone_image.jpg \
    --model  best.pt \
    --sahi --save --json

# Klasör
python inference/predict.py \
    --source images/ \
    --model  best.pt \
    --sahi --save --out output/

# Video
python inference/predict.py \
    --source flight.mp4 \
    --model  best.pt \
    --sahi --save
```

### 4. Analiz

```python
from inference.detector  import HumanDetector
from analysis.counter    import PersonCounter
from analysis.clustering import GroupAnalyzer
from analysis.heatmap    import HeatmapGenerator
import cv2

# Tespit
detector = HumanDetector("best.pt", use_sahi=True)
image    = cv2.imread("drone.jpg")
dets     = detector.detect_sahi(image)

# Sayım
counter = PersonCounter()
print(f"Toplam kişi: {counter.count(dets)}")
print(f"Sol bölge  : {counter.count_in_roi(dets, (0, 0, 640, 720))}")

# Kümeleme (grup analizi)
analyzer = GroupAnalyzer(eps=50, min_samples=2)
result   = analyzer.analyze(dets)
analyzer.print_report(result)

# Isı haritası
h, w = image.shape[:2]
gen      = HeatmapGenerator(sigma=20)
density  = gen.generate(dets, h, w)
overlay  = gen.overlay(image, density)
cv2.imwrite("heatmap.jpg", overlay)
```

### 5. Demo Modları (Gerçek Veri Olmadan Test)

```bash
python analysis/counter.py    --demo   # Izgara yoğunluk haritası
python analysis/clustering.py --demo   # DBSCAN kümeleme görselleştirme
python analysis/heatmap.py    --demo   # Gaussian density map
python evaluation/metrics.py  --demo   # PR eğrisi
```

Çıktılar `reports/` klasörüne kaydedilir.

### 6. Model Değerlendirme

```bash
python evaluation/evaluate.py \
    --model runs/train/visdrone_human/weights/best.pt \
    --data  data/ \
    --split test \
    --sahi  \
    --json
```

---

## Konfigürasyon

Tüm eşik değerleri ve parametreler `utils/config.py` dosyasında merkezi olarak yönetilir:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.30 | Tespit kabul eşiği |
| `IOU_THRESHOLD` | 0.45 | NMS IoU eşiği |
| `SAHI_SLICE_HEIGHT/WIDTH` | 640 | SAHI dilim boyutu |
| `SAHI_OVERLAP_RATIO` | 0.20 | Dilim örtüşmesi |
| `DBSCAN_EPS` | 50 | Kümeleme komşuluk yarıçapı |
| `DBSCAN_MIN_SAMPLES` | 2 | Minimum küme boyutu |
| `HEATMAP_SIGMA` | 20 | Gaussian genişliği |

---

## Teknik Notlar

### Küçük Nesne Problemi
Drone görüntülerinde insanlar genellikle 10-30 piksel boyutundadır (toplam görüntünün ~%0.01'i).
Bu problemi çözen yaklaşımlar:

1. **SAHI**: Görüntüyü 640×640 dilimlere böler, her dilim bağımsız işlenir
2. **Yüksek çözünürlük**: `imgsz=1280` küçük nesneleri daha iyi çözer
3. **Copy-paste augmentation**: Küçük nesne örneklerini çoğaltır
4. **Ölçek augmentasyonu**: Modeli farklı nesne boyutlarına karşı güçlendirir

### DBSCAN Parametreleri
- `eps` (komşuluk yarıçapı): Drone yüksekliğine göre ayarlanmalı
  - Alçak irtifa (<50m) → eps=30-50
  - Orta irtifa (50-100m) → eps=50-80
  - Yüksek irtifa (>100m) → eps=80-120
- `min_samples=2`: 2 ve üzeri kişi bir grup oluşturur

### Eğitim Önerileri
- **GPU bellek:** imgsz=1280 için en az 8GB VRAM önerilir
- **Transfer learning:** `yolov8n.pt` ile başlayın, sonra `yolov8s.pt`'ye geçin
- **Epoch:** 100+ epoch VisDrone için makul başlangıç
- **Early stopping:** `patience=50` ile en iyi noktada durar
