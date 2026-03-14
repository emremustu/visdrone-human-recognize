"""
train.py — YOLOv8 Eğitim Script'i (VisDrone İnsan Tespiti)

Bu betik YOLOv8 modelini VisDrone veri seti üzerinde eğitir.
Küçük nesne problemini çözmek için:
  - Yüksek çözünürlük (imgsz=1280 önerilir)
  - Agresif augmentation (mosaic, flipud, mixup)
  - Optimize edilmiş hiperparametreler kullanılır.

Kullanım:
    # Hızlı test (nano model, düşük çözünürlük)
    python train.py --model yolov8n.pt --imgsz 640 --epochs 50

    # Üretim kalitesi (small model, yüksek çözünürlük)
    python train.py --model yolov8s.pt --imgsz 1280 --epochs 150 --batch 8

    # WandB ile (opsiyonel)
    python train.py --model yolov8n.pt --wandb
"""

import sys
import argparse
from pathlib import Path

# ─── Proje kökü ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.config import (
    TRAIN_EPOCHS, TRAIN_IMAGE_SIZE, TRAIN_BATCH_SIZE,
    TRAIN_WORKERS, TRAIN_BASE_MODEL, TRAIN_PROJECT_DIR, TRAIN_RUN_NAME
)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 VisDrone eğitim scripti")

    # Model
    p.add_argument("--model",   default=TRAIN_BASE_MODEL,
                   choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
                   help="Başlangıç ağırlık dosyası (transfer learning)")

    # Dataset
    p.add_argument("--data",    default=str(ROOT / "configs" / "visdrone.yaml"),
                   help="Dataset YAML config dosyası")

    # Eğitim parametreleri
    p.add_argument("--epochs",  type=int,   default=TRAIN_EPOCHS)
    p.add_argument("--imgsz",   type=int,   default=TRAIN_IMAGE_SIZE)
    p.add_argument("--batch",   type=int,   default=TRAIN_BATCH_SIZE)
    p.add_argument("--workers", type=int,   default=TRAIN_WORKERS)
    p.add_argument("--device",  default="0", help="CUDA device (0, 0,1, cpu)")

    # Çıktı
    p.add_argument("--project", default=TRAIN_PROJECT_DIR)
    p.add_argument("--name",    default=TRAIN_RUN_NAME)

    # Seçenekler
    p.add_argument("--resume",     action="store_true", help="Son checkpoint'ten devam et")
    p.add_argument("--wandb",      action="store_true", help="WandB loglama aktif et")
    p.add_argument("--mlflow",     action="store_true", help="MLflow loglama aktif et")
    p.add_argument("--no-augment", action="store_true", help="Augmentation'ı devre dışı bırak")

    return p.parse_args()


def get_augmentation_params(disabled: bool = False) -> dict:
    """
    Drone görüntüleri için optimize edilmiş augmentation parametreleri.

    Küçük nesne problemi için:
    - Yüksek mosaic (0.9): Farklı sayım senaryoları yaratır
    - Yüksek flipud (0.5): Drone görüntüleri için dikey flip geçerlidir
    - Ölçek 0.9: Küçük nesneleri daha da küçülterek model genelleştirmeyi artırır
    - Mixup düşük tutulur: Küçük nesneleri bulanıklaştırabilir
    """
    if disabled:
        return {
            "mosaic": 0.0, "mixup": 0.0, "flipud": 0.0,
            "fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        }
    return {
        "hsv_h"    : 0.015,   # Renk tonu jitter
        "hsv_s"    : 0.7,     # Doygunluk jitter
        "hsv_v"    : 0.4,     # Parlaklık jitter
        "degrees"  : 5.0,     # Küçük açısal rotasyon
        "translate": 0.1,     # Yatay/dikey öteleme
        "scale"    : 0.9,     # Ölçekleme aralığı
        "shear"    : 2.0,     # Kesme transformasyonu
        "perspective": 0.0,   # Perspektif (drone'da genelde 0)
        "flipud"   : 0.5,     # Dikey flip — drone için geçerli
        "fliplr"   : 0.5,     # Yatay flip
        "mosaic"   : 0.9,     # Mozaik (4 görüntü birleştir) — küçük nesne için kritik
        "mixup"    : 0.1,     # MixUp karıştırma
        "copy_paste": 0.2,    # Küçük nesneleri kopyalayıp yapıştır
    }


def setup_logging(args) -> None:
    """Opsiyonel W&B ve MLflow entegrasyonları için kurulum."""
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project="visdrone-human-detection",
                name=args.name,
                config=vars(args),
            )
            print("[WandB] Oturum başlatıldı ✅")
        except ImportError:
            print("[WandB] 'pip install wandb' ile kurunuz")

    if args.mlflow:
        try:
            import mlflow
            mlflow.set_experiment("visdrone-human-detection")
            mlflow.start_run(run_name=args.name)
            mlflow.log_params(vars(args))
            print("[MLflow] Deney başlatıldı ✅")
        except ImportError:
            print("[MLflow] 'pip install mlflow' ile kurunuz")


def main():
    args = parse_args()

    # YOLO'yu geç import et (içe aktarma süresini azaltmak için)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics bulunamadı. 'pip install ultralytics' çalıştırın.")
        sys.exit(1)

    # Logging setup
    setup_logging(args)

    # Modeli yükle
    if args.resume:
        # 'resume' modunda son best.pt ya da last.pt yüklenir
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"❌ Checkpoint bulunamadı: {last_ckpt}")
            sys.exit(1)
        model = YOLO(str(last_ckpt))
        print(f"[Resume] Checkpoint yüklendi: {last_ckpt}")
    else:
        model = YOLO(args.model)
        print(f"[Train] Başlangıç modeli: {args.model}")

    # Augmentation parametreleri
    aug_params = get_augmentation_params(disabled=args.no_augment)

    print(f"\n{'='*60}")
    print(f"  YOLOv8 Eğitim Başlıyor")
    print(f"{'='*60}")
    print(f"  Model    : {args.model}")
    print(f"  Dataset  : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  ImgSize  : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device}")
    print(f"  Proje    : {args.project}/{args.name}")
    print(f"{'='*60}\n")

    # Eğitim
    results = model.train(
        data       = args.data,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        workers    = args.workers,
        device     = args.device,
        project    = args.project,
        name       = args.name,
        exist_ok   = False,
        pretrained = True,
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,           # Final LR = lr0 * lrf
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3.0,
        warmup_momentum = 0.8,
        box        = 7.5,            # Box loss gain (küçük nesneler için artırılabilir)
        cls        = 0.5,            # Classification loss gain
        dfl        = 1.5,            # Distribution focal loss gain
        close_mosaic = 10,           # Son 10 epoch mosaic'i kapat
        amp        = True,           # Automatic Mixed Precision (VRAM tasarrufu)
        patience   = 50,             # Early stopping (50 epoch iyileşme yoksa dur)
        save       = True,
        save_period = 10,            # Her 10 epoch'ta checkpoint
        plots      = True,           # Eğitim metrikleri ve confusion matrix
        val        = True,
        verbose    = True,
        **aug_params,
    )

    print(f"\n✅ Eğitim tamamlandı!")
    print(f"   En iyi model: {args.project}/{args.name}/weights/best.pt")
    print(f"   Sonuçlar   : {args.project}/{args.name}/results.csv")

    # Val metrikleri özetle
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        print(f"\n── Validation Metrikleri ──")
        print(f"   mAP@0.5       : {rd.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   mAP@0.5:0.95  : {rd.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"   Precision     : {rd.get('metrics/precision(B)', 0):.4f}")
        print(f"   Recall        : {rd.get('metrics/recall(B)', 0):.4f}")


if __name__ == "__main__":
    main()
