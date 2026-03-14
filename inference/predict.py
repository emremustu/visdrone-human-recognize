"""
inference/predict.py — CLI Inference Script

Tek bir görüntü, klasör veya video üzerinde human detection çalıştırır.
Sonuçlar JSON formatında kaydedilebilir ve annotated görüntüler üretilebilir.

Kullanım:
    # Tek görüntü (standart)
    python inference/predict.py --source image.jpg --model best.pt

    # Tek görüntü (SAHI ile)
    python inference/predict.py --source image.jpg --model best.pt --sahi

    # Klasör (tüm jpg/png)
    python inference/predict.py --source images/ --model best.pt --sahi --save

    # Video
    python inference/predict.py --source video.mp4 --model best.pt --save
"""

import sys
import json
import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from inference.detector  import HumanDetector
from utils.visualizer    import draw_detections
from utils.config        import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, DEVICE


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    p = argparse.ArgumentParser(description="UAV İnsan Tespit — CLI Inference")
    p.add_argument("--source", required=True, help="Görüntü, klasör veya video yolu")
    p.add_argument("--model",  default="yolov8n.pt", help="YOLOv8 ağırlık dosyası (.pt)")
    p.add_argument("--conf",   type=float, default=CONFIDENCE_THRESHOLD)
    p.add_argument("--iou",    type=float, default=IOU_THRESHOLD)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--sahi",   action="store_true", help="SAHI sliced inference kullan")
    p.add_argument("--save",   action="store_true", help="Annotated görüntüleri kaydet")
    p.add_argument("--json",   action="store_true", help="Sonuçları JSON'a yaz")
    p.add_argument("--out",    default="output", help="Çıktı klasörü")
    p.add_argument("--show",   action="store_true", help="Anlık önizleme (ESC ile kapat)")
    return p.parse_args()


def process_image(
    detector: HumanDetector,
    img_path: Path,
    args,
    out_dir: Path,
) -> dict:
    """
    Tek görüntü üzerinde tespit yapar ve isteğe göre sonuçları kaydeder.

    Returns:
        Sonuç sözlüğü: {filename, count, detections, elapsed_ms}
    """
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[UYARI] Görüntü okunamadı: {img_path}")
        return {}

    t0 = time.perf_counter()
    if args.sahi:
        dets = detector.detect_sahi(image)
    else:
        dets = detector.detect(image)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  {img_path.name:<40} → {len(dets):>4} kişi  ({elapsed_ms:.1f} ms)")

    if args.save or args.show:
        annotated = draw_detections(image, dets, show_conf=True)

        # Kişi sayısı
        cv2.putText(
            annotated,
            f"Kisi: {len(dets)} | {'SAHI' if args.sahi else 'YOLO'}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2, cv2.LINE_AA
        )

        if args.save:
            out_path = out_dir / ("ann_" + img_path.name)
            cv2.imwrite(str(out_path), annotated)

        if args.show:
            cv2.imshow(img_path.name, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return {
        "filename"  : img_path.name,
        "count"     : len(dets),
        "elapsed_ms": round(elapsed_ms, 2),
        "detections": dets,
    }


def main():
    args    = parse_args()
    src     = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = HumanDetector(
        model_path = args.model,
        conf       = args.conf,
        iou        = args.iou,
        device     = args.device,
    )

    print(f"\n{'='*60}")
    print(f"  UAV İnsan Tespit — Inference")
    print(f"{'='*60}")
    print(f"  Kaynak : {src}")
    print(f"  Model  : {args.model}")
    print(f"  Mod    : {'SAHI' if args.sahi else 'Standart YOLOv8'}")
    print(f"  Çıktı  : {out_dir}")
    print(f"{'='*60}\n")

    all_results = []

    # ─── Video ────────────────────────────────────────────────────────────────
    if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
        out_video = out_dir / ("ann_" + src.name)
        detector.process_video(
            str(src), str(out_video),
            use_sahi=args.sahi,
            show=args.show,
        )
        return

    # ─── Tek Görüntü ──────────────────────────────────────────────────────────
    if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
        result = process_image(detector, src, args, out_dir)
        all_results.append(result)

    # ─── Klasör ───────────────────────────────────────────────────────────────
    elif src.is_dir():
        img_files = [
            f for f in sorted(src.iterdir())
            if f.suffix.lower() in IMAGE_EXTS
        ]
        if not img_files:
            print(f"[HATA] Klasörde görüntü bulunamadı: {src}")
            sys.exit(1)

        print(f"  {len(img_files)} görüntü işlenecek...\n")
        for img_path in img_files:
            result = process_image(detector, img_path, args, out_dir)
            if result:
                all_results.append(result)

    else:
        print(f"[HATA] Geçersiz kaynak: {src}")
        sys.exit(1)

    # ─── Özet ─────────────────────────────────────────────────────────────────
    if all_results:
        total_persons = sum(r["count"] for r in all_results)
        avg_time      = sum(r["elapsed_ms"] for r in all_results) / len(all_results)
        print(f"\n── Özet ──────────────────────────────────")
        print(f"   İşlenen görüntü     : {len(all_results)}")
        print(f"   Toplam tespit       : {total_persons}")
        print(f"   Ortalama gecikme    : {avg_time:.1f} ms/görüntü")
        print(f"   Tahmini FPS         : {1000/avg_time:.1f}")

        if args.json:
            json_out = out_dir / "detections.json"
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n[JSON] Sonuçlar kaydedildi → {json_out}")


if __name__ == "__main__":
    main()
