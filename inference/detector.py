"""
inference/detector.py — YOLOv8 + SAHI İnsan Tespit Motoru

Küçük nesne problemini çözmek için iki mod sunar:
  1. Standart YOLOv8 inference — hızlı, yüksek çözünürlük görüntülerde yeterli
  2. SAHI SlicedInference   — görüntüyü örtüşen dilimlere ayırarak küçük nesneleri
                               çok daha yüksek hassasiyetle tespit eder.

Kullanım:
    from inference.detector import HumanDetector

    detector = HumanDetector(model_path="runs/train/visdrone_human/weights/best.pt")

    # Standart
    detections = detector.detect(image)

    # SAHI ile
    detections = detector.detect_sahi(image)
"""

import sys
from pathlib import Path
from typing import List, Dict, Union

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import (
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS,
    DEVICE, SAHI_SLICE_HEIGHT, SAHI_SLICE_WIDTH, SAHI_OVERLAP_RATIO
)


class HumanDetector:
    """
    Drone görüntülerinde insan tespiti için YOLOv8 + SAHI wrapper sınıfı.

    Args:
        model_path:  YOLOv8 ağırlık dosyası (.pt)
        conf:        Güven skoru eşiği
        iou:         NMS IoU eşiği
        device:      "cuda", "mps" veya "cpu"
        use_sahi:    Varsayılan inference modunu SAHI yap
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = CONFIDENCE_THRESHOLD,
        iou: float  = IOU_THRESHOLD,
        device: str = DEVICE,
        use_sahi: bool = False,
    ):
        self.conf     = conf
        self.iou      = iou
        self.device   = device
        self.use_sahi = use_sahi
        self._model_path = model_path

        self._yolo_model  = None   # lazy-loaded
        self._sahi_model  = None   # lazy-loaded

        print(f"[Detector] Model: {model_path} | conf={conf} | iou={iou} | device={device}")

    # ─── Lazy Loading ─────────────────────────────────────────────────────────

    def _load_yolo(self):
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("'pip install ultralytics' ile ultralytics kurunuz.")
            self._yolo_model = YOLO(self._model_path)
            print(f"[Detector] YOLOv8 modeli yüklendi: {self._model_path}")
        return self._yolo_model

    def _load_sahi(self):
        if self._sahi_model is None:
            try:
                from sahi import AutoDetectionModel
            except ImportError:
                raise ImportError("'pip install sahi' ile SAHI kurunuz.")
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type  = "yolov8",
                model_path  = self._model_path,
                confidence_threshold = self.conf,
                device      = self.device,
            )
            print(f"[Detector] SAHI modeli yüklendi: {self._model_path}")
        return self._sahi_model

    # ─── Deteksiyonları Normalize Et ──────────────────────────────────────────

    @staticmethod
    def _parse_yolo_result(result) -> List[Dict]:
        """
        Ultralytics Result nesnesini standart sözlük listesine dönüştürür.

        Returns:
            [{"x1", "y1", "x2", "y2", "conf", "cls", "cx", "cy", "w", "h"}, ...]
        """
        detections = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy  = boxes.xyxy.cpu().numpy()   # (N, 4)
        confs = boxes.conf.cpu().numpy()   # (N,)
        clss  = boxes.cls.cpu().numpy()    # (N,)

        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            if cls_id not in [0, 1]:
                continue
            
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                "x1"  : float(x1),
                "y1"  : float(y1),
                "x2"  : float(x2),
                "y2"  : float(y2),
                "conf": float(confs[i]),       
                "cls" : cls_id,
                "cx"  : float((x1 + x2) / 2),  
                "cy"  : float((y1 + y2) / 2),  
                "w"   : float(x2 - x1),        
                "h"   : float(y2 - y1),        
            })
        return detections

    @staticmethod
    def _parse_sahi_result(prediction_result) -> List[Dict]:
        """
        SAHI PredictionResult nesnesini standart sözlük listesine dönüştürür.
        """
        detections = []
        for obj in prediction_result.object_prediction_list:
            cls_id = int(obj.category.id)
            if cls_id not in [0, 1]:
                continue

            bbox = obj.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            detections.append({
                "x1"  : float(x1),
                "y1"  : float(y1),
                "x2"  : float(x2),
                "y2"  : float(y2),
                "conf": float(obj.score.value),
                "cls" : cls_id,  
                "cx"  : float((x1 + x2) / 2),  
                "cy"  : float((y1 + y2) / 2),  
                "w"   : float(x2 - x1),        
                "h"   : float(y2 - y1),        
            })
        return detections

    # ─── Ana Inference Metodları ──────────────────────────────────────────────

    def detect(
        self,
        image: Union[np.ndarray, str, Path],
        max_det: int = MAX_DETECTIONS,
    ) -> List[Dict]:
        """
        Standart YOLOv8 inference.

        Args:
            image:   BGR NumPy görüntüsü veya dosya yolu
            max_det: Maksimum tespit sayısı

        Returns:
            detections listesi
        """
        model  = self._load_yolo()
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None:
            raise ValueError("Görüntü okunamadı.")

        results = model.predict(
            source   = image,
            conf     = self.conf,
            iou      = self.iou,
            device   = self.device,
            max_det  = max_det,
            verbose  = False,
            classes  = [0],   # Sadece person sınıfı
        )
        return self._parse_yolo_result(results[0])

    def detect_sahi(
        self,
        image: Union[np.ndarray, str, Path],
        slice_h: int   = SAHI_SLICE_HEIGHT,
        slice_w: int   = SAHI_SLICE_WIDTH,
        overlap: float = SAHI_OVERLAP_RATIO,
    ) -> List[Dict]:
        """
        SAHI SlicedInference ile küçük nesne tespiti.

        Görüntü örtüşen dilimlere (slice_h x slice_w) bölünür,
        her dilim bağımsız olarak işlenir ve sonuçlar birleştirilir (NMS).

        Args:
            image:   BGR NumPy görüntüsü veya dosya yolu
            slice_h: Dilim yüksekliği (piksel)
            slice_w: Dilim genişliği (piksel)
            overlap: Örtüşme oranı [0.0 – 1.0]

        Returns:
            detections listesi
        """
        try:
            from sahi.predict import get_sliced_prediction
        except ImportError:
            raise ImportError("'pip install sahi' ile SAHI kurunuz.")

        sahi_model = self._load_sahi()

        # SAHI dosya yolu veya PIL/numpy kabul eder
        if isinstance(image, np.ndarray):
            # NumPy → PIL → yolu kaydet (SAHI ndarray ile çalışır)
            source = image
        else:
            source = str(image)

        result = get_sliced_prediction(
            image                       = source,
            detection_model             = sahi_model,
            slice_height                = slice_h,
            slice_width                 = slice_w,
            overlap_height_ratio        = overlap,
            overlap_width_ratio         = overlap,
            perform_standard_pred       = True,   # Tam görüntü + dilimler
            postprocess_type            = "NMM",  # Non-Maximum Merging
            postprocess_match_metric    = "IOU",
            postprocess_match_threshold = self.iou,
            verbose                     = 0,
        )
        return self._parse_sahi_result(result)

    def detect_auto(
        self,
        image: Union[np.ndarray, str, Path],
    ) -> List[Dict]:
        """
        Yapılandırmadaki `use_sahi` değerine göre otomatik mod seçer.
        """
        if self.use_sahi:
            return self.detect_sahi(image)
        return self.detect(image)

    # ─── Video / Frame Akışı ──────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_path: str,
        use_sahi: bool = False,
        show: bool = False,
    ) -> None:
        """
        Video dosyasını kare kare işler ve annotated video yazar.

        Args:
            video_path:  Giriş video dosyası
            output_path: Çıkış video dosyası
            use_sahi:    SAHI kullansın mı?
            show:        Gerçek zamanlı önizleme
        """
        from utils.visualizer import draw_detections

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        print(f"\n[Video] İşleniyor: {video_path} ({total} kare, {fps:.1f} FPS)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = self.detect_sahi(frame) if use_sahi else self.detect(frame)
            annotated = draw_detections(frame, dets)

            # Kare + sayaç ekle
            label = f"Frame {frame_idx+1}/{total} | Kisi: {len(dets)}"
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

            writer.write(annotated)
            if show:
                cv2.imshow("Human Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
        print(f"[Video] Kaydedildi → {output_path}")
