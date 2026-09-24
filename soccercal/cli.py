"""Command line.

    soccercal doctor                          # comprueba la instalación paso a paso
    soccercal sample                          # descarga un clip de prueba de 6 s
    soccercal track partido.mp4 -o salida     # vídeo -> datos tipo SkillCorner
    soccercal calibrate foto.png -o overlay.png
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def _doctor(args) -> int:
    ok = True

    def step(name, fn):
        nonlocal ok
        t = time.time()
        try:
            msg = fn()
            print(f"  [OK]    {name}: {msg} ({time.time() - t:.1f}s)")
        except Exception as e:  # noqa: BLE001 - report everything to the user
            ok = False
            print(f"  [FALLO] {name}: {type(e).__name__}: {e}")

    print("soccercal doctor\n")
    step("Python", lambda: f"{platform.python_version()} ({sys.executable})")

    def torch_check():
        import torch

        from .device import pick_device

        return f"torch {torch.__version__}, dispositivo = {pick_device(args.device)}"

    step("PyTorch", torch_check)

    def cv_check():
        import cv2

        return f"opencv {cv2.__version__}"

    step("OpenCV", cv_check)

    def yolo_check():
        import ultralytics

        return f"ultralytics {ultralytics.__version__}"

    step("Ultralytics (YOLO)", yolo_check)

    def weights_check():
        from .weights import get_weights

        return str(get_weights("SV_kp"))

    step("Pesos de la red de campo (265 MB, solo la 1ª vez)", weights_check)

    def run_check():
        import numpy as np

        from .field_model import FieldDetector

        det = FieldDetector(args.device)
        t = time.time()
        det([np.zeros((540, 960, 3), np.uint8)])
        return f"inferencia de prueba en {det.device}: {time.time() - t:.2f}s"

    step("Red de campo funciona", run_check)

    def yolo_run():
        import numpy as np

        from .detect import PlayerDetector

        d = PlayerDetector("yolo11m.pt", args.device)
        d([np.zeros((720, 1280, 3), np.uint8)])
        return "detector de jugadores cargado"

    step("Detector de jugadores (descarga yolo11m.pt la 1ª vez)", yolo_run)

    def ocr_check():
        from .jersey import JerseyReader

        if not JerseyReader().available:
            raise RuntimeError("falta rapidocr_onnxruntime: pip install rapidocr_onnxruntime (sin él no se leen dorsales)")
        return "lector de dorsales disponible"

    step("Lectura de dorsales (OCR)", ocr_check)
    print("\nTodo listo." if ok else "\nHay pasos con FALLO: copia el mensaje de error para diagnosticarlo.")
    return 0 if ok else 1


def _sample(args) -> int:
    from .weights import get_sample_video

    p = get_sample_video(args.out)
    print(p)
    return 0


def _track(args) -> int:
    from .config import Config
    from .pipeline import run

    cfg = Config(device=args.device, start_s=args.start, max_seconds=args.max_seconds, stride=args.stride,
                 keyframe_every=args.keyframe_every, det_model=args.model, batch=args.batch,
                 home_name=args.home_name, away_name=args.away_name, extrapolate=not args.no_extrapolate,
                 period=args.period, time_offset_s=args.time_offset, jersey_ocr=not args.no_jersey)
    t = time.time()
    res = run(args.video, args.out, cfg, render=not args.no_render, reuse_analysis=not args.redo)
    print(json.dumps(res.summary(), indent=2, ensure_ascii=False))
    print(f"\nListo en {time.time() - t:.0f} s. Resultados en: {Path(args.out).resolve()}")
    return 0


def _calibrate(args) -> int:
    import cv2

    from .calibrate import calibrate
    from .field_model import FieldDetector
    from .lines import LineMask, alignment
    from .viz import draw_calibration

    img = cv2.imread(args.image)
    if img is None:
        print(f"No se puede leer la imagen {args.image}")
        return 1
    obs = FieldDetector(args.device)([img])[0]
    lm = LineMask.from_frame(img)
    cam = calibrate(obs, line_mask=lm)
    if cam is None:
        print("No se pudo calibrar (¿se ven suficientes líneas del campo?)")
        return 2
    d = cam.to_dict()
    d["line_alignment"] = alignment(cam, lm)
    d["keypoints_detected"] = len(obs.keypoints)
    print(json.dumps(d, indent=2))
    if args.out:
        cv2.imwrite(args.out, draw_calibration(img.copy(), cam))
        print(f"Superposición guardada en {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="soccercal", description="Vídeo de retransmisión -> datos de tracking tipo SkillCorner")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (por defecto: automático)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="comprueba la instalación")
    p.set_defaults(fn=_doctor)

    p = sub.add_parser("sample", help="descarga un clip de prueba (6 s)")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_sample)

    p = sub.add_parser("track", help="procesa un vídeo")
    p.add_argument("video")
    p.add_argument("-o", "--out", default="salida")
    p.add_argument("--start", type=float, default=0.0, help="segundo de inicio")
    p.add_argument("--max-seconds", type=float, default=None, help="procesar solo N segundos")
    p.add_argument("--stride", type=int, default=None,
                   help="analizar 1 de cada N fotogramas (por defecto automático: ~25 por segundo)")
    p.add_argument("--no-jersey", action="store_true", help="no leer dorsales (un poco más rápido)")
    p.add_argument("--keyframe-every", type=int, default=5)
    p.add_argument("--model", default="yolo11m.pt", help="yolo11s.pt = más rápido, yolo11x.pt = más preciso")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--home-name", default="Team A")
    p.add_argument("--away-name", default="Team B")
    p.add_argument("--period", type=int, default=1)
    p.add_argument("--time-offset", type=float, default=0.0, help="reloj del partido (s) en el primer fotograma")
    p.add_argument("--no-extrapolate", action="store_true", help="solo posiciones vistas por la cámara")
    p.add_argument("--no-render", action="store_true", help="no generar el vídeo de verificación")
    p.add_argument("--redo", action="store_true", help="ignorar el análisis guardado y repetirlo")
    p.set_defaults(fn=_track)

    p = sub.add_parser("calibrate", help="calibra una imagen suelta")
    p.add_argument("image")
    p.add_argument("-o", "--out", default=None, help="guardar la imagen con el campo dibujado")
    p.set_defaults(fn=_calibrate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
