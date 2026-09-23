"""Generates the notebooks in notebooks/ (edit here, then run: python tools/make_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
REPO = "https://github.com/delioguzmang-maker/tracking"
BRANCH = "claude/soccernet-calibration-pkg-r8517j"

SETUP = f'''# 1) INSTALACIÓN — en Colab tarda ~2 min; en tu Mac (ya instalado) no hace nada.
import sys, subprocess, os
EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    if not os.path.exists("tracking"):
        # rama con el código; si ya está fusionada en main, git usa la rama por defecto
        r = subprocess.run(["git", "clone", "-q", "-b", "{BRANCH}", "{REPO}"])
        if r.returncode != 0:
            subprocess.run(["git", "clone", "-q", "{REPO}"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "tracking[notebooks]"], check=True)
    sys.path.insert(0, os.path.abspath("tracking"))
import soccercal
print("soccercal", soccercal.__version__, "listo")'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}, "accelerator": "GPU",
                      "colab": {"provenance": [], "gpuType": "T4"}}
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / name)
    print("escrito", OUT / name)


# ---------------------------------------------------------------------------------------
write("00_empezar_aqui.ipynb", [
    md("""
# 00 · Empieza aquí: de vídeo de retransmisión a datos tipo SkillCorner

Este cuaderno procesa un clip de prueba de 6 segundos (Chelsea–Barcelona 2009) y produce
exactamente lo que entrega SkillCorner: la posición en metros de cada jugador y del balón
10 veces por segundo, con identidades, equipos y el área que ve la cámara.

**En Google Colab:** menú *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*,
y luego *Entorno de ejecución → Ejecutar todas*.
**En tu Mac (Jupyter / Anaconda):** instala primero siguiendo el README y ejecuta las celdas en orden
con *Shift + Enter*.
"""),
    code(SETUP),
    md("""
## 2) Comprobar que todo funciona
`doctor` revisa Python, PyTorch, el acelerador (GPU en Colab, **MPS** en un MacBook con chip M),
descarga los pesos de las redes (una sola vez, ~300 MB) y hace una inferencia de prueba.
Si algo sale `[FALLO]`, el mensaje dice qué falta.
"""),
    code('''
from soccercal.cli import main as soccercal_cli
soccercal_cli(["doctor"])
'''),
    md("""
## 3) Procesar el clip de prueba
La primera pasada ejecuta las redes neuronales (lo lento: ~15 s en Colab GPU, ~40 s en un Mac M1/M2,
varios minutos en CPU). Queda guardada en `salida_demo/analysis.pkl.gz`: si vuelves a ejecutar la
celda, se reutiliza y todo lo demás tarda segundos.
"""),
    code('''
import soccercal
from soccercal import Config

video = soccercal.get_sample_video()          # descarga el clip de 6 s
cfg = Config(home_name="Chelsea", away_name="Barcelona")
res = soccercal.run(video, "salida_demo", cfg)   # analiza + calibra + sigue + exporta + vídeo
res.summary()
'''),
    md("""
## 4) Verificar con tus ojos (lo más importante)
Arriba: la retransmisión con **el campo dibujado en magenta a partir de la calibración** —si las
líneas magenta caen sobre las líneas reales, la calibración es buena— y cada persona con su
identidad. Abajo: el minimapa estilo SkillCorner (relleno = visto por la cámara, hueco = extrapolado;
zona gris = lo que ve la cámara).
"""),
    code('''
import cv2, matplotlib.pyplot as plt
cap = cv2.VideoCapture("salida_demo/verificacion.mp4")
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
for ax, n in zip(axes, (10, 140)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, fr = cap.read()
    ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); ax.axis("off"); ax.set_title(f"fotograma {n}")
plt.tight_layout(); plt.show()
'''),
    code('''
# Ver el vídeo completo dentro del cuaderno (en Colab o Jupyter)
from base64 import b64encode
from IPython.display import HTML
import subprocess, shutil
src = "salida_demo/verificacion.mp4"
# mp4v no se reproduce en todos los navegadores: se convierte a H.264 si hay ffmpeg
if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "salida_demo/verificacion_h264.mp4"])
    src = "salida_demo/verificacion_h264.mp4"
HTML(f'<video width=800 controls src="data:video/mp4;base64,{b64encode(open(src, "rb").read()).decode()}"></video>')
'''),
    md("""
## 5) Los datos
`res.table` tiene una fila por jugador y fotograma (10 fps): posición `x, y` en metros
(origen en el centro, `x` a lo largo del campo, `y` hacia la banda lejana, igual que SkillCorner),
`is_detected` (visto por la cámara o extrapolado) y velocidad.
"""),
    code('''
res.table.head(10)
'''),
    code('''
from soccercal.viz import plot_frame
plot_frame(res, k=30);
'''),
    md("## 6) Métricas físicas (distancias por bandas de velocidad, velocidad máxima, sprints)"),
    code('''
res.physical.round(1)
'''),
    md("""
## 7) Formato SkillCorner
En `salida_demo/` están `1_match.json` y `1_tracking_extrapolated.jsonl`, el mismo formato que los
datos abiertos de SkillCorner. Se pueden abrir con su visor (`SkillCorner_Tracking_Viewer.html` del
repositorio `SkillCorner/opendata`) o con **kloppy**, la librería estándar de datos de fútbol:
"""),
    code('''
from soccercal.skillcorner import load_kloppy
ds = load_kloppy("salida_demo")
print(ds.metadata.provider, ds.metadata.frame_rate, "fps,", len(ds.records), "fotogramas")
ds.to_df().head()
'''),
    code('''
import os
print("\\n".join(sorted(os.listdir("salida_demo"))))
if EN_COLAB:   # descargar todo en un zip
    import shutil
    from google.colab import files
    shutil.make_archive("salida_demo", "zip", "salida_demo")
    files.download("salida_demo.zip")
'''),
])

# ---------------------------------------------------------------------------------------
write("01_tu_partido.ipynb", [
    md("""
# 01 · Tu propio partido

1. Consigue el vídeo en `.mp4` (retransmisión con la **cámara principal**, la que sigue el juego desde
   la grada lateral; 720p o 1080p).
2. Ejecuta la instalación, sube el vídeo y ajusta los parámetros.
3. **Empieza siempre con un tramo corto** (`max_seconds=60`) y revisa el vídeo de verificación antes de
   procesar el partido entero.

Tiempos orientativos por minuto de vídeo a 25 fps: Colab GPU T4 ≈ 1 min, Mac M1/M2 ≈ 2–3 min,
CPU ≈ 30 min. Para un partido completo usa `stride=2` (analiza 12,5 fotogramas por segundo; la salida
sigue siendo 10 fps).
"""),
    code(SETUP),
    md("## Elegir el vídeo"),
    code('''
VIDEO = "mi_partido.mp4"     # en tu Mac: ruta completa, p. ej. "/Users/tu_nombre/Downloads/partido.mp4"

if EN_COLAB and not os.path.exists(VIDEO):
    # Opción A: subir desde el ordenador (lento para archivos grandes)
    from google.colab import files
    subido = files.upload()
    VIDEO = next(iter(subido))
    # Opción B (recomendada para partidos): Google Drive
    # from google.colab import drive; drive.mount("/content/drive")
    # VIDEO = "/content/drive/MyDrive/partido.mp4"
print(VIDEO)
'''),
    md("""
## Parámetros
| parámetro | qué hace |
|---|---|
| `start_s`, `max_seconds` | tramo a procesar (segundos) |
| `stride` | 1 = todos los fotogramas; 2 = la mitad (el doble de rápido) |
| `det_model` | `yolo11m.pt` (equilibrado), `yolo11s.pt` (rápido), `yolo11x.pt` (preciso) |
| `home_name`, `away_name` | nombres de los equipos (el "local" es el primer color detectado; revisa `summary.json`) |
| `period`, `time_offset_s` | parte del partido y minuto de reloj del primer fotograma |
| `extrapolate` | rellenar jugadores fuera de cámara (`is_detected=False`), como SkillCorner |
"""),
    code('''
import soccercal
from soccercal import Config
cfg = Config(
    start_s=0, max_seconds=60,      # <- primero 60 s; luego None para todo
    stride=1,
    det_model="yolo11m.pt",
    home_name="Local", away_name="Visitante",
    period=1, time_offset_s=0,
)
res = soccercal.run(VIDEO, "salida_partido", cfg)
res.summary()
'''),
    md("""
## Revisa la calidad
* `pct_frames_with_camera`: fotogramas con calibración válida. Los primeros planos, repeticiones y
  cámaras detrás de la portería se descartan a propósito (SkillCorner hace lo mismo).
* `median_line_alignment`: 0,6–0,8 es bueno; < 0,4 indica un vídeo difícil (líneas poco visibles).
* Mira `salida_partido/verificacion.mp4`: las líneas magenta deben caer sobre las líneas del campo.
"""),
    code('''
import pandas as pd
cams = pd.read_csv("salida_partido/cameras.csv")
ax = cams.plot(x="t", y="line_alignment", style=".", figsize=(12, 3), title="calidad de calibración (keyframes)")
cams["valid"].mean()
'''),
    md("""
## Si los equipos salen cambiados
El "equipo A" es el primer grupo de color. Para cambiar nombres basta con volver a ejecutar solo la
parte rápida (sin repetir las redes neuronales):
"""),
    code('''
from soccercal.pipeline import build
from soccercal.analysis import Analysis
an = Analysis.load("salida_partido/analysis.pkl.gz")
cfg.home_name, cfg.away_name = cfg.away_name, cfg.home_name
res = build(an, cfg)
res.save("salida_partido")
res.summary()
'''),
])

# ---------------------------------------------------------------------------------------
write("02_analisis.ipynb", [
    md("""
# 02 · Analizar los datos

Trabaja sobre la carpeta de salida de `00` o `01` (por defecto `salida_demo`). Todo lo de aquí sirve
igual para datos reales de SkillCorner, porque el formato es el mismo.
"""),
    code(SETUP),
    code('''
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from soccercal import pitch
CARPETA = "salida_demo"
if not os.path.exists(f"{CARPETA}/tracking.csv"):
    import soccercal
    soccercal.run(soccercal.get_sample_video(), CARPETA, render=False)
df = pd.read_csv(f"{CARPETA}/tracking.csv")
df.head()
'''),
    md("## Dibujar el campo"),
    code('''
def campo(ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_facecolor("#3a7d44")
    for pl in pitch.polylines():
        if np.abs(pl[:, 2]).max() == 0:
            ax.plot(pl[:, 0], pl[:, 1], color="white", lw=1)
    ax.set_xlim(-57, 57); ax.set_ylim(-38, 38); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return ax
'''),
    md("## Mapa de calor de un jugador (solo posiciones vistas por la cámara)"),
    code('''
jug = df[df.role != "referee"].groupby("player_id").size().idxmax()   # el jugador con más datos
d = df[(df.player_id == jug) & df.is_detected]
ax = campo()
ax.hexbin(d.x, d.y, gridsize=25, extent=(-52.5, 52.5, -34, 34), cmap="hot", alpha=0.7, mincnt=1)
ax.set_title(f"jugador {jug}: {len(d)} posiciones");
'''),
    md("## Velocidad a lo largo del tiempo"),
    code('''
fig, ax = plt.subplots(figsize=(12, 3))
for pid, g in df[df.role == "player"].groupby("player_id"):
    ax.plot(g.timestamp, g.speed_kmh, lw=0.8)
ax.axhline(20, ls="--", c="k", lw=0.8); ax.axhline(25, ls="--", c="r", lw=0.8)
ax.set_xlabel("s"); ax.set_ylabel("km/h"); ax.set_title("velocidad (líneas: 20 km/h HSR, 25 km/h sprint)");
'''),
    md("## Forma del equipo: centroide, anchura y profundidad (jugadores vistos)"),
    code('''
vis = df[df.is_detected & (df.role == "player")]
forma = vis.groupby(["frame", "team"]).agg(cx=("x", "mean"), ancho=("y", lambda s: s.max() - s.min()),
                                            largo=("x", lambda s: s.max() - s.min()), n=("x", "size")).reset_index()
forma = forma[forma.n >= 4]
fig, axs = plt.subplots(1, 2, figsize=(12, 3))
for t, g in forma.groupby("team"):
    axs[0].plot(g.frame / 10, g.ancho, label=f"equipo {t}"); axs[1].plot(g.frame / 10, g.largo, label=f"equipo {t}")
axs[0].set_title("anchura (m)"); axs[1].set_title("profundidad (m)"); axs[0].legend();
'''),
    md("## Resumen físico"),
    code('''
pd.read_csv(f"{CARPETA}/physical.csv").round(1)
'''),
])
