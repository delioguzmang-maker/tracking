"""Generates the notebooks in notebooks/ (edit here, then run: python tools/make_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
REPO = "https://github.com/delioguzmang-maker/tracking"
BRANCH = "claude/soccernet-calibration-pkg-r8517j"

SETUP = f'''# 1) INSTALACIÓN — en Colab tarda ~2 min; en tu Mac (ya instalado) no hace nada.
# Descarga SIEMPRE la última versión del código (volver a ejecutar esta celda actualiza).
import sys, subprocess, os
EN_COLAB = "google.colab" in sys.modules

def _pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\\n")
        raise RuntimeError("La instalación falló: copia el mensaje de arriba y envíamelo.")

if EN_COLAB:
    RAMA = "{BRANCH}"
    if not os.path.exists("tracking"):
        r = subprocess.run(["git", "clone", "-q", "-b", RAMA, "{REPO}"])
        if r.returncode != 0:  # la rama ya se fusionó: rama por defecto
            subprocess.run(["git", "clone", "-q", "{REPO}"], check=True)
    else:  # ya estaba descargado: traer lo último (si no, seguirías usando la versión vieja)
        subprocess.run(["git", "-C", "tracking", "fetch", "-q", "origin"], check=True)
        ref = "origin/" + RAMA
        if subprocess.run(["git", "-C", "tracking", "rev-parse", "-q", "--verify", ref], capture_output=True).returncode:
            ref = "origin/HEAD"
        subprocess.run(["git", "-C", "tracking", "reset", "-q", "--hard", ref], check=True)
    _pip("-e", "tracking[colab]")
    # lector de dorsales: su ficha dice "Python < 3.13" pero funciona en 3.13 (el Python de Colab)
    _pip("--ignore-requires-python", "rapidocr_onnxruntime==1.4.4")
    sys.path.insert(0, os.path.abspath("tracking"))
for _m in [m for m in sys.modules if m == "soccercal" or m.startswith("soccercal.")]:
    del sys.modules[_m]  # sin reiniciar el entorno: se carga el código recién descargado
import soccercal
print("Python", sys.version.split()[0], "| soccercal", soccercal.version(), "listo")'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    for i, c in enumerate(cells):  # stable ids: regenerating does not touch unchanged notebooks
        c["id"] = f"{Path(name).stem[:2]}-{i:03d}"
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
## 3) Procesar el clip de prueba (o tu vídeo)
Para tu propio vídeo de Google Drive pon su nombre en `VIDEO_EN_DRIVE`, por ejemplo
`VIDEO_EN_DRIVE = "trim psg bayern.mp4"` (se busca en todas las carpetas de *Mi unidad*).

La primera pasada ejecuta las redes neuronales (lo lento: ~15 s en Colab GPU, ~40 s en un Mac M1/M2,
varios minutos en CPU). Queda guardada en `salida_demo/analysis.pkl.gz`: si vuelves a ejecutar la
celda, se reutiliza y todo lo demás tarda segundos.
"""),
    code('''
import soccercal
from soccercal import Config

VIDEO_EN_DRIVE = None    # o el nombre de un vídeo de tu Google Drive, p. ej. "trim psg bayern.mp4"
SUBIR_MI_VIDEO = False   # True = en Colab te pide subir tu propio vídeo desde el ordenador

if VIDEO_EN_DRIVE and EN_COLAB:
    video = soccercal.video_from_drive(VIDEO_EN_DRIVE)  # pide permiso para abrir tu Drive la 1ª vez
    cfg = Config(home_name="Local", away_name="Visitante")
elif SUBIR_MI_VIDEO and EN_COLAB:
    from google.colab import files
    video = next(iter(files.upload()))
    cfg = Config(home_name="Local", away_name="Visitante")
else:
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
# Ver el vídeo de verificación dentro del cuaderno (en Colab o Jupyter)
from base64 import b64encode
from IPython.display import HTML
import subprocess, shutil, os
src = "salida_demo/verificacion.mp4"
# mp4v no se reproduce en todos los navegadores: se convierte a H.264 si hay ffmpeg. Un vídeo muy
# largo no cabe dentro del cuaderno: se muestra el primer minuto y el completo va en el zip del final.
if shutil.which("ffmpeg"):
    largo = os.path.getsize(src) > 50e6
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src] + (["-t", "60"] if largo else []) +
                   ["-vcodec", "libx264", "-pix_fmt", "yuv420p", "salida_demo/verificacion_h264.mp4"])
    if largo:
        print("Vídeo largo: aquí se ve el primer minuto; el completo está en salida_demo/verificacion.mp4")
    src = "salida_demo/verificacion_h264.mp4"
if os.path.getsize(src) < 80e6:
    display(HTML(f'<video width=800 controls src="data:video/mp4;base64,{b64encode(open(src, "rb").read()).decode()}"></video>'))
else:
    print("El vídeo es demasiado grande para verlo aquí: descárgalo con el zip del final.")
'''),
    md("""
## 5) Los datos
`res.table` tiene una fila por jugador y fotograma (10 fps): posición `x, y` en metros
(origen en el centro, `x` a lo largo del campo, `y` hacia la banda lejana, igual que SkillCorner),
`is_detected` (visto por la cámara o extrapolado) y velocidad (`speed_kmh`, y `vx, vy` en m/s).
El cuaderno `02_analisis` calcula con estos datos el control del espacio (Spearman / Laurie Shaw,
Fernández y Bornn) y el valor del espacio (EPV).
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
## 7) Control del espacio
Para cada punto del campo, la probabilidad de que el equipo con el balón lo gane si el balón se juega
ahí (modelo de Spearman con el código de Laurie Shaw). Color de un equipo = lo ganaría ese equipo,
blanco = 50/50, flechas = dónde estará cada jugador en 1 s, gris = zona que la cámara no ve (posiciones
estimadas). Más modelos y el vídeo en el cuaderno `02_analisis`.
"""),
    code('''
from soccercal.tactics import Match, plot_control
m = Match.from_result(res)
SEGUNDO = 5.6                     # cambia el momento
plot_control(m, int(round(SEGUNDO * m.fps)), model="spearman");
'''),
    md("""
## 8) Formato SkillCorner
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
CPU ≈ 30 min. Los vídeos a 50/60 fps se analizan automáticamente a ~25 fotogramas por segundo; para ir
aún más rápido en un partido completo usa `stride` = el doble de lo automático (la salida sigue a 10 fps).

**Qué se descarta a propósito** (como hace SkillCorner): primeros planos de jugadores o entrenadores,
planos del público o del banquillo, repeticiones desde otras cámaras. Ahí no se generan posiciones y el
vídeo de verificación lo indica con un aviso. Los entrenadores, jueces de línea y suplentes que aparecen
fuera de las líneas **no** se cuentan como jugadores.
"""),
    code(SETUP),
    md("## Elegir el vídeo"),
    code('''
VIDEO = "mi_partido.mp4"     # en tu Mac: ruta completa, p. ej. "/Users/tu_nombre/Downloads/partido.mp4"
VIDEO_EN_DRIVE = None        # en Colab (recomendado): nombre del vídeo en tu Google Drive, p. ej. "trim psg bayern.mp4"

if EN_COLAB and VIDEO_EN_DRIVE:
    VIDEO = str(soccercal.video_from_drive(VIDEO_EN_DRIVE))  # busca en todo "Mi unidad" y lo copia aquí
elif EN_COLAB and not os.path.exists(VIDEO):
    # sin Drive: subir desde el ordenador (lento para archivos grandes)
    from google.colab import files
    subido = files.upload()
    VIDEO = next(iter(subido))
print(VIDEO)
'''),
    md("""
## Parámetros
| parámetro | qué hace |
|---|---|
| `start_s`, `max_seconds` | tramo a procesar (segundos) |
| `stride` | `None` = automático (~25 fotogramas analizados por segundo); 2 = la mitad de eso |
| `jersey_ocr` | leer dorsales (la etiqueta pasa de `id12` a `#17` cuando se lee con seguridad) |
| `jersey_views` | vistas de cada jugador que se leen en la segunda mirada (80; menos = más rápido) |
| `ball_refine` | segunda búsqueda del balón en alta resolución donde no se encontró (`True`) |
| `players_per_team` | máximo de identidades por equipo (11 = 10 de campo + portero; `None` = sin límite) |
| `home_numbers`, `away_numbers` | **opcional pero muy útil**: los dorsales de cada equipo (titulares y suplentes), p. ej. `[1, 2, 4, 6, 9, 10, 17, 27]`. Se descartan lecturas imposibles y se sabe qué color es el local |
| `det_model` | `yolo11m.pt` (equilibrado), `yolo11s.pt` (rápido), `yolo11x.pt` (preciso) |
| `home_name`, `away_name` | nombres de los equipos (sin `home_numbers`/`away_numbers`, el "local" es el primer color detectado; revisa el vídeo) |
| `period`, `time_offset_s` | parte del partido y minuto de reloj del primer fotograma |
| `extrapolate` | rellenar jugadores fuera de cámara (`is_detected=False`), como SkillCorner |
"""),
    code('''
import soccercal
from soccercal import Config
cfg = Config(
    start_s=0, max_seconds=60,      # <- primero 60 s; luego None para todo
    stride=None,                    # automático
    det_model="yolo11m.pt",
    jersey_ocr=True,                # leer dorsales
    home_name="Local", away_name="Visitante",
    home_numbers=None,              # opcional: [1, 2, 4, 6, 9, ...] dorsales del local (titulares + suplentes)
    away_numbers=None,              # opcional: dorsales del visitante
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
CARPETA = "salida_demo"   # la carpeta de salida de 00 o 01
if not os.path.exists(f"{CARPETA}/tracking.csv") and EN_COLAB:
    # En Colab cada cuaderno tiene su propia máquina: sube el zip que descargaste al final del 00 / 01.
    print("Sube el .zip de resultados (por ejemplo salida_demo.zip). Si cancelas, se usa el clip de ejemplo.")
    import zipfile
    from google.colab import files
    try:
        subidos = files.upload()
    except Exception:
        subidos = {}
    for nombre in subidos:
        if nombre.endswith(".zip"):
            zipfile.ZipFile(nombre).extractall(CARPETA)
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
jug = df[df.role.isin(["player", "goalkeeper"])].groupby("player_id").size().idxmax()   # el jugador con más datos
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
    md("""
---
# Análisis táctico: quién controla el espacio y cuánto vale

Con las posiciones y velocidades de los 22 jugadores se pueden calcular los modelos clásicos de la
analítica de fútbol. Funcionan igual con los archivos de SkillCorner (`*_match.json` +
`*_tracking_extrapolated.jsonl`), así que también sirven con los datos abiertos de SkillCorner.

**Una advertencia importante con vídeo de televisión:** la cámara solo ve una parte del campo. Los
jugadores que no se ven en ese momento tienen una posición *estimada* (círculo con el centro blanco), y
un jugador que no aparece nunca en el clip (a menudo un portero) falta. Por eso en los mapas la zona que
la cámara no ve está en **gris**: ahí el resultado es orientativo.
"""),
    md("""
## 1) Control del espacio (William Spearman 2018, código de Laurie Shaw)

Para cada punto del campo: **probabilidad de que el equipo con el balón lo gane si el balón se juega
ahí**. Cada jugador sigue corriendo hacia donde iba durante 0,7 s (su tiempo de reacción) y luego va
en línea recta a 5 m/s; el balón viaja a 15 m/s; el portero atrapa el balón 3 veces más rápido.
Es el modelo de *Friends of Tracking* (repositorio LaurieOnTracking) con sus mismos parámetros y
ecuaciones: da **el mismo número** que el código original (lo comprobamos abajo), pero unas 30 veces
más rápido.

Cómo leerlo: color de un equipo = ese equipo ganaría el balón; blanco = 50/50; flechas = dónde estará
cada jugador dentro de 1 segundo; anillo negro = quien lleva el balón. Cambia `SEGUNDO` para ver otro
momento.
"""),
    code('''
from soccercal.tactics import Match, plot_control
m = Match.load(CARPETA)
SEGUNDO = 5.6                     # el momento que quieres ver
k = int(round(SEGUNDO * m.fps))   # fotograma (10 por segundo)
plot_control(m, k, model="spearman");
'''),
    md("""
## 2) El espacio según Fernández y Bornn (FC Barcelona, "Wide Open Spaces", 2018)

Otra forma de medirlo: cada jugador tiene una "zona de influencia" en forma de óvalo, centrada medio
segundo por delante de él y estirada en la dirección en la que corre. Cuanto más lejos está del balón,
más grande es (4 m al lado del balón, 10 m a partir de unos 18 m), porque tiene más tiempo para llegar.
El control de un equipo combina las influencias de sus jugadores menos las del rival.
"""),
    code('''
plot_control(m, k, model="fernandez");
'''),
    md("""
## 3) Dónde está el espacio que vale (control × EPV)

No todo el espacio vale lo mismo: controlar el balón cerca de la portería rival vale mucho más que en
tu propio campo. El **EPV** (*Expected Possession Value*) es la probabilidad de que la jugada acabe en
gol desde cada zona (la superficie que publicó Laurie Shaw en *Friends of Tracking*). Multiplicado por el
control del espacio indica **dónde un pase sería a la vez seguro y peligroso** para el equipo con el balón.
"""),
    code('''
plot_control(m, k, model="value");
'''),
    md("""
## 4) Territorio a lo largo del tiempo (% del campo que controla cada equipo)

La línea discontinua (solo la parte que ve la cámara) es la más fiable: fuera de cámara las posiciones
son estimadas y pueden faltar jugadores que el vídeo no ha mostrado nunca.
"""),
    code('''
from soccercal.tactics import control_series
cs = control_series(m, every=2)
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(cs.t, 100 * cs.home_share, label=f"{m.names[0]}: todo el campo")
if "home_share_in_view" in cs:
    ax.plot(cs.t, 100 * cs.home_share_in_view, "--", label=f"{m.names[0]}: solo lo que ve la cámara")
ax.axhline(50, c="k", lw=0.6); ax.set_ylim(0, 100); ax.set_xlabel("s"); ax.set_ylabel("%"); ax.legend();
'''),
    md("## 5) Vídeo del control del espacio (un mapa cada 0,1 s)"),
    code('''
from soccercal.tactics import render_control_video
from base64 import b64encode
from IPython.display import HTML
import shutil
ruta = render_control_video(m, f"{CARPETA}/control_espacio.mp4", model="spearman")
if not shutil.which("ffmpeg"):
    print("Sin ffmpeg el vídeo quizá no se vea aquí: ábrelo con VLC (o instala ffmpeg: conda install ffmpeg).")
if os.path.getsize(ruta) < 80e6:
    display(HTML(f'<video width=800 controls src="data:video/mp4;base64,{b64encode(open(ruta, "rb").read()).decode()}"></video>'))
'''),
    md("""
## 6) Usar el código original de Laurie Shaw (LaurieOnTracking)

`to_metrica` convierte los datos al formato de los tutoriales de *Friends of Tracking* (tablas
`tracking_home` y `tracking_away` con columnas `Home_7_x`, `Home_7_vx`, ...), así puedes seguir sus
tutoriales con tu partido. La celda siguiente descarga su archivo `Metrica_PitchControl.py` de GitHub y
comprueba que da el mismo resultado que el nuestro.
"""),
    code('''
from soccercal.tactics import to_metrica, pitch_control, grid
tracking_home, tracking_away, GK_numbers = to_metrica(m)
print("porteros:", GK_numbers)
tracking_home.head()
'''),
    code('''
import urllib.request, importlib, sys
urllib.request.urlretrieve("https://raw.githubusercontent.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/"
                           "master/Metrica_PitchControl.py", "Metrica_PitchControl.py")
sys.path.insert(0, ".")
import Metrica_PitchControl as mpc
s = m.snapshot(k)
if s.attacking < 0:
    print("En este momento no se sabe quién tiene el balón: elige otro SEGUNDO.")
else:
    params = mpc.default_model_params()
    att, dfn = ("Home", "Away") if s.attacking == 0 else ("Away", "Home")
    filas = {"Home": tracking_home.loc[k], "Away": tracking_away.loc[k]}
    gk = {"Home": GK_numbers[0], "Away": GK_numbers[1]}
    A = mpc.initialise_players(filas[att], att, params, gk[att])
    D = mpc.initialise_players(filas[dfn], dfn, params, gk[dfn])
    balon = np.array([tracking_home.loc[k, "ball_x"], tracking_home.loc[k, "ball_y"]])
    xg, yg = grid(50, (m.pitch_length, m.pitch_width))
    shaw = np.array([[mpc.calculate_pitch_control_at_target(np.array([x, y]), A, D, balon, params)[0] for x in xg] for y in yg])
    nuestro = pitch_control(s, offsides=False).values
    print(f"diferencia máxima con el código de Shaw: {np.abs(shaw - nuestro).max():.1e}")
'''),
    md("""
## 7) SoccerMap: ¿saldría bien un pase a cada punto?

*SoccerMap* (Fernández y Bornn, 2020) es una red neuronal que aprende, a partir de muchos pases reales, la
probabilidad de que un pase desde donde está el balón hasta cada punto del campo salga bien. La hemos
entrenado primero con 270.000 pases de StatsBomb 360 (293 partidos) y después con unos 22.000 pases de 29
partidos con datos de seguimiento abiertos (20 de SkillCorner, sacados de la televisión como los nuestros;
7 de la Bundesliga, IDSSE; 2 de Metrica), y medido en partidos que la red no vio al entrenar: acierta más
que la regresión logística del propio artículo (resultados en el README). Color del equipo con el balón = el pase saldría bien.
"""),
    code('''
plot_control(m, k, model="soccermap");
'''),
    md("""
Lo que la red "ve" son 13 imágenes del campo (posiciones y velocidades de los dos equipos, distancias y
ángulos al balón y a la portería...), orientadas para que el equipo con el balón ataque hacia la derecha:
"""),
    code('''
from soccercal.tactics import soccermap_tensor, SOCCERMAP_CHANNELS
X = soccermap_tensor(m.snapshot(k))
fig, axs = plt.subplots(3, 5, figsize=(16, 7))
for ax, ch, nombre in zip(axs.flat, X, SOCCERMAP_CHANNELS):
    ax.imshow(ch, origin="lower", cmap="viridis"); ax.set_title(nombre, fontsize=8); ax.axis("off")
for ax in axs.flat[len(X):]:
    ax.axis("off")
plt.tight_layout()
'''),
    md("""
## 8) Jugadores que el vídeo no enseña nunca

Si la cámara nunca muestra a un jugador (muchas veces un portero), ese equipo tiene menos de 11 y los
mapas regalan esa zona al rival. `complete` los añade en una posición **deducida** (círculo de puntos con
"?"): el portero con una red que aprendió de partidos reales dónde se coloca según el balón y su línea
defensiva; un jugador de campo comparando la forma de sus compañeros con cientos de formaciones reales.
Solo se usan para los mapas: los archivos de tracking no se tocan.
"""),
    code('''
from soccercal.impute import complete
mc = complete(m)
print(mc.table[mc.table.imputed].groupby("player_id").agg(equipo=("team", "first"), portero=("gk", "first")))
fig, axs = plt.subplots(2, 1, figsize=(11, 15))
plot_control(m, k, ax=axs[0]); axs[0].set_title("solo lo que se ha visto")
plot_control(mc, k, ax=axs[1]); axs[1].set_title("con los jugadores deducidos")
plt.tight_layout()
'''),
    md("""
## 9) Valor del campo y ganancia de espacio (Fernández y Bornn, "Wide Open Spaces")

**Valor del campo:** dónde se colocan los defensas según dónde está el balón; donde se juntan, el espacio
vale (aprendido de partidos reales, como en el artículo), y vale más cuanto más cerca de la portería.

**Ganancia de espacio (SOG):** en los 3 segundos después de cada posesión, cuánto mejora cada jugador la
calidad del espacio que controla (control × valor). *Activa* si corre (más de 1,5 m/s), *pasiva* si
camina: el artículo mostró que Messi ganaba espacio sobre todo andando. **Generación de espacio (SGG):** un
jugador arrastra al rival que marcaba a un compañero y le deja libre.
"""),
    code('''
from soccercal.space import pitch_value
from soccercal.tactics import grid, draw_pitch
xg, yg = grid(105)
P = np.stack(np.meshgrid(xg, yg), -1).reshape(-1, 2)
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
for ax, bx in zip(axs, (-30, 0, 30)):
    v = pitch_value([bx, 10], P).reshape(len(yg), len(xg))
    ax.imshow(v, extent=(-52.5, 52.5, -34, 34), origin="lower", cmap="Greens", vmin=0, vmax=1)
    draw_pitch(ax); ax.scatter([bx], [10], c="#ffd400", edgecolors="k", zorder=5)
    ax.set_title(f"balón en ({bx}, 10), atacando hacia la derecha", fontsize=9)
'''),
    code('''
from soccercal import space
equipo = m.snapshot(k).attacking if m.snapshot(k).attacking >= 0 else 0
print("equipo:", m.names[equipo])
space.summary(mc, equipo).round(4)
'''),
])
