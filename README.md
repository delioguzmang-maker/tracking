# soccercal — datos de tracking tipo SkillCorner a partir de un vídeo de retransmisión

Metes un vídeo de un partido (la cámara de televisión normal) y obtienes lo mismo que vende
SkillCorner: **la posición en metros de cada jugador y del balón 10 veces por segundo**, con
identidades, equipos, portero, el área que ve la cámara, jugadores fuera de plano extrapolados y
métricas físicas. Los archivos salen **en el formato exacto de SkillCorner**: se abren con su visor
oficial y con [kloppy](https://kloppy.pysport.org).

Funciona en **Google Colab** (gratis, con GPU), en **Jupyter** y en un **MacBook con Anaconda**
(usa la GPU del chip M mediante MPS).

Parte de la calibración del ganador de SoccerNet 2023
([sportlight baseline](https://github.com/NikolasEnt/soccernet-calibration-sportlight/tree/master/baseline)),
cuyos pesos nunca se publicaron: aquí se usan los pesos públicos de NBJW/PnLCalib (misma familia
HRNet, mismo vocabulario de 57 puntos del campo) sobre una implementación propia, y se añade todo lo
que hace falta para llegar a datos de tracking.

---

## Qué obtienes

`soccercal track partido.mp4 -o salida` crea en `salida/`:

| archivo | contenido |
|---|---|
| `1_tracking_extrapolated.jsonl` | **formato SkillCorner**: un fotograma cada 0,1 s con `player_data` (x, y, player_id, is_detected), `ball_data`, `possession`, `image_corners_projection` |
| `1_match.json` | **formato SkillCorner**: equipos, color de camiseta estimado, plantilla (identidades), portero, tamaño del campo |
| `tracking.csv` | lo mismo en tabla: `frame, timestamp, player_id, shirt_number, team, role, x, y, is_detected, speed_kmh` |
| `physical.csv` | por jugador: distancia total y por bandas (andar, trotar, correr, HSR 20–25 km/h, sprint >25), velocidad máxima, PSV-99, nº de esfuerzos |
| `cameras.csv` | la cámara en cada fotograma: pan, inclinación, zoom, posición, calidad |
| `verificacion.mp4` | **el control de calidad**: el campo dibujado sobre la imagen + cada jugador con su identidad + minimapa |
| `summary.json` | resumen: % de fotogramas calibrados, identidades por equipo, calidad media |
| `analysis.pkl.gz` | la parte lenta ya calculada; si repites el comando se reutiliza (borra el archivo para rehacerla) |

Coordenadas como SkillCorner: metros, origen en el centro del campo, `x` a lo largo (positivo hacia la
derecha vista desde la cámara principal), `y` a lo ancho (positivo hacia la banda lejana).

### Cómo leer el vídeo de verificación

| lo que ves | significa |
|---|---|
| líneas magenta | el campo dibujado con la cámara calculada: deben caer encima de las líneas reales |
| `#17` | dorsal **leído en la camiseta** (OCR, votado sobre muchas lecturas) |
| `id12` | jugador cuyo dorsal todavía no se ha podido leer: es un número interno de seguimiento, **no** el dorsal |
| `GK`, `ARB`, `JL` | portero, árbitro, juez de línea (árbitros y jueces de línea no salen en los datos de jugadores) |
| sin caja | entrenadores, suplentes, recogepelotas y público: se detectan pero **no** son jugadores |
| círculo amarillo | balón detectado |
| aviso "PLANO DESCARTADO" | primer plano de un jugador/entrenador, público, banquillo, gráficos: ese tramo **no genera datos** |

Reglas (como SkillCorner): solo se usan planos abiertos de la cámara principal; un plano con una persona
que ocupa más del 40 % de la altura de la imagen, o con poco césped, se descarta. Una persona que pasa la
mayor parte del tiempo fuera de las líneas (banda, banquillo) es *staff*, nunca jugador, y el seguimiento
no puede unirla con un jugador.

![vídeo de verificación](docs/verificacion_ejemplo.jpg)

*Arriba: campo dibujado a partir de la calibración (magenta) e identidades. Abajo: minimapa
(relleno = visto, hueco = extrapolado, gris = lo que ve la cámara).*

![la salida abierta en el visor oficial de SkillCorner](docs/visor_skillcorner.jpg)

*La misma salida abierta en el visor oficial de SkillCorner (`SkillCorner/opendata/viz_tools`).*

---

## Opción 1 — Google Colab (lo más fácil, no instalas nada)

1. Abre <https://colab.research.google.com> → *Archivo → Abrir cuaderno → GitHub* y pega
   `https://github.com/delioguzmang-maker/tracking`. Elige `notebooks/00_empezar_aqui.ipynb`
   (si no aparece, cambia la rama a `claude/soccernet-calibration-pkg-r8517j`).
2. Menú *Entorno de ejecución → Cambiar tipo de entorno de ejecución → **GPU T4*** → Guardar.
3. *Entorno de ejecución → **Ejecutar todas***. La primera vez tarda unos 3 minutos (instala y descarga
   los modelos). Al final verás la verificación, las tablas y un zip para descargar.
4. Para tu vídeo: `notebooks/01_tu_partido.ipynb` (subida directa o desde Google Drive).

---

## Opción 2 — MacBook Pro con Anaconda, paso a paso

Todo se escribe en la app **Terminal** (Aplicaciones → Utilidades → Terminal). Copia cada bloque,
pégalo y pulsa Enter. Espera a que termine antes de pegar el siguiente.

**Paso 1. Instalar Miniconda** (si ya tienes Anaconda, salta al paso 2).
Descarga el instalador *macOS Apple Silicon (arm64) pkg* de
<https://www.anaconda.com/download/success> (sección Miniconda), ábrelo y sigue el asistente.
Cierra la Terminal y ábrela de nuevo. Debe aparecer `(base)` al principio de la línea.

**Paso 2. Descargar el código**

```bash
cd ~/Documents
git clone -b claude/soccernet-calibration-pkg-r8517j https://github.com/delioguzmang-maker/tracking.git
cd tracking
```

(Si `git` pide instalar las *herramientas de línea de comandos*, acepta y repite el comando.
Alternativa sin git: en GitHub, botón verde *Code → Download ZIP*, descomprímelo en Documentos y
haz `cd ~/Documents/tracking-*`.)

**Paso 3. Crear el entorno e instalar** (unos 5 minutos)

```bash
conda create -n soccercal python=3.11 -y
conda activate soccercal
pip install torch torchvision
pip install -e ".[notebooks,dev]"
```

**Paso 4. Comprobar la instalación**

```bash
soccercal doctor
```

Debe terminar en `Todo listo.` y mostrar `dispositivo = mps` (la GPU del Mac). La primera vez descarga
los pesos (~300 MB). Si algún paso dice `[FALLO]`, mira *Problemas frecuentes* abajo.

**Paso 5. Probar con el clip de ejemplo (6 segundos)**

```bash
soccercal track "$(soccercal sample)" -o salida_demo
open salida_demo/verificacion.mp4
```

Mira el vídeo: **las líneas magenta deben caer encima de las líneas del campo** y cada jugador debe
llevar un número que no cambia. Esa es la prueba de que todo funciona.

**Paso 6. Tu partido.** Empieza SIEMPRE por un tramo corto para ver si tu vídeo va bien:

```bash
soccercal track ~/Downloads/partido.mp4 -o salida_partido --max-seconds 60 --home-name "Local" --away-name "Visitante"
open salida_partido/verificacion.mp4
```

Si se ve bien, el partido entero (el doble de rápido con `--stride 2`, la salida sigue a 10 fps):

```bash
soccercal track ~/Downloads/partido.mp4 -o salida_partido --stride 2 --redo
```

**Paso 7. Jupyter (opcional).** `jupyter lab notebooks/` abre el navegador con los cuadernos
`00_empezar_aqui`, `01_tu_partido` y `02_analisis`.

**Cada vez que abras una Terminal nueva** tienes que hacer `conda activate soccercal` (y `cd` a la
carpeta si vas a usar rutas relativas). Es el error más común.

Desde Python:

```python
import soccercal
res = soccercal.run("partido.mp4", "salida", soccercal.Config(max_seconds=60))
res.table.head()          # posiciones a 10 fps
res.physical              # métricas físicas
```

### Tiempos orientativos

Medido aquí: CPU de 4 núcleos ≈ 1,2 s por fotograma (el clip de 6 s tarda ~3 min). En GPU el coste lo
dominan las dos redes (YOLO11m a 1280 px en cada fotograma analizado; la red del campo 1 de cada 5):
en una T4 de Colab o un Mac M1/M2 Pro espera del orden de 10 fotogramas por segundo, es decir
**unos 1–3 minutos por minuto de vídeo** a 25 fps, la mitad con `--stride 2`. Estas cifras de GPU
son estimaciones, no medidas en este repositorio. Más rápido: `--model yolo11s.pt`.

---

## Problemas frecuentes

| síntoma | solución |
|---|---|
| `command not found: soccercal` | no está activado el entorno: `conda activate soccercal` |
| `CERTIFICATE_VERIFY_FAILED` al descargar | ya se usa `certifi`; si persiste: `pip install --upgrade certifi` |
| `doctor` dice `dispositivo = cpu` en un Mac M | `pip install --upgrade torch torchvision` dentro del entorno |
| error de memoria (MPS / CUDA out of memory) | `--batch 2` |
| `pct_frames_with_camera` bajo | el vídeo tiene muchos primeros planos/repeticiones (se descartan a propósito) o no es la cámara principal |
| los equipos salen al revés | `--home-name` / `--away-name` en el orden contrario (el "local" es el primer grupo de color) |
| no se reproduce `verificacion.mp4` en el navegador | ábrelo con QuickTime o VLC (códec mp4v) |
| quieres rehacer el análisis con otro modelo | añade `--redo` |
| el balón se pierde a menudo | ver "El balón" más abajo: con un detector entrenado en fútbol mejora mucho |
| salen etiquetas `id12` en vez de dorsales | normal cuando el dorsal no se ve (jugador de frente o lejos); con más minutos de vídeo se leen más |

---

## Cómo funciona

```
vídeo ─► [pasada 1: redes, se guarda]  YOLO11 (personas + balón) en cada fotograma
                                       HRNet 57 puntos del campo cada 5 fotogramas
                                       registro del fondo entre fotogramas (flujo óptico)
                                       píxeles de las líneas blancas, color de camisetas
      ─► [pasada 2: segundos]          cámara en cada fotograma ─► jugadores en metros ─► seguimiento
                                       en el campo ─► re-identificación ─► equipos/portero ─► 10 fps
                                       + extrapolación fuera de cámara ─► métricas ─► formato SkillCorner
```

**Calibración (la base de todo).** Puntos clave del campo → homografía RANSAC → focal y pose
iniciales → refinamiento robusto con los puntos (incluidos los largueros a 2,44 m) **más miles de
restricciones densas**: cada línea pintada proyectada debe caer sobre píxeles blancos reales
(transformada de distancia). Luego, por plano, se fija la **posición de la cámara** (va en un trípode:
solo gira y hace zoom) y la calibración de cada fotograma se reduce a 4 parámetros. Entre
fotogramas, el movimiento del fondo da la rotación y el zoom con precisión sub-píxel (modelo de
4 parámetros, no una homografía libre de 8); un suavizador lo fusiona con las calibraciones absolutas.

**Seguimiento en el campo, no en la imagen.** Cada detección es un punto en metros con una
incertidumbre calculada (grande en profundidad para los jugadores lejanos). Filtro de Kalman por
jugador, asociación en dos etapas (estilo ByteTrack) con distancia de Mahalanobis + color de
camiseta, y suavizado RTS. La cámara puede girar lo que quiera: en el campo nadie "se mueve" por ello.
Los jugadores con los pies fuera de la imagen se sitúan por la cabeza (plano a 1,75 m).

**Re-identificación.** Cuando un jugador sale de plano y vuelve, se enlazan los tramos por
cinemática en metros: dónde y a qué velocidad salió, y cuánto se ha desplazado **su equipo** (medido
con los compañeros visibles) mientras no se le veía. Asignación óptima global y un veto de
ambigüedad: si hay dos continuaciones igual de plausibles no se enlaza, porque fusionar mal estropea
a dos jugadores y no fusionar solo parte a uno.

**Equipos, árbitro, portero.** Histograma de color de la camiseta normalizado por el color del
césped (corrige sombra/sol), agrupado en todo el vídeo; una equipación que se parte en dos grupos
(sol y sombra) se sigue reconociendo como el mismo equipo. Portero: la persona de color distinto que
vive junto a una portería; su equipo es el que tiene a sus defensas más cerca de esa portería.

**Dorsales.** En cada fotograma clave se leen las espaldas de los 3 jugadores más grandes con un modelo
OCR que viene dentro del paquete (sin descargas). Cada identidad vota con todas sus lecturas (un dígito
necesita 3 votos, porque "7" suele ser medio "17"); dos fragmentos del mismo equipo y dorsal que nunca
coinciden en pantalla se unen, y dorsales distintos impiden unir.

**Salida tipo SkillCorner.** Remuestreo a 10 fps; huecos cortos con curva de Hermite; jugadores
fuera de plano movidos con su equipo (`is_detected: false`); una posición extrapolada que la cámara
estaba viendo se descarta (si estuviera ahí, se habría detectado).

---

## Precisión medida

**Calibración en vídeo real** (clip de 6 s, 153 fotogramas, sin verdad de campo: se mide cuánto del
dibujo del campo cae sobre las líneas blancas de la imagen y cuánto vibra la cámara estimada entre
fotogramas; `python tools/calibration_benchmark.py`):

| variante | calibrados | alineación ≤3 px | ≤1 px | vibración pan (°) | inclinación (°) | giro (°) | zoom (log) |
|---|---|---|---|---|---|---|---|
| A. puntos clave fotograma a fotograma (como el baseline / NBJW) | 153/153 | 0,759 | 0,574 | 0,174 | 0,110 | 0,128 | 0,0240 |
| B. + refinamiento denso con las líneas | 153/153 | 0,785 | 0,628 | 0,159 | 0,058 | 0,048 | 0,0199 |
| C. + trípode (posición de cámara fija) | 153/153 | 0,788 | 0,625 | 0,028 | 0,005 | 0,038 | 0,0026 |
| **D. final: red 1 de cada 5 fotogramas + registro + fusión** | 153/153 | **0,788** | **0,633** | **0,014** | **0,003** | **0,004** | **0,0004** |

La versión final dibuja el campo más pegado a las líneas reales y **vibra entre 12 y 60 veces
menos** que calibrar cada fotograma por separado, ejecutando la red del campo 5 veces menos. La
vibración es lo que destroza las velocidades: 0,1° de error en la cámara son decenas de centímetros
en la banda lejana. (La alineación no llega a 1 porque los jugadores y la pintura gastada tapan
parte de las líneas.) Además: decodificar los mapas de calor con el desfase correcto
(etiqueta = `floor(x/2)`) subió la alineación de 0,68 a 0,76 por sí solo.

**Seguimiento con verdad conocida** (`python tools/synthetic_benchmark.py`): trayectorias reales de
22 jugadores (datos abiertos de Metrica Sports) vistas por una cámara simulada que sigue el balón con
paneos y zoom; detecciones con ruido, fallos, oclusiones y falsos positivos; 3 tramos de 1 minuto.

| métrica | resultado |
|---|---|
| pureza de identidad (observaciones asignadas a la persona correcta) | **99,7 %** |
| IDF1 | 0,81 |
| identidades por jugador real en 1 minuto | 2,0 (se fragmenta en vez de mezclar) |
| error de posición, jugadores vistos (calibración perfecta) | 5 cm mediana, 12 cm p90 |
| error de posición, jugadores extrapolados fuera de cámara | 3,8 m mediana, 11,7 m p90 |
| con un error de calibración lento de 0,1° | identidades igual (pureza 99,7 %); posición 21 cm mediana, 50 cm p90 |

El mismo banco sirvió para corregir el diseño: la primera versión (enlace voraz sin compensar el
movimiento del equipo) asignaba mal el 6 % de las observaciones; la actual, el 0,3 %. La
extrapolación siguiendo al equipo bajó el error fuera de cámara de 6,1 m a 3,8 m.

**Balón** (clip de 1080p, 153 fotogramas): se conoce su posición en el **77 %** de los fotogramas de
salida (33 % detectado directamente; el resto interpolado entre detecciones o en los pies del jugador
que lo lleva). En una muestra de 17 posiciones elegidas, las 17 eran el balón real. Antes de los filtros
nuevos, en un clip del Bayern–PSG el "balón" elegido era casi siempre la bota flúor de un jugador.

**Personas que no son jugadores** (clip Bayern–PSG de 15 s): el plano del público del inicio se
descarta entero; juez de línea, entrenador y árbitro quedan fuera de los datos de jugadores; dorsales
leídos: #17 (20 lecturas) y #27 del Bayern.

Formato verificado: `tests/test_export.py` carga la salida con kloppy, y el visor oficial
`SkillCorner_Tracking_Viewer.html` la abre sin errores (probado en Chromium: 26 identidades, 61 fotogramas, 10 Hz, portero y posesión).

---

## En qué se parece a SkillCorner y en qué no (honestamente)

Igual: formato de archivos, sistema de coordenadas, 10 fps, `is_detected`, extrapolación fuera de
cámara, proyección de las esquinas de la imagen, posesión, métricas físicas, descarte de planos que
no son la cámara principal.

Diferente / pendiente:

* **Identidades y dorsales.** SkillCorner conoce la alineación, reconoce dorsales y revisa a mano
  (anuncian ~97 % de identidades correctas). Aquí los dorsales se leen con OCR cuando la espalda del
  jugador es visible y está cerca de la cámara (en el clip de prueba a 720p: 2 de ~20 jugadores en 15 s;
  en un partido completo y a 1080p se leen muchos más). Un jugador sin dorsal leído que sale mucho rato
  de plano puede volver con otro `id`: habrá más identidades que jugadores. Con dorsal, los fragmentos se
  unen automáticamente.
* **El balón.** El detector genérico (COCO) ve el balón con poca confianza y confunde botas blancas o
  flúor, cabezas y letras de las vallas. Por eso: se guardan también las detecciones débiles, se exige
  que el candidato esté en el césped, tenga el tamaño de un balón de 22 cm a esa distancia, no esté
  sobre un jugador y no sea de un color saturado, y se elige la trayectoria que se mueve como un balón
  (las botas y los falsos positivos saltan). Cuando el balón no se ve pero un jugador lo tiene, se sitúa
  en sus pies (`is_detected: false`). La altura (`z`) no se estima. **Para mejorarlo de verdad** usa un
  detector entrenado en fútbol: `--model ruta/a/modelo_futbol.pt` (cualquier modelo Ultralytics cuyas
  clases se llamen `player`, `goalkeeper`, `referee`, `ball`; se reconocen automáticamente).
* **Repeticiones a cámara lenta desde la cámara principal:** no se detectan (las de otras cámaras sí
  se descartan porque la cámara está en otra posición).
* **Distorsión de lente:** se asume cero (correcto en la cámara principal; no en objetivos gran
  angular).
* La pasada de redes en CPU es lenta; para partidos completos usa GPU (Colab) o el Mac con MPS.

---

## Para desarrolladores

```bash
pytest -q                                  # 35 pruebas, ~30 s, no necesitan pesos
python tools/synthetic_benchmark.py        # seguimiento con verdad conocida
python tools/calibration_benchmark.py      # calibración en el clip real
python tools/make_notebooks.py             # regenera los cuadernos
```

Código en `soccercal/`: `hrnet.py` (red), `field_model.py` (puntos del campo), `camera.py` /
`calibrate.py` / `lines.py` (calibración), `camtrack.py` / `cameras.py` (cámara en el tiempo),
`detect.py`, `tracker.py`, `stitch.py`, `teams.py`, `players.py` (jugadores), `timeline.py`,
`ball.py`, `physical.py`, `skillcorner.py` (salida), `viz.py`, `pipeline.py`, `cli.py`.

## Créditos y licencias

* Pesos HRNet: *No Bells, Just Whistles* / *PnLCalib* (M. Gutiérrez-Pérez y A. Agudo), descargados
  en tiempo de ejecución desde sus publicaciones de GitHub; consulta su licencia (GPL-2.0) antes de un
  uso comercial. La arquitectura de `hrnet.py` está escrita desde el artículo de HRNet, no copiada.
* Detector: Ultralytics YOLO11 (AGPL-3.0).
* Clip de ejemplo: repositorio de NBJW. Trayectorias del banco sintético: Metrica Sports sample data.
* Formato y visor: SkillCorner open data.
