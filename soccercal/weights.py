"""Model weight download and caching.

The original sportlight repository never published checkpoints. The HRNet networks
used here are the public NBJW / PnLCalib single-view models (SoccerNet-trained):
same architecture family, same 57-keypoint vocabulary. Files are ~265 MB each and
are cached in ``~/.cache/soccercal`` (override with ``SOCCERCAL_CACHE``).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS = {
    "SV_kp": {
        "urls": [
            "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
            "https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_kp",
        ],
        "sha256": "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113",
    },
    "SV_lines": {
        "urls": [
            "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
            "https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_lines",
        ],
        "sha256": "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455",
    },
}

SAMPLE_VIDEO_URL = ("https://raw.githubusercontent.com/mguti97/No-Bells-Just-Whistles/main/"
                    "examples/iniesta_sample.mp4")


def cache_dir() -> Path:
    d = Path(os.environ.get("SOCCERCAL_CACHE", Path.home() / ".cache" / "soccercal"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ssl_contexts() -> list[ssl.SSLContext]:
    """Certificate stores to try, in order.

    * a bundle named by SSL_CERT_FILE / REQUESTS_CA_BUNDLE (corporate proxies, clusters);
    * the system store (Linux, conda Python);
    * certifi: python.org installers on macOS ship without CA certificates, which gives
      "CERTIFICATE_VERIFY_FAILED" unless beginners run an extra script.
    """
    ctxs = []
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        path = os.environ.get(var)
        if path and os.path.exists(path):
            ctxs.append(ssl.create_default_context(cafile=path))
    ctxs.append(ssl.create_default_context())
    try:
        import certifi

        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    return ctxs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, quiet: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "soccercal"})
    last = None
    for ctx in _ssl_contexts():
        try:
            resp = urllib.request.urlopen(req, context=ctx, timeout=60)
            break
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                raise
            last = e
    else:
        raise last
    with resp as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if not quiet and total:
                sys.stderr.write(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
        if not quiet and total:
            sys.stderr.write("\n")
    shutil.move(tmp, dest)
    return dest


def get_weights(name: str, weights_dir: str | os.PathLike | None = None, quiet: bool = False) -> Path:
    """Path to a checkpoint, downloading (and verifying) it the first time."""
    spec = WEIGHTS[name]
    d = Path(weights_dir) if weights_dir else cache_dir()
    path = d / name
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    errors = []
    for url in spec["urls"]:
        try:
            if not quiet:
                print(f"Descargando {name} (una sola vez) desde {url}")
            download(url, path, quiet=quiet)
            digest = _sha256(path)
            if digest != spec["sha256"]:
                path.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {name}: {digest}")
            return path
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError(
        f"No se pudieron descargar los pesos {name}.\n" + "\n".join(errors) +
        f"\nDescárgalos a mano y colócalos en {d}/{name}")


def get_sample_video(dest: str | os.PathLike | None = None) -> Path:
    """A 6 s broadcast clip (Chelsea-Barcelona 2009) published with NBJW, for testing."""
    path = Path(dest) if dest else cache_dir() / "iniesta_sample.mp4"
    if not path.exists():
        download(SAMPLE_VIDEO_URL, path)
    return path
