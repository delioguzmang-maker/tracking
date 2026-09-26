"""soccercal: broadcast soccer video -> SkillCorner-style tracking data.

    import soccercal
    res = soccercal.run("partido.mp4", "salida/")     # todo el proceso
    res.table.head()                                  # posiciones a 10 fps
"""
__version__ = "0.4.0"


def version() -> str:
    """Version plus the exact code commit (to check that Colab / your Mac run the latest code)."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %cd", "--date=format:%Y-%m-%d %H:%M"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        out = ""
    return f"{__version__} (código {out})" if out else __version__

_LAZY = {
    "Config": ("config", "Config"),
    "run": ("pipeline", "run"),
    "build": ("pipeline", "build"),
    "analyze": ("analysis", "analyze"),
    "Analysis": ("analysis", "Analysis"),
    "render_video": ("viz", "render_video"),
    "load_kloppy": ("skillcorner", "load_kloppy"),
    "get_sample_video": ("weights", "get_sample_video"),
}


def __getattr__(name):  # lazy: `import soccercal` stays fast and never fails on a missing torch
    if name in _LAZY:
        import importlib

        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{mod}", __name__), attr)
    raise AttributeError(name)


__all__ = list(_LAZY)
