"""soccercal: broadcast soccer video -> SkillCorner-style tracking data.

    import soccercal
    res = soccercal.run("partido.mp4", "salida/")     # todo el proceso
    res.table.head()                                  # posiciones a 10 fps
"""
__version__ = "0.3.0"

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
