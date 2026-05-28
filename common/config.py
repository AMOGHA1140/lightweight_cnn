"""YAML configuration loader with single-level base inheritance.

A config file may declare ``_base_`` (a path or list of paths, relative to the
file) to inherit from. Bases are loaded and deep-merged in order, then the
current file is merged on top, so an experiment file only states the keys it
changes. Pure pyyaml -- no OmegaConf/Hydra.

    base.yaml:           model: {neck: {smooth_conv: standard}}
    exp/gaconv.yaml:     _base_: ../base.yaml
                         model: {neck: {smooth_conv: gaconv}}
    -> load_config('exp/gaconv.yaml').model.neck.smooth_conv == 'gaconv'
"""

from pathlib import Path

import yaml


class Config:
    """Nested dict with attribute access (``cfg.model.neck.smooth_conv``)."""

    def __init__(self, data=None):
        for key, value in (data or {}).items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        return f"Config({self.to_dict()!r})"

    def to_dict(self):
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _deep_merge(base, override):
    """Recursively merge ``override`` onto ``base`` (dicts merge, else replace)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    bases = raw.pop("_base_", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    merged = {}
    for base in bases:
        merged = _deep_merge(merged, _load_raw(path.parent / base))
    return _deep_merge(merged, raw)


def load_config(path):
    """Load a YAML config, resolving ``_base_`` inheritance, into a ``Config``."""
    return Config(_load_raw(path))
