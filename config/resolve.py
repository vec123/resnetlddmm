"""Config resolution, run identity and manifests.

JOB: turn YAML + overrides into validated config objects
Three Layers in order:
    1. defaults    ← ExperimentConfig() — every field with its hardcoded default
    2. file        ← YAML you pass (only the fields that differ)
    3. overrides   ← --set encoder.latent_dim=8 (command-line tweaks)

Resolution order is **defaults -> file -> --set overrides**, and every layer is
validated against the dataclass schema as it is applied. An unknown key path
raises: a silently-ignored typo'd override is the classic reason an ablation
"shows no effect", and that failure is invisible in the results.

Everything here works on plain dicts and only builds the dataclasses at the end,
which keeps frozen members out of the mutation path entirely.
"""

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import typing
from datetime import datetime, timezone

import yaml

from config.config_fields import ExperimentConfig


# --------------------------------------------------------------------------- #
# Schema walking
# --------------------------------------------------------------------------- #
def _unwrap_optional(hint):
    """``Optional[X]`` -> ``X``; anything else unchanged."""
    if typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _field_hints(cls):
    """``{field name: type hint}`` for a dataclass, or ``None`` if not one."""
    if not dataclasses.is_dataclass(cls):
        return None
    hints = typing.get_type_hints(cls)
    return {f.name: hints.get(f.name, typing.Any) for f in dataclasses.fields(cls)}


def _child_type(cls, key, path_so_far):
    """Type of ``cls.key``, raising with the valid options if it doesn't exist."""
    hints = _field_hints(cls)
    if hints is None or key not in hints:
        where = ".".join(path_so_far) or "<root>"
        valid = sorted(hints) if hints else []
        raise KeyError(
            f"unknown config key {'.'.join(path_so_far + [key])!r}: "
            f"{where} has no field {key!r}. Valid fields here: {valid}"
        )
    return _unwrap_optional(hints[key])


def _list_item_type(hint):
    """Element type of a ``List[X]`` hint, else ``None``."""
    if typing.get_origin(hint) is list:
        args = typing.get_args(hint)
        if args:
            return _unwrap_optional(args[0])
    return None


def _coerce(value, hint):
    """Light numeric coercion; YAML already produces the right type for the rest."""
    hint = _unwrap_optional(hint)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint is int and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# --------------------------------------------------------------------------- #
# Merge / override
# --------------------------------------------------------------------------- #
def _merge(base, incoming, cls, path=()):
    """Recursively overlay ``incoming`` onto ``base``, validating against ``cls``.

    Lists REPLACE rather than merge element-wise: ``encoder.layers`` and
    ``training.losses.terms`` are whole specifications, and half-overlaying one
    onto a default of a different length has no sensible meaning.
    """
    out = dict(base)
    for key, value in (incoming or {}).items():
        child_cls = _child_type(cls, key, list(path))
        if isinstance(value, dict) and dataclasses.is_dataclass(child_cls):
            out[key] = _merge(out.get(key, {}), value, child_cls, path + (key,))
        else:
            out[key] = _coerce(value, child_cls)
    return out


def apply_override(data, cls, dotted_path, value):
    """Apply one ``a.b.c=value`` override in place, validating the whole path.

    List indices are supported (``encoder.layers.0.target_irreps``) so a sweep can
    reach into a layer stack without restating it.
    """
    parts = dotted_path.split(".")
    node, node_cls, walked = data, cls, []

    for key in parts[:-1]:
        item_type = _list_item_type(node_cls) if node_cls is not None else None
        if item_type is not None:
            index = _as_index(key, node, dotted_path)
            node, node_cls = node[index], item_type
        else:
            node_cls = _child_type(node_cls, key, walked)
            if key not in node:
                node[key] = {}
            node = node[key]
        walked.append(key)

    leaf = parts[-1]
    item_type = _list_item_type(node_cls) if node_cls is not None else None
    if item_type is not None:
        node[_as_index(leaf, node, dotted_path)] = _coerce(value, item_type)
    else:
        node[leaf] = _coerce(value, _child_type(node_cls, leaf, walked))
    return data


def _as_index(key, sequence, dotted_path):
    if not key.lstrip("-").isdigit():
        raise KeyError(f"{dotted_path!r}: expected a list index, got {key!r}")
    index = int(key)
    if not -len(sequence) <= index < len(sequence):
        raise KeyError(
            f"{dotted_path!r}: index {index} out of range for a list of "
            f"{len(sequence)} item(s)")
    return index


def parse_override(text):
    """``"a.b=1"`` -> ``("a.b", 1)``. The value is parsed as YAML, so ``20`` is an
    int, ``true`` a bool, ``null`` None, and bare words stay strings."""
    if "=" not in text:
        raise ValueError(f"--set expects key.path=value, got {text!r}")
    path, _, raw = text.partition("=")
    return path.strip(), yaml.safe_load(raw)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def _build(cls, data):
    """Recursively construct dataclass ``cls`` from a plain dict."""
    hints = _field_hints(cls)
    if hints is None or not isinstance(data, dict):
        return data

    kwargs = {}
    for key, value in data.items():
        hint = _unwrap_optional(hints[key])
        item_type = _list_item_type(hint)
        if item_type is not None and dataclasses.is_dataclass(item_type) and isinstance(value, list):
            kwargs[key] = [_build(item_type, item) for item in value]
        elif dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[key] = _build(hint, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def resolve_config(path=None, overrides=(), schema=ExperimentConfig):
    """defaults -> file -> ``--set``. Returns ``(config, resolved_dict, applied)``.

    ``resolved_dict`` is the fully-merged plain-dict form -- the thing that gets
    hashed for the run id and written to the manifest, so the id describes exactly
    what ran.
    """
    data = dataclasses.asdict(schema())

    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            file_data = yaml.safe_load(f) or {}
        if not isinstance(file_data, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        data = _merge(data, file_data, schema)

    applied = {}
    for item in overrides:
        dotted, value = parse_override(item)
        apply_override(data, schema, dotted, value)
        applied[dotted] = value

    config = _build(schema, data)
    config.validate()
    return config, data, applied


# --------------------------------------------------------------------------- #
# Run identity + manifest
# --------------------------------------------------------------------------- #
def config_hash(resolved, length=10):
    """Short stable digest of a resolved config: same config => same id.

    Sorted-key JSON of the resolved dict, so key order and Python's per-process
    hash randomization can't perturb it. The DIFF between two configs is the
    description of an ablation; this is its name.
    """
    blob = json.dumps(resolved, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def _git_state():
    def _git(*args):
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        # A dirty tree means the code that ran isn't the code at that SHA -- results
        # from a dirty run are not reproducible from the SHA alone.
        "dirty": None if status is None else bool(status.strip()),
    }


def _package_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "torch_geometric", "e3nn", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def build_manifest(resolved, run_id, overrides, device, config_path=None):
    """Everything needed to attribute a result to its inputs.

    Without this, a metrics curve is not attributable to a config, a commit or a
    seed -- and an unattributable result is not a result.
    """
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path else None,
        "overrides": dict(overrides or {}),
        "git": _git_state(),
        "versions": _package_versions(),
        "device": str(device),
        "seed": resolved.get("seed"),
        "config": resolved,
    }


def write_manifest(manifest, output_dir, filename="manifest.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    return path