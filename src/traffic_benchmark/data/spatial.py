"""Training-only PatchSTG spatial maps; see THIRD_PARTY_NOTICES.md for reference."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .pipeline import GlobalStandardScaler


@dataclass(frozen=True)
class SpatialMaps:
    """Gather padded sensors, then recover originals in canonical sensor order."""

    gather: np.ndarray
    restore: np.ndarray
    patch_count: int
    patch_size: int

    def validate(self, nodes):
        for array in (self.gather, self.restore):
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
                raise ValueError("Index maps must be one-dimensional integers.")
        if self.patch_count < 1 or self.patch_size < 1 or len(self.gather) != self.patch_count * self.patch_size:
            raise ValueError("Patch dimensions do not match gather map.")
        if len(self.restore) != nodes or len(np.unique(self.restore)) != nodes:
            raise ValueError("Each original sensor must have exactly one restore position.")
        if ((self.gather < 0) | (self.gather >= nodes)).any():
            raise ValueError("Gather sensor out of range.")
        if ((self.restore < 0) | (self.restore >= len(self.gather))).any():
            raise ValueError("Restore position out of range.")
        if not np.array_equal(self.gather[self.restore], np.arange(nodes)):
            raise ValueError("Restore map does not recover original sensor order.")


def training_similarity(values, train_end, scaler, segment_length=288):
    """Reference segment-average cosine similarity, using only training frames.

    Preserves 288-step segments at SD cadence; no division by zero or invented
    similarities when all pairwise cosine values have zero variance.
    """
    if not isinstance(train_end, int) or isinstance(train_end, bool) or not 0 < train_end <= len(values):
        raise ValueError("Invalid training boundary.")
    if not isinstance(segment_length, int) or isinstance(segment_length, bool) or segment_length < 1:
        raise ValueError("segment_length must be a positive integer.")
    train = np.asarray(values[:train_end], dtype=np.float64)
    if train.ndim != 3 or train.shape[-1] != 1 or train.shape[1] < 2:
        raise ValueError("Expected (time,at-least-two-sensors,1) traffic.")
    if not np.isfinite(train).all() or (train < 0).any():
        raise ValueError("Training values must be finite and nonnegative.")
    # Refit check prevents passing validation-contaminated normalization stats.
    if scaler != GlobalStandardScaler.fit(train, len(train)):
        raise ValueError("Scaler must be fitted on exactly this training interval.")
    segments = len(train) // segment_length
    if not segments:
        raise ValueError("Training interval shorter than one similarity segment.")
    normalized = (train[:segments * segment_length, :, 0] - scaler.mean) / scaler.std
    profiles = normalized.reshape(segments, segment_length, train.shape[1]).mean(axis=0).T
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    unit = np.divide(profiles, norms, out=np.zeros_like(profiles), where=norms != 0)
    cosine = np.clip(unit @ unit.T, -1, 1)
    std = cosine.std()
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate cosine similarity variance; cannot construct reference graph.")
    with np.errstate(over="raise", invalid="raise"):
        similarity = np.exp((cosine - cosine.mean()) / std)
    if not np.isfinite(similarity).all():
        raise ValueError("Nonfinite similarity.")
    return similarity


def build_spatial_maps(coordinates, similarity, depth=9, leaf_size=2, merge_factor=32):
    """Balanced latitude-first alternating splits, similarity padding, merging.

    Ties use original sensor index. Padding ranks maximum similarity to leaf
    members; equal scores use ascending sensor index. This specifies reference
    tie cases that otherwise depend on sorting implementation.
    """
    coordinates, similarity = np.asarray(coordinates), np.asarray(similarity)
    for size in (depth, leaf_size, merge_factor):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("Spatial sizes must be positive integers.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("Expected finite (nodes,2) latitude/longitude coordinates.")
    nodes = len(coordinates)
    if (np.abs(coordinates[:, 0]) > 90).any() or (np.abs(coordinates[:, 1]) > 180).any():
        raise ValueError("Coordinates out of range.")
    if similarity.shape != (nodes, nodes) or not np.isfinite(similarity).all() or (similarity < 0).any():
        raise ValueError("Expected finite nonnegative sensor similarity matrix.")
    leaves = 2 ** depth
    if leaves > nodes or leaves % merge_factor or leaf_size > nodes or int(np.ceil(nodes / leaves)) > leaf_size:
        raise ValueError("Incompatible leaf count, capacity or merge factor.")

    def partition(indices, remaining, axis):
        # Sort by coordinate, then canonical sensor ID (array index).
        order = np.lexsort((indices, coordinates[indices, axis]))
        midpoint = len(indices) // 2
        left, right = np.sort(indices[order[:midpoint]]), np.sort(indices[order[midpoint:]])
        if remaining == 1:
            return [left, right]
        return partition(left, remaining - 1, axis ^ 1) + partition(right, remaining - 1, axis ^ 1)

    groups = partition(np.arange(nodes), depth, 0)
    gather, restore = [], np.empty(nodes, dtype=np.int64)
    for group in groups:
        restore[group] = np.arange(len(gather), len(gather) + len(group))
        padded = group.tolist()
        if len(group) < leaf_size:
            candidates = np.setdiff1d(np.arange(nodes), group)
            scores = similarity[np.ix_(group, candidates)].max(axis=0)
            order = np.lexsort((candidates, -scores))
            padded.extend(candidates[order[:leaf_size - len(group)]].tolist())
        gather.extend(padded)
    maps = SpatialMaps(np.asarray(gather, dtype=np.int64), restore, leaves // merge_factor, leaf_size * merge_factor)
    maps.validate(nodes)
    maps.gather.setflags(write=False)
    maps.restore.setflags(write=False)
    return maps


def _map_checksum(maps):
    layout = json.dumps({"patch_count": maps.patch_count, "patch_size": maps.patch_size,
                         "gather_dtype": maps.gather.dtype.str,
                         "restore_dtype": maps.restore.dtype.str}, sort_keys=True)
    return hashlib.sha256(layout.encode() + maps.gather.tobytes() + maps.restore.tobytes()).hexdigest()


def save_spatial_maps(path, maps, provenance):
    """Save a cache once; identity binds contents, source inputs and split policy."""
    maps.validate(len(maps.restore))
    path = Path(path)
    identity = json.dumps(provenance, sort_keys=True, allow_nan=False)
    payload = _map_checksum(maps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, gather=maps.gather, restore=maps.restore,
                            patch_count=maps.patch_count, patch_size=maps.patch_size,
                            provenance=identity, payload_sha256=payload)


def load_spatial_maps(path, expected_provenance):
    """Refuse cache identity mismatch instead of silently reusing stale maps."""
    expected = json.dumps(expected_provenance, sort_keys=True, allow_nan=False)
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["provenance"].item()) != expected:
            raise ValueError("Spatial cache provenance differs from requested inputs/config.")
        maps = SpatialMaps(cache["gather"], cache["restore"], int(cache["patch_count"]), int(cache["patch_size"]))
        digest = _map_checksum(maps)
        if digest != str(cache["payload_sha256"].item()):
            raise ValueError("Spatial map checksum mismatch.")
    maps.validate(len(maps.restore))
    maps.gather.setflags(write=False)
    maps.restore.setflags(write=False)
    return maps
