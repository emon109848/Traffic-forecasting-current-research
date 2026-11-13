"""Integrity checks for frozen, validation-selected PatchSTG checkpoints."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from traffic_benchmark.models import PatchSTG, PatchSTGConfig


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def validate_checkpoint(checkpoint, report, source_hashes, splits, scaler):
    """Fail closed on diagnostic runs, stale inputs/code, or wrong selection."""
    if (report.get('status') != 'completed' or report.get('diagnostic_subset') is not False
            or not report.get('history')):
        raise ValueError('A completed full-split training run is required.')
    if checkpoint.get('kind') != 'best':
        raise ValueError('Test evaluation requires the best checkpoint.')
    identity = report['identity']
    if canonical(checkpoint['identity']) != canonical(identity):
        raise ValueError('Checkpoint and training report identities differ.')
    selected = min(report['history'], key=lambda row: row['validation']['overall_pooled']['mae'])
    score = selected['validation']['overall_pooled']['mae']
    if (checkpoint['epoch'] != selected['epoch'] or report['best_epoch'] != selected['epoch']
            or checkpoint['validation_mae'] != score or report['best_validation_mae'] != score):
        raise ValueError('Checkpoint does not match validation-only best selection.')
    if canonical(identity['source_sha256']) != canonical(source_hashes):
        raise ValueError('Dataset checksums differ from training.')
    if canonical(identity['splits']) != canonical(splits):
        raise ValueError('Split boundaries differ from training.')
    if checkpoint['scaler'] != asdict(scaler) or identity['scaler'] != asdict(scaler):
        raise ValueError('Training-only scaler differs from checkpoint.')
    if identity['spatial']['source_sha256'] != source_hashes or identity['spatial']['scaler'] != asdict(scaler):
        raise ValueError('Spatial preprocessing identity differs from dataset/scaler.')
    for path, expected in identity['implementation_sha256'].items():
        if file_sha256(path) != expected:
            raise ValueError(f'Training implementation changed: {path}')
    return identity


def restore_frozen_model(checkpoint, maps):
    """Check spatial buffers before strict loading; loading would overwrite them."""
    state = checkpoint['model']
    for name, expected in (('gather_indices', maps.gather), ('restore_indices', maps.restore)):
        actual = state[name]
        if actual.dtype != torch.long or not np.array_equal(actual.cpu().numpy(), expected):
            raise ValueError(f'Checkpoint spatial buffer mismatch: {name}')
    model = PatchSTG(PatchSTGConfig(**checkpoint['identity']['model_config']), maps)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError('Nonfinite checkpoint parameters.')
    model.requires_grad_(False)
    return model.eval()
