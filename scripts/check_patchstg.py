"""Prepare real SD spatial maps and check full-width PatchSTG on CPU; no training."""

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, load_processed_sd, make_splits
from traffic_benchmark.data.spatial import build_spatial_maps, training_similarity, save_spatial_maps, load_spatial_maps
from traffic_benchmark.models import PatchSTG, PatchSTGConfig


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(['git',*args],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/raw/largest_sd'))
    parser.add_argument('--output-dir',type=Path,required=True,help='New directory; existing outputs are never replaced.')
    parser.add_argument('--cache',type=Path,help='Optional existing spatial cache; identity must match exactly.')
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new path.')
    seed=2023
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(2)
    values,metadata=load_processed_sd(args.data_dir)
    splits=make_splits(len(values));train_end=splits['train'][1]
    scaler=GlobalStandardScaler.fit(values,train_end)
    identity={
        'schema':1,'numpy_version':np.__version__,'source_sha256':{name:sha256(args.data_dir/name) for name in ('flowsd.npz','sd_meta.csv')},
        'train_end':train_end,'scaler':asdict(scaler),'segment_length':288,
        'depth':9,'leaf_size':2,'merge_factor':32,
        'tie_policy':'coordinate_then_sensor_index; padding_max_similarity_then_sensor_index',
        'spatial_implementation_sha256':sha256('src/traffic_benchmark/data/spatial.py'),
    }
    if args.cache:
        maps=load_spatial_maps(args.cache,identity)
    else:
        similarity=training_similarity(values,train_end,scaler)
        coords=np.asarray([[float(row['Lat']),float(row['Lng'])] for row in metadata])
        maps=build_spatial_maps(coords,similarity)
        del similarity
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cache_path=args.output_dir/'spatial_maps.npz'
    save_spatial_maps(cache_path,maps,identity)
    cached=load_spatial_maps(cache_path,identity)
    np.testing.assert_array_equal(cached.gather,maps.gather)
    dataset=TrafficWindows(values,splits['train'],scaler)
    sample=dataset[0]
    x=torch.from_numpy(sample['x']).unsqueeze(0)
    calendar=torch.from_numpy(sample['x_calendar']).unsqueeze(0)
    config=PatchSTGConfig()
    model=PatchSTG(config,maps)
    model.train();prediction=model(x,calendar)
    if not torch.isfinite(prediction).all():raise RuntimeError('Nonfinite CPU prediction.')
    # An artificial scalar probes autograd, not a fitted forecasting objective.
    prediction.square().mean().backward()
    for name,param in model.named_parameters():
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise RuntimeError(f'Missing/nonfinite gradient: {name}')
    model.eval()
    with torch.no_grad():
        expected=model(x,calendar)
        original_units=expected.double()*scaler.std+scaler.mean
        if not torch.isfinite(original_units).all():raise RuntimeError('Nonfinite inverse scaling.')
    checkpoint=io.BytesIO();torch.save(model.state_dict(),checkpoint);checkpoint.seek(0)
    restored=PatchSTG(config,maps)
    restored.load_state_dict(torch.load(checkpoint,weights_only=True));restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(expected,restored(x,calendar),rtol=0,atol=0)
    report={
        'check':'PatchSTG SD real-data CPU forward/backward; untrained',
        'implementation':'code-aligned adaptation; not official reproduction',
        'reference_commit':'8b298b230227a304a082e05241c69dca3740a73d',
        'config':asdict(config),'seed':seed,'device':'cpu','cpu_threads':2,
        'software':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
        'git':{'revision':git_output('rev-parse','HEAD'),'status':git_output('status','--porcelain')},
        'implementation_sha256':{p:sha256(p) for p in ('scripts/check_patchstg.py','src/traffic_benchmark/models/patchstg.py','src/traffic_benchmark/data/spatial.py','src/traffic_benchmark/data/pipeline.py')},
        'spatial_provenance':identity,'cache_sha256':sha256(cache_path),
        'spatial':{'original_sensors':config.nodes,'padded_positions':len(maps.gather),
                   'extra_positions':len(maps.gather)-config.nodes,'patch_count':maps.patch_count,'patch_size':maps.patch_size,
                   'training_segments':train_end//288,'gather_sha256':hashlib.sha256(maps.gather.tobytes()).hexdigest(),
                   'restore_sha256':hashlib.sha256(maps.restore.tobytes()).hexdigest()},
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'output_shape':list(prediction.shape),'finite_prediction':True,'finite_parameter_gradients':True,
        'checkpoint_reload_exact':True,'original_unit_conversion_finite':True,
        'training_performed':False,'gpu_checked':False,'forecast_accuracy_measured':False,
    }
    with (args.output_dir/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('spatial','parameter_count','output_shape','finite_parameter_gradients','checkpoint_reload_exact')},indent=2))
    print('Saved',args.output_dir/'report.json')


if __name__=='__main__':
    main()
