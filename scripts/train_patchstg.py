"""Bounded PatchSTG training/validation; no test evaluation or automatic full run."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import subprocess

# Must be set before CUDA BLAS operations when deterministic execution is used.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, load_processed_sd, make_splits
from traffic_benchmark.data.spatial import build_spatial_maps, training_similarity, load_spatial_maps, save_spatial_maps
from traffic_benchmark.models import PatchSTG, PatchSTGConfig
from traffic_benchmark.training import TrainingConfig, run_training
from traffic_benchmark.training.trainer import atomic_save, seed_everything


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def git_output(*args):
    try:return subprocess.check_output(['git',*args],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('data/raw/largest_sd'))
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--resume',type=Path,help='Local last.pt; new output directory required.')
    parser.add_argument('--train-batches',type=int,help='Explicit diagnostic override; positive.')
    parser.add_argument('--val-batches',type=int,help='Explicit diagnostic override; positive.')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Output directory exists; choose a new path.')
    raw=json.loads(args.config.read_text())
    if set(raw)!={'model','training'}:parser.error('Config must contain model and training sections only.')
    training=dict(raw['training'])
    if args.train_batches is not None:training['max_train_batches']=args.train_batches
    if args.val_batches is not None:training['max_val_batches']=args.val_batches
    config=TrainingConfig(**training);model_config=PatchSTGConfig(**raw['model'])
    if model_config.nodes!=716 or model_config.history!=12:parser.error('This CLI uses the fixed SD 716-node 12-to-12 protocol.')
    if args.device=='cuda:0' and not torch.cuda.is_available():
        args.output_dir.mkdir(parents=True,exist_ok=False)
        atomic_save(args.output_dir/'report.json',{'status':'cuda_unavailable','training_started':False,
                    'model_config':asdict(model_config),'training_config':asdict(config)},json_format=True)
        print('CUDA unavailable here; no CPU fallback or training. Run in your host pml terminal.')
        return 2
    torch.set_num_threads(2);seed_everything(config.seed)
    try:
        values,metadata=load_processed_sd(args.data_dir)
        splits=make_splits(len(values));train_end=splits['train'][1]
        scaler=GlobalStandardScaler.fit(values,train_end)
        source_hashes={name:sha256(args.data_dir/name) for name in ('flowsd.npz','sd_meta.csv')}
        spatial_identity={'schema':1,'numpy_version':np.__version__,'source_sha256':source_hashes,
             'train_end':train_end,'scaler':asdict(scaler),'segment_length':288,'depth':9,'leaf_size':2,'merge_factor':32,
             'tie_policy':'coordinate_then_sensor_index; padding_max_similarity_then_sensor_index',
             'spatial_implementation_sha256':sha256('src/traffic_benchmark/data/spatial.py')}
        if args.cache:maps=load_spatial_maps(args.cache,spatial_identity)
        else:
            similarity=training_similarity(values,train_end,scaler)
            coordinates=np.asarray([[float(r['Lat']),float(r['Lng'])] for r in metadata])
            maps=build_spatial_maps(coordinates,similarity);del similarity
        train=TrafficWindows(values,splits['train'],scaler)
        validation=TrafficWindows(values,splits['val'],scaler)
        implementation_paths=['scripts/train_patchstg.py','src/traffic_benchmark/training/trainer.py',
            'src/traffic_benchmark/models/patchstg.py','src/traffic_benchmark/data/spatial.py',
            'src/traffic_benchmark/data/pipeline.py','src/traffic_benchmark/evaluation/metrics.py']
        software={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                  'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        identity={'model_config':asdict(model_config),'spatial':spatial_identity,'splits':splits,
                  'source_sha256':source_hashes,'scaler':asdict(scaler),'software':software,'device':args.device,
                  'implementation_sha256':{p:sha256(p) for p in implementation_paths},
                  'calendar':'assumed 2019-01-01 00:00; 15 min; Monday=0; no timezone',
                  'mask':'nonzero targets only; no input imputation','precision':'FP32; deterministic algorithms; no AMP/compile/TF32',
                  'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']}
        # Model initialization also uses the recorded seed, before run_training.
        model=PatchSTG(model_config,maps)
        provenance={'created_utc':datetime.now(timezone.utc).isoformat(),
            'config_source_sha256':sha256(args.config),'resolved_model':asdict(model_config),
            'resolved_training':asdict(config),'overrides':{'train_batches':args.train_batches,'val_batches':args.val_batches},
            'git':{'revision':git_output('rev-parse','HEAD'),'status':git_output('status','--porcelain')},
            'platform':platform.platform(),'software':software,'seed':config.seed,
            'installed_python_packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in distributions()],key=lambda d:d['name'].lower()),
            'environment_inventory_note':'Observed installed versions, including unrelated packages; not a solver lock or install specification.',
            'dataset_version':'author-processed SD 2019 assumption; identified by recorded file hashes',
            'resume_checkpoint_sha256':sha256(args.resume) if args.resume else None}
        if args.device=='cuda:0':
            props=torch.cuda.get_device_properties(0)
            provenance['gpu']={'name':props.name,'total_memory_bytes':props.total_memory,
                               'compute_capability':list(torch.cuda.get_device_capability(0))}
        report=run_training(model,train,validation,scaler,config,args.output_dir,identity,args.device,args.resume,run_metadata=provenance)
        save_spatial_maps(args.output_dir/'spatial_maps.npz',maps,spatial_identity)
        print('Diagnostic subset:',report['diagnostic_subset'])
        print('Saved',args.output_dir/'report.json')
        print('No test-set evaluation performed.')
        return 0
    except Exception as exc:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        if 'provenance' in locals() and not (args.output_dir/'provenance.json').exists():
            atomic_save(args.output_dir/'provenance.json',provenance,json_format=True)
        if not (args.output_dir/'report.json').exists():
            atomic_save(args.output_dir/'report.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}'},json_format=True)
        print(f'Training command failed: {type(exc).__name__}: {exc}')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
