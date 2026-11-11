"""Host CUDA feasibility for full-width PatchSTG batches 1/2/4, no training run."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows, load_processed_sd, make_splits
from traffic_benchmark.data.spatial import build_spatial_maps, training_similarity, save_spatial_maps, load_spatial_maps
from traffic_benchmark.evaluation.feasibility import optimizer_probe, sweep_batches
from traffic_benchmark.models import PatchSTG, PatchSTGConfig


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path, report):
    with Path(path).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare_inputs(data_dir, output_dir, cache):
    values, metadata = load_processed_sd(data_dir)
    train_end = make_splits(len(values))['train'][1]
    scaler = GlobalStandardScaler.fit(values, train_end)
    identity = {
        'schema': 1, 'numpy_version': np.__version__,
        'source_sha256': {name: sha256(data_dir/name) for name in ('flowsd.npz', 'sd_meta.csv')},
        'train_end': train_end, 'scaler': asdict(scaler), 'segment_length': 288,
        'depth': 9, 'leaf_size': 2, 'merge_factor': 32,
        'tie_policy': 'coordinate_then_sensor_index; padding_max_similarity_then_sensor_index',
        'spatial_implementation_sha256': sha256('src/traffic_benchmark/data/spatial.py'),
    }
    if cache:
        maps = load_spatial_maps(cache, identity)
    else:
        similarity = training_similarity(values, train_end, scaler)
        coordinates = np.array([[float(r['Lat']), float(r['Lng'])] for r in metadata])
        maps = build_spatial_maps(coordinates, similarity)
    save_spatial_maps(output_dir/'spatial_maps.npz', maps, identity)
    dataset = TrafficWindows(values, (0, train_end), scaler)
    samples = [dataset[i] for i in range(4)]
    with (output_dir/'probe_inputs.npz').open('xb') as stream:
        np.savez(stream, x=np.stack([s['x'] for s in samples]),
                 calendar=np.stack([s['x_calendar'] for s in samples]))
    write_json(output_dir/'prepared.json', {'spatial_provenance': identity,
               'target_starts': [s['target_start'] for s in samples],
               'cache_sha256': sha256(output_dir/'spatial_maps.npz'),
               'inputs_sha256': sha256(output_dir/'probe_inputs.npz')})
    return identity


def memory_report(device):
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'current_allocated_bytes': torch.cuda.memory_allocated(device),
            'current_reserved_bytes': torch.cuda.memory_reserved(device)}


def run_worker(output_dir, batch):
    """One process owns one disposable model, AdamW state and CUDA context."""
    row = {'batch_size': batch, 'status': 'failed', 'optimizer_steps': None}
    device = torch.device('cuda:0')
    stage = 'cuda_initialization'
    initialized = False
    try:
        if not torch.cuda.is_available():
            row['status'] = 'cuda_unavailable'
            row['optimizer_steps'] = 0
            return row
        torch.cuda.set_device(device)
        torch.manual_seed(2023)
        np.random.seed(2023)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        properties = torch.cuda.get_device_properties(device)
        row['gpu'] = {'name': properties.name, 'total_memory_bytes': properties.total_memory,
                      'compute_capability': list(torch.cuda.get_device_capability(device))}
        prepared = json.loads((output_dir/'prepared.json').read_text())
        if sha256(output_dir/'spatial_maps.npz') != prepared['cache_sha256'] or sha256(output_dir/'probe_inputs.npz') != prepared['inputs_sha256']:
            raise ValueError('Prepared input/cache file checksum mismatch.')
        maps = load_spatial_maps(output_dir/'spatial_maps.npz', prepared['spatial_provenance'])
        with np.load(output_dir/'probe_inputs.npz', allow_pickle=False) as inputs:
            x = torch.from_numpy(inputs['x'][:batch])
            calendar = torch.from_numpy(inputs['calendar'][:batch])
        model = PatchSTG(PatchSTGConfig(), maps)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        row['initial_device_free_bytes'], row['initial_device_total_bytes'] = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        initialized = True
        begin = perf_counter()
        stage = 'model_and_input_transfer'
        model = model.to(device)
        x, calendar = x.to(device), calendar.to(device)
        stage = 'forward_backward_optimizer_step'
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001, foreach=False)
        row.update(optimizer_probe(model, x, calendar, optimizer))
        torch.cuda.synchronize(device)
        row['one_off_wall_seconds'] = perf_counter() - begin
        row.update(memory_report(device))
        row['parameter_count'] = sum(p.numel() for p in model.parameters())
        row['status'] = 'passed'
    except torch.cuda.OutOfMemoryError as exc:
        row.update(status='oom', failed_stage=stage, error=f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        row.update(status='failed', failed_stage=stage, error=f'{type(exc).__name__}: {exc}')
    finally:
        if initialized and row['status'] != 'passed':
            try:
                row.update(memory_report(device))
            except Exception as exc:
                row['memory_report_error'] = f'{type(exc).__name__}: {exc}'
        write_json(output_dir/f'batch_{batch}.json', row)
    return row


def launch_worker(output_dir, batch):
    command = [sys.executable, str(Path(__file__).resolve()), '--output-dir', str(output_dir), '--worker-batch', str(batch)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (output_dir/f'batch_{batch}.log').write_text(completed.stdout + completed.stderr)
    path = output_dir/f'batch_{batch}.json'
    if not path.exists():
        return {'batch_size': batch, 'status': 'failed', 'worker_returncode': completed.returncode,
                'error': 'Worker exited without a JSON report; inspect batch log.'}
    row = json.loads(path.read_text())
    if row.get('status') == 'passed' and completed.returncode != 0:
        return {'batch_size': batch, 'status': 'failed', 'worker_returncode': completed.returncode,
                'error': 'Worker exit code disagrees with passed report.'}
    return {**row, 'worker_returncode': completed.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw/largest_sd'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, help='Optional CPU-prepared spatial cache with matching identity.')
    parser.add_argument('--worker-batch', type=int, choices=(1,2,4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_batch:
        row = run_worker(args.output_dir, args.worker_batch)
        return 0 if row['status'] == 'passed' else 1
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new path.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = ['scripts/check_patchstg_gpu.py', 'src/traffic_benchmark/evaluation/feasibility.py',
             'src/traffic_benchmark/models/patchstg.py', 'src/traffic_benchmark/data/spatial.py',
             'src/traffic_benchmark/data/pipeline.py']
    report = {
        'check': 'PatchSTG GPU execution/memory feasibility; not training or inference benchmark',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': asdict(PatchSTGConfig()), 'batch_sizes': [1,2,4], 'seed': 2023,
        'precision': 'FP32; no AMP/compile; TF32 disabled',
        'optimizer': {'name': 'AdamW', 'lr': 0.002, 'weight_decay': 0.0001, 'foreach': False},
        'objective': 'Artificial mean squared normalized output; no labels or forecast metric',
        'steps_per_attempt': 1, 'models_reinitialized_between_batches': True,
        'device': 'cuda:0', 'software': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
        'platform': platform.platform(),
        'git': {'revision': git_output('rev-parse','HEAD'), 'status': git_output('status','--porcelain')},
        'implementation_sha256': {p: sha256(p) for p in paths},
        'memory_scope': 'fresh worker process; reset after context setup, before model/input transfer; includes forward/backward and first AdamW state allocation',
        'timing_scope': 'one-off synchronized transfer/model setup plus probe; no warm-up; not inference latency or epoch throughput',
        'training_experiment_performed': False, 'cuda_available': bool(torch.cuda.is_available()),
    }
    if not report['cuda_available']:
        report.update(status='cuda_unavailable', results=[{'batch_size': b, 'status': 'not_attempted', 'reason': 'CUDA unavailable'} for b in (1,2,4)], largest_passed_batch=None)
        write_json(args.output_dir/'report.json', report)
        print('CUDA unavailable in this process. No GPU batches attempted. Saved', args.output_dir/'report.json')
        return 2
    try:
        report['spatial_provenance'] = prepare_inputs(args.data_dir, args.output_dir, args.cache)
        sweep = sweep_batches(lambda batch: launch_worker(args.output_dir,batch))
        report.update(sweep)
        report['status'] = 'passed' if sweep['all_requested_passed'] else ('partial' if sweep['largest_passed_batch'] else 'failed')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    write_json(args.output_dir/'report.json', report)
    for row in report.get('results', []):
        print('batch', row['batch_size'], row['status'],
              'peak allocated MiB:', round(row['peak_allocated_bytes']/1024**2,2) if 'peak_allocated_bytes' in row else None,
              'peak reserved MiB:', round(row['peak_reserved_bytes']/1024**2,2) if 'peak_reserved_bytes' in row else None)
    print('Saved', args.output_dir/'report.json')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
