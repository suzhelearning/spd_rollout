"""Measure a real rollout run, including startup and whole-card GPU telemetry."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('original_cpu', 'gpu_vector'), required=True)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--source-ids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-count', type=int, default=2)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--cache-state', choices=('cold_scene_assets', 'warm_scene_assets'), default='cold_scene_assets')
    args = parser.parse_args()
    root = args.code_root.resolve()
    sys.path[:0] = [str(root), str(root / 'abc')]
    gpu = int(os.environ['SPD_ROLLOUT_GPU_ID'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('GPU selection differs between renderer and policy')
    selected = set(json.loads(args.source_ids.read_text()))
    selection = json.loads(args.selection.read_text())
    planned = [row for i, row in enumerate(selection['episodes'])
               if i % args.shard_count == args.shard_index and row['source_id'] in selected]
    if not planned:
        raise ValueError('Benchmark shard has no selected episodes')
    if any(args.output.glob('*/*/*/result.json')):
        raise ValueError('Benchmark output already contains rollout results; use a fresh output directory')
    if args.method == 'gpu_vector':
        if args.cache_dir is None:
            raise ValueError('GPU benchmark requires an explicit scene cache directory')
        if args.cache_state == 'cold_scene_assets' and args.cache_dir.exists() and any(args.cache_dir.iterdir()):
            raise ValueError('Cold benchmark requires an empty scene cache directory')
        os.environ['SPD_GPU_ALLOW_LEGACY_CACHE'] = '0'
    elif args.cache_state != 'cold_scene_assets':
        raise ValueError('Original CPU method imports every scene and has no warm scene asset cache')
    source_files = ['baseline_raw_rollout.py', 'rollout_policy.py', 'physx_scene.py',
                    'physx_scene_builder.py', 'isaac_headless_renderer.py', 'training_render.py',
                    'eval_selection.py', 'abc/abc_minimal/policy.py']
    if args.method == 'gpu_vector':
        source_files += ['gpu_rollout.py', 'isaac_gpu_client.py', 'gpu_physx_worker.py',
                         'gpu_physx_world.py', 'gpu_rgb_renderer.py']
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    telemetry = subprocess.Popen(
        ['nvidia-smi', f'--id={gpu}', '--query-gpu=utilization.gpu,memory.used,power.draw',
         '--format=csv,noheader,nounits', '--loop-ms=500'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def collect():
        for line in telemetry.stdout:
            parts = [piece.strip() for piece in line.split(',')]
            if len(parts) != 3:
                continue
            try:
                samples.append({'epoch_s': time.time(), 'gpu_utilization_percent': float(parts[0]),
                                'gpu_memory_mib': float(parts[1]), 'power_w': float(parts[2])})
            except ValueError:
                continue
    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    metrics = {key: 0. if args.method == 'original_cpu' else None
               for key in ('startup_s', 'load_s', 'step_roundtrip_s', 'policy_s', 'model_load_s')}
    start_epoch = time.time()
    started = time.perf_counter()
    failure = None
    try:
        module = importlib.import_module('baseline_raw_rollout' if args.method == 'original_cpu' else 'gpu_rollout')
        if args.method == 'original_cpu':
            original_pending = module._pending_rows
            def pending(planned, output, *, retry_errors):
                return [row for row in original_pending(planned, output, retry_errors=retry_errors)
                        if row['source_id'] in selected]
            module._pending_rows = pending
            def timed(owner, name, field):
                original = getattr(owner, name)
                def call(*positional, **kwargs):
                    begin = time.perf_counter()
                    try:
                        return original(*positional, **kwargs)
                    finally:
                        metrics[field] += time.perf_counter() - begin
                setattr(owner, name, call)
            from abc_minimal.policy import SPDInferencePolicy
            import abc_minimal.policy
            if Path(abc_minimal.policy.__file__).resolve() != root / 'abc/abc_minimal/policy.py':
                raise RuntimeError('Benchmark did not load the original policy snapshot')
            timed(module, 'load_policy', 'model_load_s')
            timed(module.IsaacPhysxClient, '__init__', 'startup_s')
            timed(module.IsaacPhysxClient, 'load', 'load_s')
            timed(module.IsaacPhysxClient, 'step', 'step_roundtrip_s')
            timed(SPDInferencePolicy, 'observe', 'policy_s')
            timed(SPDInferencePolicy, 'infer', 'policy_s')
        sys.argv = [str(root / ('baseline_raw_rollout.py' if args.method == 'original_cpu' else 'gpu_rollout.py')),
                    '--selection', str(args.selection), '--output', str(args.output),
                    '--shard-count', str(args.shard_count), '--shard-index', str(args.shard_index),
                    '--horizon-ticks', '720', '--seed', '123', '--diffusion-steps', '10', '--allow-diagnostic']
        if args.method == 'gpu_vector':
            sys.argv += ['--source-ids', str(args.source_ids), '--batch-size', str(args.batch_size),
                         '--dino-frame-batch-size', str(3 * args.batch_size)]
            if args.cache_dir:
                sys.argv += ['--cache-dir', str(args.cache_dir)]
        module.main()
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        elapsed = time.perf_counter() - started
        telemetry.terminate()
        try:
            telemetry.wait(timeout=10)
        except subprocess.TimeoutExpired:
            telemetry.kill(); telemetry.wait()
        thread.join(timeout=5)
        rows = []
        for path in args.output.glob('*/*/*/result.json'):
            row = json.loads(path.read_text())
            if row['source_id'] in selected:
                rows.append(row)
        runtime_metrics = args.output / 'run_metrics.json'
        if args.method == 'gpu_vector' and runtime_metrics.exists():
            metrics.update(json.loads(runtime_metrics.read_text()))
        cache_evidence = []
        if args.method == 'gpu_vector':
            for row in rows:
                metadata_path = row.get('physical_environment')
                if metadata_path and Path(metadata_path).exists():
                    metadata = json.loads(Path(metadata_path).read_text())
                    cache_evidence.append({'source_id': row['source_id'],
                                           'cache_source': metadata.get('cache_source'),
                                           'cache_hit': metadata.get('cache_hit')})
            expected_source = 'cold_import' if args.cache_state == 'cold_scene_assets' else 'gpu_prepared'
            expected_hit = args.cache_state == 'warm_scene_assets'
            cache_verified = (len(cache_evidence) == len(planned)
                              and all(item['cache_source'] == expected_source
                                      and item['cache_hit'] is expected_hit for item in cache_evidence))
        else:
            cache_verified = len(rows) == len(planned)
        report = {'method': args.method, 'gpu_id': gpu, 'shard_index': args.shard_index,
                  'shard_count': args.shard_count, 'batch_size': 1 if args.method == 'original_cpu' else args.batch_size,
                  'horizon_ticks': 720, 'selection_digest': selection['selection_digest'],
                  'code_root': str(root), 'source_hashes': source_hashes,
                  'benchmark_wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'cache_state': args.cache_state, 'cache_verified': cache_verified, 'cache_evidence': cache_evidence,
                  'shader_cache_state': 'existing_machine_shader_cache',
                  'legacy_cache_allowed': False if args.method == 'gpu_vector' else None,
                  'source_ids': [row['source_id'] for row in planned], 'started_at_epoch': start_epoch,
                  'finished_at_epoch': time.time(), 'elapsed_s': elapsed, 'failure': failure,
                  **metrics, 'episodes': [{key: row.get(key) for key in
                        ('source_id', 'task_id', 'episode_id', 'seed', 'tick', 'inference_count',
                         'status', 'reason', 'wall_seconds')} for row in rows],
                  'completed': sum(row.get('tick') == 720 and row.get('inference_count') == 90 for row in rows),
                  'gpu_utilization': sum(s['gpu_utilization_percent'] for s in samples) / len(samples) if samples else None,
                  'peak_gpu_memory_mib': max((s['gpu_memory_mib'] for s in samples), default=None),
                  'telemetry_samples': samples}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2))
        print('BENCHMARK_RESULT', json.dumps({key: value for key, value in report.items()
              if key not in ('telemetry_samples', 'episodes')}), flush=True)
    if report['completed'] != len(planned):
        raise RuntimeError('Benchmark did not complete every selected full-horizon rollout')
    if not report['cache_verified']:
        raise RuntimeError('Measured scene asset cache provenance differs from the benchmark cache label')


if __name__ == '__main__':
    main()
