"""Run a fixed strategy comparison and report every scheduled outcome."""

import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .experiment import ExperimentSettings, run_experiment
from .persistence import write_json
from .selection_strategies import NoveltySelectionStrategy, NoveltyPredictabilitySelectionStrategy, _image_hash

CONDITIONS = ("novelty", "predictability-random", "predictability-imagenet")


def _mean(values):
    return float(np.mean(values)) if values else None


def summarize_run(directory, *, expected_steps=None):
    """Read recorded outcomes; never substitute the highest-scoring earlier image."""
    directory = Path(directory)
    data = json.loads((directory / "session.json").read_text())
    events = [e for e in data['events'] if e['action'] == 'select']
    genomes = {g['key']: g for g in data['genomes']}
    presentations = sum(len(e['displayed']) for e in events)
    if expected_steps is not None:
        summary = data.get('summary') or {}
        if (len(events) != expected_steps or presentations != 9 * expected_steps or
            len(genomes) != 9 + 8 * (expected_steps - 1) or
            summary.get('status') != 'complete' or summary.get('decisions') != expected_steps or
            summary.get('candidate_presentations') != presentations or summary.get('unique_candidates') != len(genomes)):
            raise ValueError('Completed-run counts do not match the scheduled budget.')
    if events and data['selected'] != events[-1]['genome']:
        raise ValueError('Final image does not match the final recorded selection.')
    selected_images, selected_hashes, trajectory, fresh_errors = [], [], [], []
    selected_novelty, repeated, retained = [], [], []
    scoring_times, training_times = [], []
    updates = samples = masked_inputs = 0
    for step, event in enumerate(events):
        evaluation = event.get('evaluation') or {}
        meta = evaluation.get('metadata', {})
        position = event['position']
        with Image.open(directory / genomes[event['genome']]['image']) as source:
            pixels = np.array(source.convert('RGB'))
        selected_images.append(pixels.astype(np.float64) / 255)
        selected_hashes.append(_image_hash(pixels))
        novelty = None
        if meta.get('reference_available', False):
            column = evaluation['names'].index('pixel_novelty')
            novelty = float(evaluation['values'][position][column])
            selected_novelty.append(novelty)
        flags = meta.get('seen_before', [False] * len(event['displayed']))
        new_errors = []
        if meta.get('comprehension_available', False):
            new_errors = [float(error) for error, seen in zip(meta['masked_mse'], flags, strict=True)
                          if not seen and error is not None]
            fresh_errors.extend(new_errors)
        if step:
            retained.append(position == 0)
            repeated.append(bool(flags[position]))
        if 'scoring_seconds' in meta:
            scoring_times.append(meta['scoring_seconds'])
        training_times.append(meta.get('training_seconds', 0))
        updates += meta.get('training', {}).get('updates', 0)
        samples += meta.get('training', {}).get('sampled_examples', 0)
        masked_inputs += meta.get('masked_inputs_scored', 0)
        trajectory.append({'decision': step + 1, 'selected_id': event['genome'],
                           'selected_novelty': novelty, 'fresh_masked_mse': _mean(new_errors),
                           'fresh_error_count': len(new_errors), 'selected_seen_before': flags[position],
                           'retained_parent': position == 0 if step else None})
    # Mean squared distance over all unordered selected-image pairs, duplicates included.
    n = len(selected_images)
    diversity = float(2 * n / (n - 1) * np.var(np.stack(selected_images), axis=0).mean()) if n > 1 else 0.0
    return {
        'decisions': len(events), 'candidate_presentations': presentations, 'genomes': len(genomes),
        'final_image': genomes[data['selected']]['image'] if events else None,
        'mean_selected_novelty': _mean(selected_novelty), 'mean_fresh_masked_mse': _mean(fresh_errors),
        'fresh_error_count': len(fresh_errors), 'selected_repeat_rate': _mean(repeated),
        'parent_retention_rate': _mean(retained), 'unique_selected_images': len(set(selected_hashes)),
        'selected_pairwise_pixel_mse': diversity,
        'elapsed_seconds': (data.get('summary') or {}).get('elapsed_seconds'),
        'scoring_seconds': sum(scoring_times) if scoring_times else None,
        'training_seconds': sum(training_times), 'optimizer_updates': updates,
        'training_examples': samples, 'masked_inputs_scored': masked_inputs, 'trajectory': trajectory,
    }


def write_comparison_report(directory):
    """Assemble scheduled rows, including missing, pending and failed runs."""
    directory = Path(directory)
    manifest = json.loads((directory / 'comparison.json').read_text())
    rows, trajectory = [], []
    for entry in manifest['runs']:
        row = dict(entry)
        session = directory / entry['path'] / 'session.json'
        if session.exists():
            try:
                metrics = summarize_run(session.parent, expected_steps=manifest['settings']['steps']
                                        if entry['status'] == 'complete' else None)
                for point in metrics.pop('trajectory'):
                    trajectory.append({'condition': entry['condition'], 'seed': entry['seed'], **point})
                if metrics['final_image']:
                    metrics['final_image'] = str(Path(entry['path']) / metrics['final_image'])
                row.update(metrics)
            except (ValueError, KeyError, OSError) as exc:
                row.update(status='failed', error=f'Report validation: {exc}')
        elif entry['status'] == 'complete':
            row.update(status='failed', error='Completed run is missing session.json')
        rows.append(row)
    report = {'status': 'complete' if all(r['status'] == 'complete' for r in rows) else 'incomplete',
              'runs': rows, 'settings': manifest['settings'], 'observer_settings': manifest.get('observer_settings', {})}
    write_json(directory / 'report.json', report)
    fields = ['condition', 'seed', 'decision', 'selected_id', 'selected_novelty', 'fresh_masked_mse',
              'fresh_error_count', 'selected_seen_before', 'retained_parent']
    with (directory / 'trajectories.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trajectory)
    seeds = list(dict.fromkeys(r['seed'] for r in rows))
    sheet = Image.new('RGB', (720, 245 * len(seeds)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row in rows:
        x, y = CONDITIONS.index(row['condition']) * 240, seeds.index(row['seed']) * 245
        draw.text((x + 8, y + 6), f"{row['condition']} / seed {row['seed']}", fill='black')
        draw.text((x + 8, y + 23), row['status'], fill='black')
        if row.get('final_image'):
            with Image.open(directory / row['final_image']) as source:
                sheet.paste(source.convert('RGB').resize((180, 180)), (x + 30, y + 48))
        else:
            draw.text((x + 12, y + 110), 'No selected image', fill='gray')
    sheet.save(directory / 'final-images.png')
    def show(value):
        return '—' if value is None else f'{value:.4g}'
    lines = ['# Experiment 1 comparison', '', f"Status: {report['status']}. Every scheduled run is shown.", '',
             '[Recorded configuration](comparison.json)', '', '![Final selected images](final-images.png)', '',
             '| Condition | Seed | Status | Repeat rate | Fresh-image MSE | Pairwise pixel MSE | Seconds |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| [{row['condition']}]({row['path']}/session.json) | {row['seed']} | {row['status']} | "
                     f"{show(row.get('selected_repeat_rate'))} | {show(row.get('mean_fresh_masked_mse'))} | "
                     f"{show(row.get('selected_pairwise_pixel_mse'))} | {show(row.get('elapsed_seconds'))} |")
    lines.extend(['', 'Repeat rate excludes the first decision and counts selected images previously displayed, '
                  'including rejected candidates. Fresh-image MSE excludes repeats and unavailable first-grid values. '
                  'The novelty-only condition has no prediction error. Pairwise pixel MSE uses all selected images '
                  '(including repeats) and the same calculation in every condition.', '',
                  'Independent breeding trajectories see different images: their prediction errors do not isolate '
                  'initialization effects. Raw pixel distances and rank scores are proxies, not an interestingness scale. '
                  'Final images are endpoints, never the highest-scoring earlier image. Failed runs show their last '
                  'available selection, if any.', '',
                  'See [report.json](report.json) for counts and costs, [trajectories.csv](trajectories.csv) for '
                  'per-decision measurements, and each run’s `grids/` directory for full trajectories. '
                  'Novelty-only scoring time is not separately instrumented; total seconds include rendering and saving.'])
    for row in rows:
        if row.get('error'):
            lines.extend(['', f"- {row['condition']} / seed {row['seed']}: {row['error']}"])
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report


def run_comparison(*, output_dir, seeds=(7, 8, 9), steps=100, size=96, mutation_strength=.2,
                   topology=True, comprehension_weight=.5, training_steps=20, batch_size=16,
                   learning_rate=.001, cache_dir=None, device="cpu", progress=print):
    """Fresh complete strategies, shared settings, existing runner; never overwrite runs."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Supply a nonempty list of distinct seeds.')
    settings = ExperimentSettings(seed=seeds[0], steps=steps, size=size, mutation_strength=mutation_strength, topology=topology)
    all_settings = [replace(settings, seed=seed) for seed in seeds]
    observer_options = dict(comprehension_weight=comprehension_weight, training_steps=training_steps,
                            batch_size=batch_size, learning_rate=learning_rate, cache_dir=cache_dir, device=device)
    # Validate observer options without initializing a model or creating output.
    NoveltyPredictabilitySelectionStrategy(**observer_options)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'settings': {k: v for k, v in asdict(settings).items() if k not in ('seed', 'selection_seed')},
                'observer_settings': {k: v for k, v in observer_options.items() if k != 'cache_dir'},
                'runs': [{'condition': condition, 'seed': seed, 'path': f'{condition}-seed{seed}', 'status': 'pending'}
                         for seed in seeds for condition in CONDITIONS]}
    write_json(directory / 'comparison.json', manifest)
    try:
        for index, entry in enumerate(manifest['runs']):
            entry['status'] = 'running'
            write_json(directory / 'comparison.json', manifest)
            if progress:
                progress(f"Run {index + 1}/{len(manifest['runs'])}: {entry['condition']}, seed {entry['seed']}")
            try:
                strategy = (NoveltySelectionStrategy() if entry['condition'] == 'novelty' else
                            NoveltyPredictabilitySelectionStrategy(observer_initialization=entry['condition'].split('-', 1)[1], **observer_options))
                run_experiment(selection_strategy=strategy, output_dir=directory / entry['path'],
                               settings=all_settings[list(seeds).index(entry['seed'])], progress=progress)
                entry['status'] = 'complete'
            except KeyboardInterrupt:
                entry.update(status='interrupted', error='Interrupted before completion; existing artifacts retained.')
                write_json(directory / 'comparison.json', manifest)
                raise
            except Exception as exc:
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if progress:
                    progress(entry['error'])
            write_json(directory / 'comparison.json', manifest)
    finally:
        # Also leave an inspectable partial report after interruption.
        report = write_comparison_report(directory)
    return report
