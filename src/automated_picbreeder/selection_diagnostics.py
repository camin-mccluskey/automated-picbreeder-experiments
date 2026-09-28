"""Small held-out-image and common-grid checks, separate from breeding outcomes."""

import hashlib
import json
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageDraw

from .persistence import write_json
from .selection_strategies import NoveltySelectionStrategy, NoveltyPredictabilitySelectionStrategy


def synthetic_images(seed, *, per_family=6, size=32):
    """Independent draws of flat colours, gradients, repeated patterns and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:1:complex(size), 0:1:complex(size)]
    families = {name: [] for name in ('flat', 'gradient', 'pattern', 'noise')}
    for _ in range(per_family):
        a, b = rng.random((2, 3))
        angle = rng.uniform(0, 2 * np.pi)
        ramp = x * np.cos(angle) + y * np.sin(angle)
        ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
        repeat = .5 + .5 * np.sin(2 * np.pi * (rng.integers(1, 5) * x + rng.integers(1, 5) * y) + rng.uniform(0, 2 * np.pi))
        values = {'flat': np.broadcast_to(a, (size, size, 3)),
                  'gradient': a + ramp[:, :, None] * (b - a),
                  'pattern': a + repeat[:, :, None] * (b - a), 'noise': rng.random((size, size, 3))}
        for name, pixels in values.items():
            families[name].append(np.rint(pixels * 255).astype(np.uint8))
    return families


def mean_fill_errors(images):
    """Predict every hidden tile with its visible region's per-channel mean."""
    import torch
    from .image_predictability import MaskedImageObserver, masked_inputs, masked_mse
    errors = []
    for image in images:
        targets = MaskedImageObserver.prepare(image)[None].expand(16, -1, -1, -1)
        _, visible = masked_inputs(targets, torch.arange(16))
        mean = (targets * visible).sum((2, 3), keepdim=True) / visible.sum((2, 3), keepdim=True)
        errors.append(float(masked_mse(mean.expand_as(targets), targets, visible).mean()))
    return np.asarray(errors)


def run_observer_diagnostics(*, output_dir, cache_dir=None, updates=40, batch_size=16, device="cpu"):
    """Train on one synthetic set, inspect fresh examples, retaining all families."""
    from .image_predictability import MaskedImageObserver
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    train, heldout = synthetic_images(101), synthetic_images(202, per_family=3)
    for split, families in (('train', train), ('heldout', heldout)):
        (directory / split).mkdir()
        for family, images in families.items():
            for i, image in enumerate(images):
                Image.fromarray(image).save(directory / split / f'{family}-{i}.png')
    training_images = [image for images in train.values() for image in images]
    test_images = [image for images in heldout.values() for image in images]
    baseline = mean_fill_errors(test_images)
    rows, records, reconstructions = [], {}, {}
    for initialization in ('random', 'imagenet'):
        observer = MaskedImageObserver(initialization=initialization, cache_dir=cache_dir, device=device)
        observer.initialize(7)
        for trained in (False, True):
            if trained:
                records[initialization] = observer.train([observer.prepare(i) for i in training_images],
                                                        steps=updates, batch_size=batch_size, seed=9)
            errors, reconstructed = observer.predict(test_images)
            for index, family in enumerate(heldout):
                sl = slice(index * 3, (index + 1) * 3)
                rows.append({'initialization': initialization, 'updates': updates if trained else 0,
                             'family': family, 'masked_mse': float(errors[sl].mean()),
                             'mean_fill_mse': float(baseline[sl].mean())})
            if trained:
                reconstructions[initialization] = reconstructed
        records[initialization].update(observer=observer.describe(), model_hash=observer.state_hash())
    report = {'train_seed': 101, 'heldout_seed': 202, 'train_images': 24, 'heldout_images': 12,
              'model_seed': 7, 'update_seed': 9, 'batch_size': batch_size, 'rows': rows, 'training': records}
    write_json(directory / 'diagnostics.json', report)
    canvas = Image.new('RGB', (600, 4 * 180 + 25), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(('Fresh image', 'Random after training', 'Pretrained after training')):
        draw.text((col * 200 + 5, 5), label, fill='black')
    for row, family in enumerate(heldout):
        index = row * 3
        samples = [test_images[index], np.rint(reconstructions['random'][index] * 255).astype(np.uint8),
                   np.rint(reconstructions['imagenet'][index] * 255).astype(np.uint8)]
        for col, pixels in enumerate(samples):
            canvas.paste(Image.fromarray(pixels).resize((150, 150)), (col * 200 + 20, row * 180 + 25))
        draw.text((5, row * 180 + 177), family, fill='black')
    canvas.save(directory / 'predictions.png')
    lines = ['# Held-out synthetic image diagnostics', '',
             'Train and held-out images are separate draws. The fixed budget is a small learning check, not an interestingness metric.', '',
             '![Predictions](predictions.png)', '',
             '| Initialization | Updates | Family | Masked MSE | Visible-mean fill MSE |', '| --- | --- | --- | --- | --- |']
    lines += [f"| {r['initialization']} | {r['updates']} | {r['family']} | {r['masked_mse']:.5f} | {r['mean_fill_mse']:.5f} |" for r in rows]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report


def inspect_common_grids(session_path, *, output_dir, max_grids=5, training_steps=20, batch_size=16, cache_dir=None, device="cpu"):
    """Feed the identical recorded grids to each strategy, ignoring their chosen paths."""
    session_path, directory = Path(session_path), Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    data = json.loads(session_path.read_text())
    events = [e for e in data['events'] if e['action'] == 'select'][:max_grids]
    if len(events) < 2:
        raise ValueError('Common-grid inspection needs at least two recorded grids.')
    records = {g['key']: g for g in data['genomes']}
    rows = []
    for condition in ('novelty', 'random', 'imagenet'):
        strategy = (NoveltySelectionStrategy() if condition == 'novelty' else
                    NoveltyPredictabilitySelectionStrategy(observer_initialization=condition,
                        training_steps=training_steps, batch_size=batch_size, cache_dir=cache_dir, device=device))
        rng = Random(data['metadata']['settings']['selection_seed'])
        for index, event in enumerate(events):
            images = []
            for key in event['displayed']:
                with Image.open(session_path.parent / records[key]['image']) as source:
                    images.append(np.array(source.convert('RGB')))
            decision = strategy.choose(images, rng=rng)
            rows.append({'condition': condition, 'decision': index + 1, 'position': decision.position,
                         'scores': decision.scores.tolist(), 'values': decision.evaluation.values.tolist(),
                         'names': decision.evaluation.names, 'metadata': decision.evaluation.metadata})
    report = {'source_session': str(session_path.resolve()),
              'source_sha256': hashlib.sha256(session_path.read_bytes()).hexdigest(),
              'grid_count': len(events), 'device': device, 'rows': rows,
              'interpretation': 'Identical exposure, not independent breeding trajectories. Choices do not determine subsequent grids.'}
    write_json(directory / 'common-grids.json', report)
    lines = ['# Common-grid score inspection', '', report['interpretation'], '',
             '| Grid | Novelty choice | Random-observer choice | Pretrained-observer choice |', '| --- | --- | --- | --- |']
    for index in range(len(events)):
        choices = [r['position'] + 1 for r in rows if r['decision'] == index + 1]
        lines.append(f"| {index + 1} | {' | '.join(map(str, choices))} |")
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report
