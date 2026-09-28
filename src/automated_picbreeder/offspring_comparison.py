"""Experiment 2: two complete strategies, factual forecast errors, all outcomes."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .experiment import ExperimentSettings, run_experiment
from .persistence import write_json
from .selection_comparison import summarize_run
from .selection_strategies import OffspringValueSelectionStrategy

CONDITIONS = {'current-image': 0., 'offspring-value': 1.}


def _prediction_metrics(rows):
    result = {'count': len(rows)}
    for name in ('predictor', 'running_mean', 'current_value'):
        errors = [r[f'{name}_forecast'] - r['target'] for r in rows]
        result[f'{name}_mse'] = float(np.mean(np.square(errors))) if errors else None
        result[f'{name}_mae'] = float(np.mean(np.abs(errors))) if errors else None
    result['target_variance'] = float(np.var([r['target'] for r in rows])) if rows else None
    return result


def summarize_offspring_run(directory, *, expected_steps=None):
    """Use original forecasts; missing outcomes are never zero-valued observations."""
    directory = Path(directory)
    result = summarize_run(directory, expected_steps=expected_steps)
    data = json.loads((directory/'session.json').read_text())
    events = [e for e in data['events'] if e['action'] == 'select']
    genomes = {g['key']: g for g in data['genomes']}
    forecasts, boundaries = [], []
    costs = {k: 0. for k in ('target_scoring_seconds', 'predictor_training_seconds', 'inference_seconds', 'snapshot_seconds')}
    updates = samples = target_inputs = used = changed = 0
    for t, event in enumerate(events):
        meta = event['evaluation']['metadata']['offspring']
        if t:
            parent = events[t-1]['genome']
            if event['displayed'][0] != parent or any(genomes[k]['parent'] != parent for k in event['displayed'][1:]):
                raise ValueError('Recorded offspring ancestry does not match the selected parent.')
        for key in costs: costs[key] += meta[key]
        training = meta['training'] or {}
        updates += training.get('updates', 0)
        samples += training.get('sampled_examples', 0)
        target_inputs += meta['target_masked_inputs_scored']
        used += meta['forecast_used']
        changed += meta['choice_changed']
        result['trajectory'][t].update(forecast_used=meta['forecast_used'], choice_changed=meta['choice_changed'],
            predictor_training_loss=training.get('mean_loss'), completed_targets=meta['completed_targets'])
        feedback = meta['feedback']
        if (feedback is None) != (t < 2):
            raise ValueError('Missing or unexpected completed offspring target.')
        if feedback is None:
            continue
        origin = t-1
        pending = events[origin]['evaluation']['metadata']['offspring']['pending']
        keys = ('forecast', 'running_mean_forecast', 'current_value', 'parent_image_hash',
                'predictor_state_hash', 'completed_targets_at_forecast', 'selected_position')
        if (feedback['origin_generation'] != origin or pending is None or
                any(feedback[k] != pending[k] for k in keys)):
            raise ValueError('Delayed forecast does not match its original prediction record.')
        values = np.asarray(feedback['child_values'], dtype=float)
        if values.shape != (8,5) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Expected eight finite child measurement rows in [0,1].')
        target = float(values[:, -1].mean())
        if target != feedback['target']:
            raise ValueError('Target is not the mean of eight child values.')
        prior_targets = [r['target'] for r in forecasts]
        running_mean = float(np.mean(prior_targets)) if prior_targets else .5
        if pending['running_mean_forecast'] != running_mean or pending['completed_targets_at_forecast'] != len(forecasts):
            raise ValueError('Running-mean baseline includes incorrect or future targets.')
        boundary = float(((values[:,2:4] == 0) | (values[:,2:4] == 1)).mean())
        boundaries.append(boundary)
        after = pending['completed_targets_at_forecast'] >= meta['warmup_targets']
        row = {'origin_generation': origin, 'received_generation': t, 'parent_id': events[origin]['genome'],
            'completed_targets_at_forecast': pending['completed_targets_at_forecast'],
            'phase': 'after_warmup' if after else 'warmup', 'forecast_used': pending['forecast_used'],
            'predictor_forecast': pending['forecast'], 'running_mean_forecast': pending['running_mean_forecast'],
            'current_value_forecast': pending['current_value'], 'target': target, 'rank_boundary_fraction': boundary}
        for name in ('predictor', 'running_mean', 'current_value'):
            if not np.isfinite(row[f'{name}_forecast']) or not 0 <= row[f'{name}_forecast'] <= 1:
                raise ValueError('Invalid original forecast.')
            row[f'{name}_error'] = row[f'{name}_forecast'] - target
        forecasts.append(row)
    if len(forecasts) != max(len(events)-2, 0):
        raise ValueError('Unexpected number of completed targets.')
    windows = []
    for start in range(0, len(forecasts), 10):
        window = forecasts[start:start+10]
        windows.append({'first_origin': window[0]['origin_generation'], 'last_origin': window[-1]['origin_generation'],
                        **_prediction_metrics(window)})
    result.update(completed_targets=len(forecasts), first_transition_ineligible=int(len(events)>=2),
        final_selection_unobserved=int(bool(events)), forecasts=forecasts, prediction_windows=windows,
        prediction={'all': _prediction_metrics(forecasts),
                    'warmup': _prediction_metrics([r for r in forecasts if r['phase']=='warmup']),
                    'after_warmup': _prediction_metrics([r for r in forecasts if r['phase']=='after_warmup']),
                    'first_20': _prediction_metrics(forecasts[:20]), 'last_20': _prediction_metrics(forecasts[-20:])},
        rank_boundary_fraction=float(np.mean(boundaries)) if boundaries else None,
        forecast_used_decisions=used, changed_choices=changed,
        changed_choice_rate=changed/used if used else None, predictor_updates=updates,
        predictor_training_examples=samples, target_masked_inputs_scored=target_inputs, **costs)
    return result


def _csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _show(value):
    return '—' if value is None else f'{value:.5g}'


def write_offspring_report(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'comparison.json').read_text())
    rows, forecasts, trajectories, windows = [], [], [], []
    for entry in manifest['runs']:
        row = dict(entry)
        session = directory/entry['path']/'session.json'
        if session.exists():
            try:
                metrics = summarize_offspring_run(session.parent,
                    expected_steps=manifest['settings']['steps'] if entry['status']=='complete' else None)
                prefix = {'condition':entry['condition'], 'seed':entry['seed']}
                forecasts.extend(prefix | point for point in metrics.pop('forecasts'))
                trajectories.extend(prefix | point for point in metrics.pop('trajectory'))
                windows.extend(prefix | point for point in metrics.pop('prediction_windows'))
                if metrics['final_image']:
                    metrics['final_image'] = str(Path(entry['path'])/metrics['final_image'])
                row.update(metrics)
            except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
                row.update(status='failed', error=f'Report validation: {exc}')
        elif entry['status']=='complete':
            row.update(status='failed', error='Completed run missing session.json')
        rows.append(row)
    report = {'status':'complete' if rows and all(r['status']=='complete' for r in rows) else 'incomplete',
              'settings':manifest['settings'], 'strategy_settings':manifest['strategy_settings'], 'runs':rows}
    write_json(directory/'report.json', report)
    _csv(directory/'forecasts.csv', forecasts, ['condition','seed','origin_generation','received_generation','parent_id',
        'completed_targets_at_forecast','phase','forecast_used','predictor_forecast','running_mean_forecast',
        'current_value_forecast','target','rank_boundary_fraction','predictor_error','running_mean_error','current_value_error'])
    _csv(directory/'trajectories.csv', trajectories, ['condition','seed','decision','selected_id','selected_novelty',
        'fresh_masked_mse','fresh_error_count','selected_seen_before','retained_parent','forecast_used','choice_changed',
        'predictor_training_loss','completed_targets'])
    _csv(directory/'prediction-windows.csv', windows, ['condition','seed','first_origin','last_origin','count',
        'predictor_mse','predictor_mae','running_mean_mse','running_mean_mae','current_value_mse','current_value_mae','target_variance'])
    seeds = list(dict.fromkeys(r['seed'] for r in rows))
    sheet = Image.new('RGB',(480,245*max(1,len(seeds))),'white')
    draw = ImageDraw.Draw(sheet)
    for row in rows:
        x,y = list(CONDITIONS).index(row['condition'])*240, seeds.index(row['seed'])*245
        draw.text((x+8,y+6), f"{row['condition']} / seed {row['seed']}",fill='black')
        draw.text((x+8,y+23),row['status'],fill='black')
        if row.get('final_image'):
            with Image.open(directory/row['final_image']) as image:
                sheet.paste(image.convert('RGB').resize((180,180)),(x+30,y+48))
            points=[p for p in trajectories if p['condition']==row['condition'] and p['seed']==row['seed']]
            data=json.loads((directory/row['path']/'session.json').read_text())
            genomes={g['key']:g for g in data['genomes']}
            trajectory=Image.new('RGB',(1000,116*max(1,(len(points)+9)//10)),'white')
            for i,point in enumerate(points):
                tx,ty=(i%10)*100,(i//10)*116
                with Image.open(directory/row['path']/genomes[point['selected_id']]['image']) as image:
                    trajectory.paste(image.convert('RGB').resize((96,96)),(tx+2,ty+18))
                ImageDraw.Draw(trajectory).text((tx+2,ty+2),f"{point['decision']} / #{point['selected_id']}",fill='black')
            trajectory.save(directory/f"trajectory-{row['condition']}-seed{row['seed']}.png")
        else:
            draw.text((x+12,y+110),'No selected image',fill='gray')
    sheet.save(directory/'final-images.png')
    lines=['# Experiment 2 comparison','',f"Status: {report['status']}. Every scheduled run is shown.",'',
        '[Configuration](comparison.json) · [All metrics](report.json) · [Original forecasts](forecasts.csv) · '
        '[Per-decision measurements](trajectories.csv) · [Ten-outcome error windows](prediction-windows.csv)','',
        '![All final selected images](final-images.png)','',
        '## Prediction after the initial learning period','',
        'Errors compare the original selected-parent forecast with its later eight-child mean, before training '
        'on that outcome. Phase membership uses the number of targets available at forecast time in both arms; '
        'gamma zero is still separated into early and later forecasts. The running mean uses only previously '
        'observed targets (0.5 initially). The other baseline predicts the selected parent’s current value.','',
        '| Condition | Seed | Status | Later targets | Predictor MSE | Mean baseline MSE | Parent value MSE | Predictor MAE |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        m=row.get('prediction',{}).get('after_warmup',{})
        lines.append(f"| [{row['condition']}]({row['path']}/session.json) | {row['seed']} | {row['status']} | "
            f"{m.get('count',0)} | {_show(m.get('predictor_mse'))} | {_show(m.get('running_mean_mse'))} | "
            f"{_show(m.get('current_value_mse'))} | {_show(m.get('predictor_mae'))} |")
    lines += ['', '### All periods and baselines', '',
        '| Condition | Seed | Period | N | Predictor MSE / MAE | Mean baseline MSE / MAE | Parent value MSE / MAE | Target variance |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        for phase in ('all','warmup','after_warmup'):
            m=row.get('prediction',{}).get(phase,{})
            pairs=[f"{_show(m.get(name+'_mse'))} / {_show(m.get(name+'_mae'))}" for name in ('predictor','running_mean','current_value')]
            lines.append(f"| {row['condition']} | {row['seed']} | {phase} | {m.get('count',0)} | " +
                         ' | '.join(pairs) + f" | {_show(m.get('target_variance'))} |")
    lines += ['', '## Error over time', '',
        'First and last 20 observed outcomes are descriptive windows; they overlap in runs with fewer than 40 '
        'outcomes. Ten-outcome windows below preserve the full time course. Target difficulty and selection '
        'also change over time, so falling error alone does not isolate learning.', '',
        '| Condition | Seed | First 20 MSE | Last 20 MSE | Last 20 mean baseline | Last 20 target variance |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        early=row.get('prediction',{}).get('first_20',{});late=row.get('prediction',{}).get('last_20',{})
        lines.append(f"| {row['condition']} | {row['seed']} | {_show(early.get('predictor_mse'))} | "
            f"{_show(late.get('predictor_mse'))} | {_show(late.get('running_mean_mse'))} | {_show(late.get('target_variance'))} |")
    lines += ['', '| Condition | Seed | Origin generations (zero-based) | N | Predictor MSE | Mean baseline MSE | Parent value MSE |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for w in windows:
        lines.append(f"| {w['condition']} | {w['seed']} | {w['first_origin']}–{w['last_origin']} | {w['count']} | "
            f"{_show(w['predictor_mse'])} | {_show(w['running_mean_mse'])} | {_show(w['current_value_mse'])} |")
    lines += ['', '## Selection and exploration', '',
        '| Condition | Seed | Changed choices / enabled | Repeat rate | Retained parent rate | Pixel diversity | Rank boundary fraction |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {r.get('changed_choices',0)} / {r.get('forecast_used_decisions',0)} | "
            f"{_show(r.get('selected_repeat_rate'))} | {_show(r.get('parent_retention_rate'))} | "
            f"{_show(r.get('selected_pairwise_pixel_mse'))} | {_show(r.get('rank_boundary_fraction'))} |")
    lines += ['', 'Changed choices compares the actual choice with current-image selection on that same grid. '
        'Repeat rates exclude the first decision and include images previously displayed but rejected. '
        'Pixel diversity is mean squared distance over all pairs of selected images, duplicates included. '
        'Boundary fraction counts child novelty/comprehension ranks at 0 or 1. These are diagnostics, '
        'not independent measurements of interestingness.', '', '## Counts and costs', '',
        '| Condition | Seed | Presentations | Genomes | Targets | Seconds | Observer score/train | Target score | Predictor infer/train | Snapshot |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {r.get('candidate_presentations',0)} | {r.get('genomes',0)} | "
            f"{r.get('completed_targets',0)} | {_show(r.get('elapsed_seconds'))} | "
            f"{_show(r.get('scoring_seconds'))} / {_show(r.get('training_seconds'))} | {_show(r.get('target_scoring_seconds'))} | "
            f"{_show(r.get('inference_seconds'))} / {_show(r.get('predictor_training_seconds'))} | {_show(r.get('snapshot_seconds'))} |")
    lines += ['', 'Each completed S-decision run has 9*S presentations, 9+8*(S-1) genomes and max(S-2,0) targets. '
        'The first transition is ineligible; the final selected parent has no observed brood. Neither receives '
        'an invented target. Extra target/predictor computation does not generate extra offspring. Timed components '
        'exclude some initialization, hashing, rendering and saving overhead; total seconds cover the run.', '',
        '## Full selected trajectories', '']
    for r in rows:
        if r.get('final_image'):
            lines += [f"### {r['condition']} / seed {r['seed']} ({r['status']})",'',
                f"![Every selection](trajectory-{r['condition']}-seed{r['seed']}.png)",'',
                f"[Session and ancestry]({r['path']}/session.json); all nine-candidate grids: `{r['path']}/grids/`.",'']
        if r.get('error'): lines += [f"- {r['condition']} / seed {r['seed']}: {r['error']}",'']
    lines += ['## Limits','', 'Only selected parents reveal offspring outcomes. These errors do not establish correct '
        'rankings for rejected alternatives. Independent conditions see different images after their choices diverge. '
        'Targets use changing selection-time contexts and are not absolute interestingness scores. Lower error or '
        'different images do not establish creativity, human preference, evolvability or UFR. All endpoints are '
        'the final selection, never a higher-scoring earlier image; failed runs show their last available endpoint.']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def run_offspring_comparison(*, output_dir, seeds=(7,8,9), steps=100, size=96, mutation_strength=.2,
        topology=True, comprehension_weight=.5, training_steps=20, batch_size=16, learning_rate=.001,
        warmup_targets=10, predictor_training_steps=10, predictor_batch_size=16, predictor_learning_rate=.001,
        device='cpu', progress=print):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Supply distinct, nonempty seeds.')
    settings=ExperimentSettings(seed=seeds[0], steps=steps, size=size, mutation_strength=mutation_strength, topology=topology)
    all_settings={seed:replace(settings,seed=seed) for seed in seeds}
    options=dict(comprehension_weight=comprehension_weight,training_steps=training_steps,batch_size=batch_size,
        learning_rate=learning_rate,warmup_targets=warmup_targets,predictor_training_steps=predictor_training_steps,
        predictor_batch_size=predictor_batch_size,predictor_learning_rate=predictor_learning_rate,device=device)
    OffspringValueSelectionStrategy(**options)  # Validate before creating output; no weights initialized.
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=False)
    manifest={'settings':{k:v for k,v in asdict(settings).items() if k not in ('seed','selection_seed')},
        'strategy_settings':options,'seeds':list(seeds),
        'runs':[{'condition':condition,'gamma':gamma,'seed':seed,'path':f'{condition}-seed{seed}','status':'pending'}
                for seed in seeds for condition,gamma in CONDITIONS.items()]}
    write_json(directory/'comparison.json',manifest)
    try:
        for i,entry in enumerate(manifest['runs']):
            entry['status']='running';write_json(directory/'comparison.json',manifest)
            if progress: progress(f"Run {i+1}/{len(manifest['runs'])}: {entry['condition']}, seed {entry['seed']}")
            try:
                strategy=OffspringValueSelectionStrategy(gamma=entry['gamma'],**options)
                run_experiment(selection_strategy=strategy,output_dir=directory/entry['path'],
                               settings=all_settings[entry['seed']],progress=progress)
                entry['status']='complete'
            except KeyboardInterrupt:
                entry.update(status='interrupted',error='Interrupted; existing artifacts retained.')
                write_json(directory/'comparison.json',manifest)
                raise
            except Exception as exc:
                entry.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                if progress: progress(entry['error'])
            write_json(directory/'comparison.json',manifest)
    finally:
        report=write_offspring_report(directory)
    return report
