"""Comparison reports include every scheduled run and preserve metric meanings."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from automated_picbreeder.selection_comparison import summarize_run, write_comparison_report, run_comparison


def fixture_run(path):
    path.mkdir(parents=True)
    for key, value in enumerate((0, 255)):
        Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(path / f'{key}.png')
    # Earlier score is higher: the final image must still be the reported endpoint.
    events = [
        {'action': 'select', 'position': 0, 'genome': 0, 'displayed': [0] * 9,
         'decision': {'mode': 'random', 'scores': [1] * 9},
         'evaluation': {'names': ['pixel_novelty', 'comprehension'], 'values': [[0, 0]] * 9,
                        'metadata': {'reference_available': False, 'comprehension_available': False,
                                     'seen_before': [False] * 9, 'masked_mse': [None] * 9}}},
        {'action': 'select', 'position': 1, 'genome': 1, 'displayed': [0, 1] + [0] * 7,
         'decision': {'mode': 'greedy', 'scores': [.5] * 9},
         'evaluation': {'names': ['pixel_novelty', 'comprehension'], 'values': [[.5, .9]] * 9,
                        'metadata': {'reference_available': True, 'comprehension_available': True,
                                     'seen_before': [True, False] + [True] * 7,
                                     'masked_mse': [.01, .4] + [.01] * 7}}},
    ]
    data = {'selected': 1, 'genomes': [{'key': i, 'image': f'{i % 2}.png'} for i in range(17)],
            'events': events, 'summary': {'status': 'complete', 'elapsed_seconds': 1, 'decisions': 2,
            'candidate_presentations': 18, 'unique_candidates': 17}}
    (path / 'session.json').write_text(json.dumps(data))


def test_summary_excludes_unavailable_and_repeated_prediction_errors(tmp_path):
    fixture_run(tmp_path / 'run')
    result = summarize_run(tmp_path / 'run', expected_steps=2)
    assert result['final_image'] == '1.png'
    assert result['mean_selected_novelty'] == .5
    assert result['mean_fresh_masked_mse'] == .4
    assert result['selected_pairwise_pixel_mse'] == 1
    assert result['parent_retention_rate'] == 0
    assert result['trajectory'][0]['fresh_masked_mse'] is None
    assert result['candidate_presentations'] == 18


def test_one_decision_has_no_invented_prediction_or_retention_measure(tmp_path):
    fixture_run(tmp_path / 'run')
    path = tmp_path / 'run/session.json'
    data = json.loads(path.read_text())
    data['events'] = data['events'][:1]
    data['genomes'] = data['genomes'][:9]
    data['selected'] = 0
    data['summary'].update(decisions=1, candidate_presentations=9, unique_candidates=9)
    path.write_text(json.dumps(data))
    result = summarize_run(path.parent, expected_steps=1)
    assert result['mean_fresh_masked_mse'] is None
    assert result['mean_selected_novelty'] is None
    assert result['parent_retention_rate'] is None
    assert result['selected_pairwise_pixel_mse'] == 0


def test_report_keeps_failed_and_missing_runs_and_uses_final_images(tmp_path):
    fixture_run(tmp_path / 'novelty-seed7')
    manifest = {'settings': {'steps': 2}, 'runs': [
        {'condition': 'novelty', 'seed': 7, 'path': 'novelty-seed7', 'status': 'complete'},
        {'condition': 'predictability-random', 'seed': 7, 'path': 'missing', 'status': 'failed', 'error': 'test failure'},
        {'condition': 'predictability-imagenet', 'seed': 7, 'path': 'pending', 'status': 'pending'},
    ]}
    (tmp_path / 'comparison.json').write_text(json.dumps(manifest))
    report = write_comparison_report(tmp_path)
    assert len(report['runs']) == 3
    assert [r['status'] for r in report['runs']] == ['complete', 'failed', 'pending']
    assert report['runs'][0]['final_image'] == 'novelty-seed7/1.png'
    assert 'test failure' in (tmp_path / 'REPORT.md').read_text()
    assert (tmp_path / 'final-images.png').exists()
    assert (tmp_path / 'trajectories.csv').exists()


def test_bad_counts_are_reported_as_failed_not_silently_included(tmp_path):
    fixture_run(tmp_path / 'run')
    with pytest.raises(ValueError, match='count'):
        summarize_run(tmp_path / 'run', expected_steps=3)


def test_batch_uses_fresh_strategies_and_shared_settings_even_after_failure(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    calls = []
    def runner(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError('deliberate failure')
        fixture_run(Path(kwargs['output_dir']))
    monkeypatch.setattr('automated_picbreeder.selection_comparison.run_experiment', runner)
    result = run_comparison(output_dir=tmp_path / 'batch', seeds=(7,), steps=2, progress=None)
    assert len(calls) == 3
    assert len({id(c['selection_strategy']) for c in calls}) == 3
    assert len({c['settings'] for c in calls}) == 1
    assert [r['status'] for r in result['runs']] == ['complete', 'failed', 'complete']
    with pytest.raises(FileExistsError):
        run_comparison(output_dir=tmp_path / 'batch', progress=None)


def test_interruption_leaves_an_explicit_partial_report(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    def interrupt(**kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr('automated_picbreeder.selection_comparison.run_experiment', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_comparison(output_dir=tmp_path / 'batch', seeds=(7,), progress=None)
    report = json.loads((tmp_path / 'batch/report.json').read_text())
    assert report['status'] == 'incomplete'
    assert [r['status'] for r in report['runs']] == ['interrupted', 'pending', 'pending']


def test_comparison_cli_dispatches_defaults_and_creates_no_extra_strategies(monkeypatch, tmp_path):
    import runpy
    from unittest.mock import Mock
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'experiments/compare_selection.py'))['main']
    runner = Mock(return_value={'status': 'complete'})
    monkeypatch.setitem(main.__globals__, 'run_comparison', runner)
    main(['--output', str(tmp_path / 'batch')])
    args = runner.call_args.kwargs
    assert args['seeds'] == [7, 8, 9] and args['steps'] == 100 and args['size'] == 96
    assert args['training_steps'] == 20 and args['batch_size'] == 16
    assert args['comprehension_weight'] == .5
    assert args['device'] == 'cpu'
    main(['--output', str(tmp_path / 'mps'), '--device', 'mps'])
    assert runner.call_args.kwargs['device'] == 'mps'
