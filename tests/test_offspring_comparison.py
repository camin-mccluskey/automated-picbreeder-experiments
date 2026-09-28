"""Every scheduled outcome and genuinely pre-outcome forecasts enter the report."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from automated_picbreeder.offspring_comparison import summarize_offspring_run, write_offspring_report, run_offspring_comparison


def fixture_run(directory, steps=5):
    directory.mkdir(parents=True)
    for i in (0, 1):
        Image.fromarray(np.full((2,2,3), i*255, np.uint8)).save(directory/f'{i}.png')
    records = [{'key': k, 'image': f'{k%2}.png', 'parent': None} for k in range(9+8*(steps-1))]
    events, targets = [], []
    for t in range(steps):
        ids = list(range(9)) if t == 0 else [events[-1]['genome']] + list(range(9+8*(t-1), 9+8*t))
        if t:
            for k in ids[1:]: records[k]['parent'] = events[-1]['genome']
        # Repeat the retained parent: every transition still has a separate target.
        selected = ids[0]
        feedback = None
        if t >= 2:
            pending = events[t-1]['evaluation']['metadata']['offspring']['pending']
            target = (.4,.6,.3)[(t-2)%3]
            feedback = pending | {'received_generation': t, 'target': target,
                'child_names': ['pixel_novelty','comprehension','novelty_rank','comprehension_rank','fixed_reference_value'],
                'child_values': [[.1,.9,0,1,target]]*8,
                'target_context': {'origin_generation': t-1},
                # Deliberately incorrect cached errors: derive from original forecasts.
                'squared_error': 999, 'absolute_error': 999}
            targets.append(target)
        count = max(t-1,0)
        pending = None if t == 0 else {'origin_generation': t, 'selected_position': 0,
            'forecast': (.2,.8,.4,.7)[(t-1)%4], 'current_value': .5,
            'running_mean_forecast': float(np.mean(targets)) if targets else .5,
            'completed_targets_at_forecast': count, 'parent_image_hash': f'h{selected}',
            'predictor_state_hash': f'm{t}', 'forecast_used': count >= 2}
        meta = {'reference_available': t>0, 'comprehension_available': t>0,
                'seen_before': [t>0]*9, 'masked_mse': [.1]*9,
                'offspring': {'feedback': feedback, 'pending': pending, 'warmup_targets': 2,
                    'forecast_used': t>=3, 'choice_changed': False, 'completed_targets': count,
                    'training': {'updates': 10, 'sampled_examples': 160, 'mean_loss': .1} if feedback else None,
                    'target_masked_inputs_scored': 128 if feedback else 0,
                    'target_scoring_seconds': .1 if feedback else 0, 'predictor_training_seconds': .2 if feedback else 0,
                    'inference_seconds': .01, 'snapshot_seconds': .01}}
        events.append({'action':'select','position':0,'genome':selected,'displayed':ids,
            'decision':{'scores':[.5]*9,'mode':'greedy' if t else 'random'},
            'evaluation':{'names':['pixel_novelty','comprehension','current_value','predicted_offspring_value'],
                          'values':[[.1,.9,.5,.2]]*9,'metadata':meta}})
    data={'selected':events[-1]['genome'], 'genomes':records, 'events':events,
          'summary':{'status':'complete','decisions':steps,'candidate_presentations':9*steps,
                     'unique_candidates':len(records),'elapsed_seconds':1}}
    (directory/'session.json').write_text(json.dumps(data))


def test_original_forecasts_baselines_phases_repeats_and_missing_targets(tmp_path):
    fixture_run(tmp_path/'run')
    result=summarize_offspring_run(tmp_path/'run',expected_steps=5)
    assert result['completed_targets']==3
    assert result['first_transition_ineligible']==1 and result['final_selection_unobserved']==1
    assert result['prediction']['all']['predictor_mse']==pytest.approx((.2**2+.2**2+.1**2)/3)
    assert result['prediction']['warmup']['count']==2
    assert result['prediction']['after_warmup']['count']==1
    assert result['prediction']['all']['running_mean_mse']==pytest.approx((.1**2+.2**2+.2**2)/3)
    assert result['rank_boundary_fraction']==1
    assert result['selected_repeat_rate']==1
    assert len(result['forecasts'])==3  # Do not deduplicate the identical parent.
    assert result['forecasts'][0]['origin_generation']==1
    assert result['predictor_updates']==30
    assert result['final_image']=='0.png'


@pytest.mark.parametrize('steps',[1,2])
def test_no_invented_targets_before_first_valid_transition(tmp_path,steps):
    fixture_run(tmp_path/'run',steps)
    result=summarize_offspring_run(tmp_path/'run',expected_steps=steps)
    assert result['completed_targets']==0 and result['prediction']['all']['predictor_mse'] is None
    assert result['forecasts']==[]


def test_mismatched_forecast_and_missing_label_rejected(tmp_path):
    fixture_run(tmp_path/'run')
    p=tmp_path/'run/session.json'; original=json.loads(p.read_text())
    data=json.loads(p.read_text());data['events'][2]['evaluation']['metadata']['offspring']['feedback']['forecast']=.9
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='forecast'): summarize_offspring_run(p.parent,expected_steps=5)
    original['events'][2]['evaluation']['metadata']['offspring']['feedback']=None
    p.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='target'): summarize_offspring_run(p.parent,expected_steps=5)


def test_all_scheduled_endpoints_failures_and_partial_runs_in_report(tmp_path):
    fixture_run(tmp_path/'control')
    manifest={'settings':{'steps':5},'seeds':[7,8], 'strategy_settings':{},'runs':[
        {'condition':'current-image','gamma':0,'seed':7,'path':'control','status':'complete'},
        {'condition':'offspring-value','gamma':1,'seed':7,'path':'failed','status':'failed','error':'fixture failure'},
        {'condition':'current-image','gamma':0,'seed':8,'path':'missing','status':'complete'},
        {'condition':'offspring-value','gamma':1,'seed':8,'path':'pending','status':'pending'}]}
    (tmp_path/'comparison.json').write_text(json.dumps(manifest))
    report=write_offspring_report(tmp_path)
    assert [r['status'] for r in report['runs']]==['complete','failed','failed','pending']
    assert report['status']=='incomplete' and len(report['runs'])==4
    assert report['runs'][0]['final_image']=='control/0.png'
    assert 'fixture failure' in (tmp_path/'REPORT.md').read_text()
    for name in ('forecasts.csv','trajectories.csv','prediction-windows.csv','final-images.png'):
        assert (tmp_path/name).exists()


def test_batch_is_fresh_matched_and_continues_after_failure(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    calls=[]
    def run(**kw):
        calls.append(kw)
        if len(calls)==2: raise RuntimeError('deliberate')
        fixture_run(Path(kw['output_dir']))
    monkeypatch.setattr('automated_picbreeder.offspring_comparison.run_experiment',run)
    result=run_offspring_comparison(output_dir=tmp_path/'batch',seeds=(7,8),steps=5,progress=None)
    assert len(calls)==4 and len({id(c['selection_strategy']) for c in calls})==4
    assert [c['selection_strategy'].gamma for c in calls]==[0,1,0,1]
    assert calls[0]['settings']==calls[1]['settings']
    assert [r['status'] for r in result['runs']]==['complete','failed','complete','complete']
    with pytest.raises(FileExistsError): run_offspring_comparison(output_dir=tmp_path/'batch',progress=None)


def test_interrupt_retains_manifest_and_partial_report(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    def stop(**kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr('automated_picbreeder.offspring_comparison.run_experiment',stop)
    with pytest.raises(KeyboardInterrupt): run_offspring_comparison(output_dir=tmp_path/'batch',seeds=(7,),progress=None)
    report=json.loads((tmp_path/'batch/report.json').read_text())
    assert [r['status'] for r in report['runs']]==['interrupted','pending']


def test_cli_defaults_and_report_only_do_not_start_extra_runs(tmp_path,monkeypatch):
    import runpy
    from unittest.mock import Mock
    main=runpy.run_path(str(Path(__file__).resolve().parents[1]/'experiments/compare_offspring_value.py'))['main']
    runner=Mock(return_value={'status':'complete'})
    report=Mock(return_value={'status':'complete'})
    monkeypatch.setitem(main.__globals__,'run_offspring_comparison',runner)
    monkeypatch.setitem(main.__globals__,'write_offspring_report',report)
    main(['--output',str(tmp_path/'batch'),'--device','mps'])
    args=runner.call_args.kwargs
    assert args['seeds']==[7,8,9] and args['steps']==100 and args['size']==96
    assert args['device']=='mps' and args['warmup_targets']==10
    assert args['training_steps']==20 and args['predictor_training_steps']==10
    main(['--report-only',str(tmp_path/'batch')])
    assert runner.call_count==1 and report.call_count==1
