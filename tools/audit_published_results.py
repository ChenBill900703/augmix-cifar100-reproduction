"""Audit published CSV/JSON evidence using only Python's standard library.

This does not claim to re-run inference or validate unavailable model bytes.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
CORRUPTIONS=['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression']
METRICS=['clean_top1','clean_top5','corruption_top1','corruption_top5']

def read(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def close(a,b):
    if not math.isclose(float(a),float(b),abs_tol=1e-9,rel_tol=1e-10): raise AssertionError((a,b))

def audit():
    protocol=json.loads((ROOT/'results/formal_protocol.json').read_text())
    for name,expected in protocol['code'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    expected_rows={}
    manifest={(x['run'],x['checkpoint']):x for x in json.loads((ROOT/'checkpoint_manifest.json').read_text())}
    total_epochs=total_eval=0
    for seed in range(3):
        for method in ['baseline','augmix']:
            name=f'{method}_seed{seed}'; run=ROOT/'experiment_records'/name
            epochs=read(run/'epochs.csv'); ev=read(run/'evaluation.csv')
            assert [int(r['epoch']) for r in epochs]==list(range(1,101))
            assert len(ev)==152
            assert {(r['checkpoint'],r['corruption'],int(r['severity'])) for r in ev}=={(cp,c,s) for cp in ['best','last'] for c,s in [('clean',0)]+[(c,s) for c in CORRUPTIONS for s in range(1,6)]}
            assert all(int(r['n'])==10000 and 0<=float(r['top5_error'])<=float(r['top1_error'])<=100 for r in ev)
            assert all(math.isfinite(float(r[k])) for r in epochs for k in ['train_loss','ce','jsd','clean_loss','clean_top1','clean_top5'])
            assert all(float(r['reserved_GiB'])<=7.199560546875 for r in epochs)
            identity=json.loads((run/'evaluation_identity.json').read_text())
            for cp in ['best','last']:
                clean=next(r for r in ev if r['checkpoint']==cp and r['corruption']=='clean')
                epoch=next(r for r in epochs if int(r['epoch'])==int(clean['epoch']))
                close(clean['top1_error'],epoch['clean_top1']); close(clean['top5_error'],epoch['clean_top5'])
                assert identity[cp]==manifest[(name,cp)]['sha256']
                if cp=='best': assert int(clean['epoch'])==int(min(epochs,key=lambda r:float(r['clean_top1']))['epoch'])
                else: assert int(clean['epoch'])==100
                cs=[r for r in ev if r['checkpoint']==cp and r['corruption']!='clean']
                expected_rows[(method,seed,cp)]=dict(clean_top1=float(clean['top1_error']),clean_top5=float(clean['top5_error']),corruption_top1=statistics.mean(float(r['top1_error']) for r in cs),corruption_top5=statistics.mean(float(r['top5_error']) for r in cs))
            total_epochs+=len(epochs); total_eval+=len(ev)
    summaries=read(ROOT/'results/per_seed.csv'); assert len(summaries)==12
    for row in summaries:
        expected=expected_rows[(row['method'],int(row['seed']),row['checkpoint'])]
        for k in METRICS: close(row[k],expected[k])
    stats=json.loads((ROOT/'results/statistics.json').read_text())
    for row in stats['summary']:
        values=[expected_rows[(row['method'],s,row['checkpoint'])][row['metric']] for s in range(3)]
        assert row['n']==3; close(row['mean'],statistics.mean(values)); close(row['sample_sd'],statistics.stdev(values))
    for row in stats['paired_improvement_pp']:
        for k in METRICS: close(row[k],expected_rows[('baseline',row['seed'],row['checkpoint'])][k]-expected_rows[('augmix',row['seed'],row['checkpoint'])][k])
    assert total_epochs==600 and total_eval==912
    print(f'PASS: {total_epochs} epochs, {total_eval} evaluation rows, 12 clean-checkpoint matches, summary means/SD and paired improvements.')
    for method in ['baseline','augmix']:
        values=[expected_rows[(method,s,'last')]['corruption_top1'] for s in range(3)]
        print(f'{method}: CIFAR-100-C Top-1 error = {statistics.mean(values):.6f}% +/- {statistics.stdev(values):.6f}% (sample SD, n=3)')
    print('Scope: published record consistency only; weights/data are not included and inference was not rerun.')

if __name__=='__main__': audit()
