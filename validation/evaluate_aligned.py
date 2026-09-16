"""Re-evaluate unchanged checkpoints with the training backend settings.

Initial evaluations are retained in each run/evaluation_initial_default_backend.
No training source, configuration, checkpoint selection, or weights are changed.
"""
import ctypes
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import evaluate_run, save_json
import torch

def main():
    kernel=ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
    kernel.SetThreadExecutionState.restype=ctypes.c_uint
    if not kernel.SetThreadExecutionState(0x80000001): raise OSError('Awake request failed')
    faulthandler.enable(); faulthandler.dump_traceback_later(600,repeat=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    settings=dict(cudnn_benchmark=False,cudnn_deterministic=True,cudnn_tf32=False,matmul_tf32=False,threads=4,amp=False)
    progress=json.loads((ROOT/'progress.json').read_text(encoding='utf-8'))
    progress.pop('traceback',None); progress.pop('child_pid',None)
    progress['controller_pid']=os.getpid()
    try:
        for seed in range(3):
            for method in ['baseline','augmix']:
                run=ROOT/'runs'/f'{method}_seed{seed}'
                archive=run/'evaluation_initial_default_backend'
                if not archive.exists():
                    archive.mkdir()
                    for name in ['evaluation.csv','evaluation.partial.csv','evaluation_identity.json']:
                        path=run/name
                        if path.exists(): shutil.move(str(path),str(archive/name))
                save_json(run/'evaluation_backend.json',settings)
                progress.update(stage=f'eval_aligned_{method}_seed{seed}',updated=time.time(),remaining_hours=(progress['deadline']-time.time())/3600)
                save_json(ROOT/'progress.json',progress)
                if not (run/'evaluation.csv').exists(): evaluate_run(run)
                faulthandler.cancel_dump_traceback_later(); faulthandler.dump_traceback_later(600,repeat=True)
                rows=list(__import__('csv').DictReader((run/'evaluation.csv').open()))
                assert len(rows)==152
                for name in ['best','last']:
                    ck=torch.load(run/f'{name}.pt',map_location='cpu',weights_only=False)
                    epochrow=ck['rows'][-1]
                    actual=next(r for r in rows if r['checkpoint']==name and r['corruption']=='clean')
                    assert float(actual['top1_error'])==epochrow['clean_top1'],(run,name,actual,epochrow)
                    assert float(actual['top5_error'])==epochrow['clean_top5']
        progress.update(stage='evaluation_complete_report_pending',updated=time.time(),evaluated_runs=6)
        save_json(ROOT/'progress.json',progress)
        subprocess.run([sys.executable,str(ROOT/'report.py')],cwd=ROOT,check=True)
        save_json(ROOT/'results/evaluation_alignment_verified.json',dict(all_12_clean_top1_top5_match_epoch_records=True,settings=settings,completed=time.time()))
    except BaseException:
        progress.update(stage='aligned_evaluation_error',updated=time.time(),traceback=traceback.format_exc())
        save_json(ROOT/'progress.json',progress)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        kernel.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
