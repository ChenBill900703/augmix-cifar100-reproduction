"""Sequential supervised experiment controller. Stops on failures; no automatic tuning."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import shutil
import psutil
from core import ROOT, save_json

START=1789417552.0
DEADLINE=START+168*3600

def progress(stage,**extra):
    save_json(ROOT/'progress.json',dict(stage=stage,updated=time.time(),start=START,deadline=DEADLINE,remaining_hours=max(0,(DEADLINE-time.time())/3600),controller_pid=os.getpid(),**extra))

def execute(args,name):
    with (ROOT/'logs'/f'{name}.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        last_activity=time.time(); last_size=0
        while child.poll() is None:
            if time.time()>DEADLINE:
                stop_child(child); raise TimeoutError('168 hour deadline reached; last completed epoch retained')
            size=(ROOT/'logs'/f'{name}.log').stat().st_size
            if size!=last_size: last_activity=time.time(); last_size=size
            if time.time()-last_activity>3600:
                stop_child(child); raise TimeoutError(f'{name}: no log progress for one hour; last completed epoch retained')
            progress(name,child_pid=child.pid)
            time.sleep(30)
        if child.returncode: raise RuntimeError(f'{name} failed with {child.returncode}; inspect logs/{name}.log')

def stop_child(child):
    try:
        descendants=psutil.Process(child.pid).children(recursive=True)
        for p in reversed(descendants):
            try: p.terminate()
            except psutil.NoSuchProcess: pass
        child.terminate()
    except psutil.NoSuchProcess: pass

def digest_code():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}

def main():
    lock=ROOT/'controller.json'
    if lock.exists():
        old=json.loads(lock.read_text()); pid=old['pid']
        if psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-old['created'])<1: raise RuntimeError('Controller already running')
    save_json(lock,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    try:
        while not (ROOT/'clean_ready.json').exists():
            progress('waiting_verified_clean_data')
            if time.time()>DEADLINE: raise TimeoutError('Data deadline')
            time.sleep(30)
        gate=ROOT/'results/tests_passed.json'
        code=digest_code()
        if not gate.exists() or json.loads(gate.read_text())['code']!=code:
            execute(['-m','pytest','test_core.py','-q'],'tests_full')
            save_json(gate,dict(time=time.time(),code=code))
        pre=json.loads((ROOT/'results/preflight.json').read_text())
        assert all(any(r['method']==m and not r['amp'] and r['pass_limit'] for r in pre) for m in ['baseline','augmix'])
        timings={}; valtime=[]
        for method in ['baseline','augmix']:
            run=ROOT/'runs'/f'timing_{method}'
            execute(['run.py','train','--config',str(ROOT/'configs'/f'{method}.json'),'--run',str(run),'--stop-epoch','5'],f'timing_{method}')
            with (run/'epochs.csv').open() as f: rows=list(csv.DictReader(f))
            assert len(rows)>=5
            # Full epoch includes validation, checkpoint and worker startup; discard first warm-up epoch.
            timings[method]=sum(float(r['epoch_seconds']) for r in rows[1:5])/4
            valtime.extend(float(r['val_seconds']) for r in rows[1:5])
        hours=3*100*sum(timings.values())/3600
        evalhours=12*76*max(valtime)*1.5/3600
        save_json(ROOT/'results/budget.json',dict(stable_epoch_seconds=timings,training_hours_six=hours,evaluation_hours_estimate=evalhours,evaluation_basis='12 checkpoints × 76 sets of 10000 × slowest measured validation × 1.5 IO margin',target_training_hours=120,total_elapsed_hours=(time.time()-START)/3600,within_168_estimate=time.time()+3600*(hours+evalhours)<DEADLINE))
        frozen=ROOT/'results/formal_protocol.json'
        protocol=dict(code=code,configs={m:json.loads((ROOT/'configs'/f'{m}.json').read_text()) for m in ['baseline','augmix']},selection='first strictly lowest clean test top1; primary last, secondary best',seeds=[0,1,2])
        if frozen.exists() and json.loads(frozen.read_text())!=protocol: raise RuntimeError('Formal code/protocol changed; requires a new preserved experiment series')
        save_json(frozen,protocol)
        snapshot=ROOT/'results/formal_source'; snapshot.mkdir(exist_ok=True)
        for p in ROOT.glob('*.py'): shutil.copy2(p,snapshot/p.name)
        shutil.copy2(ROOT/'SOURCE_AUDIT.md',snapshot/'SOURCE_AUDIT.md')
        for seed in range(3):
            if seed>0 and time.time()+100*sum(timings.values())+evalhours*3600>DEADLINE:
                progress('insufficient_time_for_next_pair',next_seed=seed); break
            for method in ['baseline','augmix']:
                c=dict(protocol['configs'][method],seed=seed); cfg=ROOT/'configs'/f'{method}_seed{seed}.json'; save_json(cfg,c)
                execute(['run.py','train','--config',str(cfg),'--run',str(ROOT/'runs'/f'{method}_seed{seed}')],f'{method}_seed{seed}')
                execute(['report.py'],'report')
        # Wait for external corruption curl to finish, then checksum/extract via prepare.py.
        while any(p.info['cmdline'] and any('CIFAR-100-C.tar?download=1' in s for s in p.info['cmdline']) and Path(p.info['cmdline'][0]).name.lower()=='curl.exe' for p in psutil.process_iter(['cmdline'])):
            progress('waiting_corruption_download')
            if time.time()>DEADLINE: raise TimeoutError('Corruption data deadline')
            time.sleep(30)
        archive=ROOT/'data/CIFAR-100-C.tar'; part=Path(str(archive)+'.part')
        if part.exists() and not archive.exists():
            with part.open('rb') as f: good=hashlib.file_digest(f,'md5').hexdigest()=='11f0ed0f1191edbf9fa23466ae6021d3'
            if good: part.rename(archive)
        execute(['prepare.py'],'data_validate')
        for seed in range(3):
            for method in ['baseline','augmix']:
                run=ROOT/'runs'/f'{method}_seed{seed}'; summary=run/'summary.json'
                if summary.exists() and json.loads(summary.read_text())['complete'] and not (run/'evaluation.csv').exists(): execute(['run.py','eval','--run',str(run)],f'eval_{method}_seed{seed}')
        progress('experiments_finished_report_pending'); execute(['report.py'],'report_final')
        count=sum((ROOT/'runs'/f'{m}_seed{s}'/'evaluation.csv').exists() for s in range(3) for m in ['baseline','augmix'])
        progress('complete' if count==6 else 'partial_complete',evaluated_runs=count)
        subprocess.run([sys.executable,'report.py'],cwd=ROOT,check=True)
    except BaseException:
        progress('stopped_on_error',traceback=traceback.format_exc())
        subprocess.run([sys.executable,'report.py'],cwd=ROOT)
        raise

if __name__=='__main__': main()
