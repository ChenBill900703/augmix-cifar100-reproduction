"""Observational watchdog; never changes training state or restarts experiments."""
import ctypes
import json
from pathlib import Path
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    kernel=ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
    kernel.SetThreadExecutionState.restype=ctypes.c_uint
    previous=kernel.SetThreadExecutionState(0x80000001)
    if not previous: raise OSError('Could not request temporary system-awake state')
    print('Temporary automatic-sleep prevention active; display may turn off.',flush=True)
    last_stage=None; stage_start=time.time()
    try:
        while True:
            now=time.time(); state=json.loads((ROOT/'progress.json').read_text())
            pid=state['controller_pid']
            alive=psutil.pid_exists(pid)
            if not alive or state['stage'] in ['complete','partial_complete','stopped_on_error']:
                print(f'Controller finished/stopped: {state["stage"]}; releasing awake request.',flush=True)
                return
            controller=psutil.Process(pid)
            # Check executable and command as well as PID, to avoid PID reuse misclassification.
            if 'pipeline.py' not in controller.cmdline(): raise RuntimeError('Controller PID identity mismatch')
            stage=state['stage']
            if stage!=last_stage: stage_start=now; last_stage=stage
            processes=[]
            for p in [controller]+controller.children(recursive=True):
                try:
                    cpu=p.cpu_times(); io=p.io_counters()
                    processes.append(dict(pid=p.pid,created=p.create_time(),status=p.status(),cpu_user=cpu.user,cpu_system=cpu.system,rss=p.memory_info().rss,threads=p.num_threads(),read_bytes=io.read_bytes,write_bytes=io.write_bytes,command=p.cmdline()))
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
            summary=ROOT/'runs'/stage/'summary.json'
            age=now-max(stage_start,summary.stat().st_mtime) if summary.exists() else now-stage_start
            is_training=stage.startswith(('baseline_seed','augmix_seed','timing_'))
            record=dict(time=now,stage=stage,heartbeat_age=now-state['updated'],epoch_age_seconds=age,stall_suspected=is_training and age>180,processes=processes,system_memory_available=psutil.virtual_memory().available)
            try:
                record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader'],text=True,timeout=10).strip()
            except Exception as e: record['gpu_error']=str(e)
            with (ROOT/'logs/health.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
            temp=ROOT/'logs/health_latest.tmp'; temp.write_text(json.dumps(record,indent=2),encoding='utf-8')
            # Windows readers/antivirus can briefly deny replacement. The append-only
            # history above remains authoritative; never abandon monitoring for this.
            for attempt in range(10):
                try:
                    temp.replace(ROOT/'logs/health_latest.json')
                    break
                except PermissionError:
                    if attempt==9: print('Status replacement deferred; full record retained in health.jsonl',flush=True)
                    else: time.sleep(.2)
            if record['stall_suspected']:
                alert=ROOT/'logs'/f'stall_diagnostic_{stage}_{int(now)}.json'
                alert.write_text(json.dumps(record,indent=2),encoding='utf-8')
            time.sleep(30)
    finally:
        kernel.SetThreadExecutionState(0x80000000)

if __name__=='__main__': main()
