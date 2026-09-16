"""Download-only helper; no model training, agent wakeups, or API/token usage."""
import hashlib
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parent
def running(fragment):
    for p in psutil.process_iter(['cmdline']):
        cmd=p.info['cmdline'] or []
        if any(s==fragment or (fragment.startswith('https:') and s.startswith(fragment)) for s in cmd):
            return True
    return False

if __name__=='__main__':
    while running('download_clean.py'): time.sleep(30)
    if not (ROOT/'clean_ready.json').exists():
        with (ROOT/'logs/download_clean_retry.log').open('a') as f:
            subprocess.run([sys.executable,'-u',str(ROOT/'download_clean.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    while running('https://zenodo.org/records/3555552/files/CIFAR-100-C.tar'): time.sleep(30)
    part=ROOT/'data/CIFAR-100-C.tar.part'; final=ROOT/'data/CIFAR-100-C.tar'
    if part.exists() and not final.exists():
        with part.open('rb') as f: digest=hashlib.file_digest(f,'md5').hexdigest()
        if digest=='11f0ed0f1191edbf9fa23466ae6021d3': part.rename(final)
    print('Download-only helper finished. Further verification/training awaits user continuation.',flush=True)
