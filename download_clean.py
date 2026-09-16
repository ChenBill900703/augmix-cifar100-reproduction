"""Eight bounded HTTP ranges from the official CIFAR host, verified before use."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import subprocess
from pathlib import Path
import tarfile
from core import ROOT, save_json

def main():
    size=169001437; n=8; directory=ROOT/'data/ranges'; directory.mkdir(parents=True,exist_ok=True)
    def fetch(i):
        start=size*i//n; end=size*(i+1)//n-1; path=directory/str(i)
        if not path.exists() or path.stat().st_size!=end-start+1:
            subprocess.run(['curl.exe','-sS','-L','--fail','--retry','5','--retry-all-errors','--connect-timeout','30','--max-time','3600','--range',f'{start}-{end}','https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz','-o',str(path)],check=True)
        assert path.stat().st_size==end-start+1
        return path
    with ThreadPoolExecutor(max_workers=n) as pool: paths=list(pool.map(fetch,range(n)))
    archive=ROOT/'data/cifar-verified.tar.gz'
    with archive.open('wb') as out:
        for path in paths:
            with path.open('rb') as f:
                while chunk:=f.read(1024*1024): out.write(chunk)
    with archive.open('rb') as f: md5=hashlib.file_digest(f,'md5').hexdigest()
    assert md5=='eb9058c3a382ffc7106e4002c42a8d85'
    with tarfile.open(archive) as tar: tar.extractall(ROOT/'data',filter='data')
    save_json(ROOT/'clean_ready.json',dict(md5=md5,source='https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz'))
    print('CLEAN DATA VERIFIED',flush=True)

if __name__=='__main__': main()
