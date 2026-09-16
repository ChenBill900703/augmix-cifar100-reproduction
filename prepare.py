"""Trusted-source download, checksum verification, structural validation."""
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import psutil
import torch
import torchvision
from torchvision.datasets import CIFAR100
from core import ROOT, CORRUPTIONS, save_json

def main():
    data=ROOT/'data'; data.mkdir(exist_ok=True)
    save_json(ROOT/'environment.json',dict(time=time.time(),os=platform.platform(),cpu=platform.processor(),logical_cpus=psutil.cpu_count(),physical_cpus=psutil.cpu_count(False),ram_bytes=psutil.virtual_memory().total,python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),vram=torch.cuda.get_device_properties(0).total_memory,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True)))
    (ROOT/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    CIFAR100(data,train=True,download=True); test=CIFAR100(data,train=False,download=True)
    archive=data/'CIFAR-100-C.tar'
    if not archive.exists():
        subprocess.run(['curl.exe','-L','--fail','--retry','4','-C','-','https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1','-o',str(archive)+'.part'],check=True)
        Path(str(archive)+'.part').rename(archive)
    with archive.open('rb') as f: md5=hashlib.file_digest(f,'md5').hexdigest()
    assert md5=='11f0ed0f1191edbf9fa23466ae6021d3',md5
    with tarfile.open(archive) as tar: tar.extractall(data,filter='data')
    labels=np.load(data/'CIFAR-100-C/labels.npy'); assert labels.shape==(50000,)
    for severity in range(5): np.testing.assert_array_equal(labels[severity*10000:(severity+1)*10000],test.targets)
    files={}
    for name in CORRUPTIONS:
        path=data/'CIFAR-100-C'/f'{name}.npy'; x=np.load(path,mmap_mode='r')
        assert x.shape==(50000,32,32,3) and x.dtype==np.uint8
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        files[name]=dict(shape=list(x.shape),dtype=str(x.dtype),sha256=digest)
    save_json(ROOT/'data_manifest.json',dict(cifar100_archive_md5=CIFAR100.tgz_md5,cifar100c_md5=md5,labels_match_all_five_blocks=True,corruptions=files,excluded_extra_files=[p.name for p in (data/'CIFAR-100-C').glob('*.npy') if p.stem not in CORRUPTIONS+['labels']]))
    print('DATA VERIFIED',flush=True)

if __name__=='__main__': main()
