"""CIFAR port of google-research/augmix (Apache-2.0; see vendor licenses).

WRN and augmentation operators are imported unchanged from the pinned source.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/augmix'))
from augmentations import augmentations
from third_party.WideResNet_pytorch.wideresnet import WideResNet

CORRUPTIONS = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur',
 'glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness',
 'contrast','elastic_transform','pixelate','jpeg_compression']
PREPROCESS = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)])
BASIC = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4)])
METHOD_SPEC = dict(model='WRN-40-2',num_classes=100,dropout=0,optimizer='SGD',lr=.1,momentum=.9,weight_decay=.0005,nesterov=True,lr_min=1e-6,scheduler='per-batch cosine after optimizer',mean=[.5]*3,std=[.5]*3,mixture_width=3,mixture_depth=[1,2,3],severity=3,dirichlet_alpha=1,beta_alpha=1,jsd_weight=12,mixture_clamp=[1e-7,1],three_view_concat=True,cudnn_benchmark=False,cudnn_deterministic=True,tf32=False,deterministic_algorithms=False,resume_scope='epoch-boundary')

def source_hashes():
    paths=list(ROOT.glob('*.py'))+[ROOT/'vendor/augmix/augmentations.py',ROOT/'vendor/augmix/third_party/WideResNet_pytorch/wideresnet.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])

def aug(image, trace=False):
    ws=np.float32(np.random.dirichlet([1]*3)); m=np.float32(np.random.beta(1,1))
    mix=torch.zeros_like(PREPROCESS(image)); chains=[]
    for i in range(3):
        im=image.copy(); names=[]
        for _ in range(np.random.randint(1,4)):
            op=np.random.choice(augmentations); names.append(op.__name__); im=op(im,3)
        mix += ws[i]*PREPROCESS(im); chains.append(names)
    result=(1-m)*PREPROCESS(image)+m*mix
    return (result,dict(weights=ws.tolist(),m=float(m),chains=chains)) if trace else result

def jsd(logits):
    ps=[F.softmax(x.float(),dim=1) for x in logits]
    logm=torch.clamp(sum(ps)/3,1e-7,1).log()
    return sum(F.kl_div(logm,p,reduction='batchmean') for p in ps)/3

class TrainData(Dataset):
    def __init__(self, method, seed, epoch, limit=None):
        self.base=datasets.CIFAR100(ROOT/'data',train=True,download=False)
        self.method,self.seed,self.epoch=method,seed,epoch
        self.n=limit or len(self.base)
    def __len__(self): return self.n
    def __getitem__(self,i):
        # Independent per-example streams permit epoch-boundary resume even with workers.
        # This changes RNG assignment, not augmentation distribution; never claim old-code bit equivalence.
        value=int.from_bytes(hashlib.blake2b(f'{self.seed}:{self.epoch}:{i}'.encode(),digest_size=4).digest(),'little')
        random.seed(value); np.random.seed(value); torch.manual_seed(value)
        x,y=self.base[i]; x=BASIC(x)
        return ((PREPROCESS(x),aug(x),aug(x)) if self.method=='augmix' else PREPROCESS(x)),y,i

def train_loader(c,epoch):
    g=torch.Generator().manual_seed(c['seed']*100000+epoch)
    return DataLoader(TrainData(c['method'],c['seed'],epoch,c.get('limit')),batch_size=c['batch_size'],shuffle=True,generator=g,num_workers=c['workers'],pin_memory=True)

def clean_loader(c):
    d=datasets.CIFAR100(ROOT/'data',train=False,transform=PREPROCESS,download=False)
    if c.get('test_limit'): d=torch.utils.data.Subset(d,range(c['test_limit']))
    return DataLoader(d,batch_size=c['eval_batch_size'],num_workers=0,pin_memory=True)

def model(): return WideResNet(40,100,2,0)

def setup(c):
    seed_all(c['seed']); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    net=model().cuda()
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9,weight_decay=.0005,nesterov=True)
    steps=c['epochs']*math.ceil((c.get('limit') or 50000)/c['batch_size'])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: 1e-5+(1-1e-5)*.5*(1+math.cos(s/steps*math.pi)))
    scaler=torch.amp.GradScaler('cuda',enabled=c['amp'])
    return net,opt,sched,scaler

def step(net,opt,sched,scaler,images,targets,amp):
    opt.zero_grad(set_to_none=True); y=targets.cuda(non_blocking=True)
    x=(torch.cat(images,0) if isinstance(images,(list,tuple)) else images).cuda(non_blocking=True)
    with torch.autocast('cuda',enabled=amp):
        logits=net(x)
        if isinstance(images,(list,tuple)):
            parts=logits.split(len(y)); ce=F.cross_entropy(parts[0],y); j=jsd(parts)
        else: ce=F.cross_entropy(logits,y); j=ce.new_zeros(())
        loss=ce+12*j
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    finite=torch.stack([torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]).all().item()
    if not finite and not amp: raise FloatingPointError('Nonfinite FP32 gradient')
    before=scaler.get_scale(); scaler.step(opt); scaler.update()
    skipped=scaler.get_scale()<before
    # Official scheduler advances once per batch; a skipped AMP update is recorded explicitly.
    sched.step()
    return [loss.item(),ce.item(),j.item()],int(skipped),str(logits.dtype)

def metrics(logits,y):
    ids=logits.topk(5,dim=1).indices
    return [F.cross_entropy(logits,y,reduction='sum').item(),ids[:,0].eq(y).sum().item(),ids.eq(y[:,None]).any(1).sum().item(),len(y)]

@torch.no_grad()
def evaluate(net,loader):
    net.eval(); totals=np.zeros(4); start=time.perf_counter()
    for x,y in loader: totals+=metrics(net(x.cuda(non_blocking=True)),y.cuda(non_blocking=True))
    torch.cuda.synchronize(); loss,c1,c5,n=totals
    return dict(loss=float(loss/n),top1_error=float(100*(1-c1/n)),top5_error=float(100*(1-c5/n)),n=int(n),seconds=time.perf_counter()-start)

def memory():
    return dict(allocated_GiB=torch.cuda.max_memory_allocated()/2**30,reserved_GiB=torch.cuda.max_memory_reserved()/2**30)

def save_ckpt(path,state):
    tmp=Path(str(path)+'.tmp'); torch.save(state,tmp); os.replace(tmp,path)

def write_csv(path,rows):
    if not rows: return
    tmp=Path(str(path)+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)

def train(c,run,stop_epoch=None):
    run=Path(run); run.mkdir(parents=True,exist_ok=True)
    net,opt,sched,scaler=setup(c); rows=[]; best=100.; best_state=None; start=0
    last=run/'last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['config']!=c: raise ValueError('Resume config mismatch')
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        restore_rng(ck['rng']); rows=ck['rows']; best=ck['best']; best_state=ck['best_model']; start=ck['epoch']
        write_csv(run/'epochs.csv',rows)
        with (run/'events.jsonl').open('a') as f: f.write(json.dumps({'event':'resume','epoch':start,'time':time.time()})+'\n')
    save_json(run/'metadata.json',dict(config=c,method_spec=METHOD_SPEC,code_hashes=source_hashes(),selection='minimum clean test top1 error; first tie wins',source_sha='9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4'))
    for epoch in range(start,stop_epoch or c['epochs']):
        begin=time.perf_counter(); net.train(); torch.cuda.reset_peak_memory_stats()
        sums=np.zeros(3); count=0; skipped=0; order=hashlib.sha256(); lr=opt.param_groups[0]['lr']
        for x,y,ids in train_loader(c,epoch):
            try:
                vals,skip,dtype=step(net,opt,sched,scaler,x,y,c['amp'])
            except BaseException:
                save_ckpt(run/f'diagnostic_epoch{epoch+1}_sample{count}.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),config=c,rng=rng_state(),sample_ids=ids,targets=y))
                raise
            sums+=np.array(vals)*len(y); count+=len(y); skipped+=skip; order.update(ids.numpy().tobytes())
        torch.cuda.synchronize(); train_secs=time.perf_counter()-begin; train_mem=memory()
        val=evaluate(net,clean_loader(c)); is_best=val['top1_error']<best
        if rows and (sums[0]/count>max(20,rows[-1]['train_loss']*4) or (val['top1_error']>95 and rows[-1]['clean_top1']<80)):
            save_ckpt(run/f'anomaly_epoch{epoch+1}.pt',dict(model=net.state_dict(),config=c,loss=sums.tolist(),val=val))
            raise FloatingPointError('Severe loss/error spike: diagnostic retained; manual investigation required')
        if is_best:
            best=val['top1_error']; best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        row=dict(epoch=epoch+1,lr=lr,lr_end=opt.param_groups[0]['lr'],train_loss=sums[0]/count,ce=sums[1]/count,jsd=sums[2]/count,clean_loss=val['loss'],clean_top1=val['top1_error'],clean_top5=val['top5_error'],train_seconds=train_secs,val_seconds=val['seconds'],allocated_GiB=train_mem['allocated_GiB'],reserved_GiB=train_mem['reserved_GiB'],amp_scale=scaler.get_scale(),skipped_updates=skipped,forward_dtype=dtype,order_sha256=order.hexdigest(),best_top1=best,is_best=is_best,epoch_seconds=0.)
        rows.append(row)
        state=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,best=best,best_model=best_state,config=c,method_spec=METHOD_SPEC,rng=rng_state(),rows=rows)
        save_ckpt(last,state)
        if is_best: save_ckpt(run/'best.pt',state)
        row['epoch_seconds']=time.perf_counter()-begin
        # Commit authoritative CSV history into checkpoint; recovery regenerates CSV from it.
        save_ckpt(last,state); write_csv(run/'epochs.csv',rows)
        save_json(run/'summary.json',dict(completed_epochs=epoch+1,complete=epoch+1==c['epochs'],last=row))
        print(json.dumps(dict(run=str(run),**row)),flush=True)
    return net,rows

class CorruptionData(Dataset):
    def __init__(self,path,severity):
        if severity not in range(1,6): raise ValueError('severity')
        self.x=np.load(path,mmap_mode='r'); self.y=np.load(Path(path).parent/'labels.npy',mmap_mode='r'); self.offset=(severity-1)*10000
        assert self.x.shape==(50000,32,32,3) and self.x.dtype==np.uint8
        assert self.y.shape==(50000,)
    def __len__(self): return 10000
    def __getitem__(self,i):
        j=self.offset+i
        return PREPROCESS(Image.fromarray(self.x[j])),int(self.y[j])

def evaluate_run(run):
    run=Path(run); output=[]
    identity={name:hashlib.sha256((run/f'{name}.pt').read_bytes()).hexdigest() for name in ['best','last']}
    manifest=run/'evaluation_identity.json'; partial=run/'evaluation.partial.csv'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise ValueError('Checkpoint changed since partial evaluation; preserve old evaluation before restarting')
    save_json(manifest,identity)
    if partial.exists():
        with partial.open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                output.append({k:(int(v) if k in ['epoch','severity','n'] else float(v) if k in ['loss','top1_error','top5_error','seconds'] else v) for k,v in r.items()})
    completed={(r['checkpoint'],r['corruption'],r['severity']) for r in output}
    for name in ['best','last']:
        ck=torch.load(run/f'{name}.pt',map_location='cpu',weights_only=False); net=model().cuda(); net.load_state_dict(ck['model']); c=ck['config']
        if (name,'clean',0) not in completed:
            result=evaluate(net,clean_loader(c)); output.append(dict(checkpoint=name,epoch=ck['epoch'],corruption='clean',severity=0,**result))
            write_csv(partial,output)
        for corruption in CORRUPTIONS:
            for severity in range(1,6):
                if (name,corruption,severity) in completed: continue
                loader=DataLoader(CorruptionData(ROOT/'data/CIFAR-100-C'/f'{corruption}.npy',severity),batch_size=c['eval_batch_size'],num_workers=0,pin_memory=True)
                result=evaluate(net,loader); output.append(dict(checkpoint=name,epoch=ck['epoch'],corruption=corruption,severity=severity,**result))
                write_csv(run/'evaluation.partial.csv',output)
                print(json.dumps(dict(run=str(run),checkpoint=name,corruption=corruption,severity=severity,**result)),flush=True)
        del net
    assert len(output)==152
    write_csv(run/'evaluation.csv',output)
    return output
