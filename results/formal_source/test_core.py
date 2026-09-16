import json
import hashlib
import numpy as np
import pytest
import torch
from PIL import Image
import core

def test_architecture():
    net=core.model(); net.eval()
    assert net(torch.zeros(2,3,32,32)).shape==(2,100)
    assert sum(p.numel() for p in net.parameters())==2255156
    assert len(net.block1.layer)==len(net.block2.layer)==len(net.block3.layer)==6
    assert all(m.eps==1e-5 and m.momentum==.1 for m in net.modules() if isinstance(m,torch.nn.BatchNorm2d))

def test_aug():
    im=Image.fromarray(np.random.default_rng(1).integers(0,256,(32,32,3),dtype=np.uint8))
    core.seed_all(1)
    a,t=core.aug(im,True); b,u=core.aug(im,True)
    assert not torch.equal(a,b) and t!=u
    assert len(t['weights'])==3 and abs(sum(t['weights'])-1)<1e-6 and all(w>=0 for w in t['weights'])
    assert 0<=t['m']<=1 and all(1<=len(c)<=3 for c in t['chains'])
    for op in core.augmentations:
        out=op(im,3); assert out.size==(32,32) and np.asarray(out).dtype==np.uint8
    assert a.min()>=-1.000001 and a.max()<=1.000001

def test_jsd():
    x=torch.randn(8,100,requires_grad=True)
    assert abs(core.jsd([x,x,x]).item())<1e-6
    xs=[torch.randn(8,100,requires_grad=True) for _ in range(3)]
    j=core.jsd(xs); assert 0<=j.item()<=np.log(3)
    j.backward(); assert all(torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in xs)
    ps=[x.detach().softmax(1) for x in xs]; m=sum(ps)/3
    expected=sum((p*(p.log()-m.log())).sum(1).mean() for p in ps)/3
    torch.testing.assert_close(j,expected)

def test_metrics_weighting():
    logits=torch.eye(6)*10; labels=torch.tensor([0,1,2,3,4,0])
    total=np.array(core.metrics(logits[:4],labels[:4]))+core.metrics(logits[4:],labels[4:])
    np.testing.assert_allclose(total,core.metrics(logits,labels),rtol=1e-6)
    assert total[1]==5 and total[3]==6
    assert core.metrics(torch.tensor([[6.,5.,4.,3.,2.,1.]]),torch.tensor([4]))[1:3]==[0,1]
    assert core.metrics(torch.tensor([[6.,5.,4.,3.,2.,1.]]),torch.tensor([5]))[1:3]==[0,0]

def test_corruption_slices(tmp_path):
    x=np.lib.format.open_memmap(tmp_path/'x.npy',mode='w+',dtype='uint8',shape=(50000,32,32,3))
    for s in range(5): x[s*10000:(s+1)*10000]=s*20
    del x
    np.save(tmp_path/'labels.npy',np.tile(np.arange(10000)%100,5))
    for s in range(1,6):
        d=core.CorruptionData(tmp_path/'x.npy',s)
        assert len(d)==10000 and d[0][1]==0 and d[9999][1]==99
        assert abs(d[0][0][0,0,0].item()-((s-1)*20/255-.5)/.5)<1e-6

@pytest.mark.parametrize('amp',[False,True])
def test_amp(amp):
    c=json.loads((core.ROOT/'configs/smoke.json').read_text()); c['amp']=amp
    net,opt,sched,scaler=core.setup(c)
    before=net.fc.weight.detach().clone()
    vals,skip,dtype=core.step(net,opt,sched,scaler,[torch.randn(8,3,32,32) for _ in range(3)],torch.randint(100,(8,)),amp)
    assert dtype==('torch.float16' if amp else 'torch.float32')
    assert scaler.is_enabled()==amp and not torch.equal(before,net.fc.weight)
    assert all(np.isfinite(vals))

@pytest.mark.parametrize('method',['baseline','augmix'])
def test_resume(tmp_path,method):
    c=json.loads((core.ROOT/'configs/smoke.json').read_text())
    c['method']=method
    a,ra=core.train(c,tmp_path/'continuous')
    core.train(c,tmp_path/'resumed',stop_epoch=1)
    b,rb=core.train(c,tmp_path/'resumed')
    maxdiff=max((a.state_dict()[k]-b.state_dict()[k]).abs().max().item() for k in a.state_dict())
    assert [r['order_sha256'] for r in ra]==[r['order_sha256'] for r in rb]
    assert maxdiff==0
    assert [r['train_loss'] for r in ra]==[r['train_loss'] for r in rb]
    core.save_json(core.ROOT/f'results/resume_test_{method}.json',dict(max_weight_difference=maxdiff,orders_equal=True,losses_equal=True,config=c,scope='same environment, epoch boundaries, smoke test only'))
