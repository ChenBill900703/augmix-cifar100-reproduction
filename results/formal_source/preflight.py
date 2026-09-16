import json
import gc
import torch
from core import ROOT, setup, step, memory, save_json

def main():
    results=[]; limit=min(7.2,torch.cuda.get_device_properties(0).total_memory/2**30*.9)
    for method in ['baseline','augmix']:
        for amp in [False,True]:
            c=json.loads((ROOT/'configs'/f'{method}.json').read_text()); c['amp']=amp
            net,opt,sched,scaler=setup(c); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                net.train()
                for _ in range(5):
                    xs=[torch.rand(128,3,32,32)*2-1 for _ in range(3)] if method=='augmix' else torch.rand(128,3,32,32)*2-1
                    vals,skipped,dtype=step(net,opt,sched,scaler,xs,torch.randint(100,(128,)),amp)
                torch.cuda.synchronize(); training=memory()
                del xs; opt.zero_grad(set_to_none=True); net.eval(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for _ in range(3): net(torch.rand(1000,3,32,32,device='cuda'))
                torch.cuda.synchronize(); validation=memory()
                result=dict(method=method,amp=amp,training=training,validation=validation,limit_GiB=limit,pass_limit=max(training['reserved_GiB'],validation['reserved_GiB'])<=limit,forward_dtype=dtype,scaler_enabled=scaler.is_enabled(),last_loss=vals)
            except torch.cuda.OutOfMemoryError as e:
                result=dict(method=method,amp=amp,pass_limit=False,oom=str(e),limit_GiB=limit)
            results.append(result); save_json(ROOT/'results/preflight.json',results); print(result,flush=True)
            del net,opt,sched,scaler; gc.collect(); torch.cuda.empty_cache()
    assert all(any(r['method']==m and r['pass_limit'] for r in results) for m in ['baseline','augmix'])

if __name__=='__main__': main()
