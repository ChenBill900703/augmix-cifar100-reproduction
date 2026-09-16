"""Public entry point: unchanged training core, explicitly aligned evaluation."""
import argparse
import json
from pathlib import Path

BACKEND=dict(cudnn_benchmark=False,cudnn_deterministic=True,cudnn_tf32=False,matmul_tf32=False,threads=4,amp=False)

def configure_backend():
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--config',required=True); tr.add_argument('--run',required=True)
    ev=sub.add_parser('evaluate'); ev.add_argument('--run',required=True)
    args=parser.parse_args()
    from core import train,evaluate_run,save_json
    if args.action=='train':
        train(json.loads(Path(args.config).read_text(encoding='utf-8')),Path(args.run))
    else:
        run=Path(args.run)
        for name in ['best.pt','last.pt']:
            if not (run/name).is_file(): raise FileNotFoundError(f'{run/name}: weights are not distributed with this repository; train first.')
        backend_file=run/'evaluation_backend.json'
        has_results=any((run/name).exists() for name in ['evaluation.csv','evaluation.partial.csv'])
        if has_results and (not backend_file.exists() or json.loads(backend_file.read_text(encoding='utf-8'))!=BACKEND):
            raise RuntimeError('Existing evaluation backend is missing/different. Preserve those files in a separate directory before evaluating; do not mix results.')
        configure_backend(); save_json(backend_file,BACKEND)
        evaluate_run(run)

if __name__=='__main__': main()
