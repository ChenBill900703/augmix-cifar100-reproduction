import argparse
import json
import traceback
import time
from pathlib import Path
from core import ROOT, train, evaluate_run, save_json

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['train','eval']); p.add_argument('--config'); p.add_argument('--run',required=True); p.add_argument('--stop-epoch',type=int); a=p.parse_args()
    try:
        if a.action=='train': train(json.loads(Path(a.config).read_text()),a.run,a.stop_epoch)
        else: evaluate_run(a.run)
    except BaseException:
        run=Path(a.run); run.mkdir(parents=True,exist_ok=True)
        with (run/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(dict(event='exception',time=time.time(),traceback=traceback.format_exc()))+'\n')
        raise
