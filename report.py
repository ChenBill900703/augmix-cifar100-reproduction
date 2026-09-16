"""Generate tables/plots from raw records, with missing experiments explicit."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import ROOT, CORRUPTIONS, write_csv, save_json

def read_csv(path):
    with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    table=[]; all_cells=[]
    for seed in range(3):
        for method in ['baseline','augmix']:
            run=ROOT/'runs'/f'{method}_seed{seed}'; ep=run/'epochs.csv'; ev=run/'evaluation.csv'
            if ep.exists():
                rows=read_csv(ep)
                fig,axes=plt.subplots(1,2,figsize=(10,3.5))
                for key in ['train_loss','ce','jsd']: axes[0].plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],label=key)
                for key in ['clean_top1','clean_top5']: axes[1].plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],label=key)
                for ax in axes: ax.legend(); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
                fig.suptitle(f'{method}, seed {seed}'); fig.tight_layout(); fig.savefig(out/f'{method}_seed{seed}_curves.png',dpi=150); plt.close(fig)
            if not ev.exists(): continue
            rows=read_csv(ev)
            for checkpoint in ['best','last']:
                rs=[r for r in rows if r['checkpoint']==checkpoint]; clean=next(r for r in rs if r['corruption']=='clean'); cs=[r for r in rs if r['corruption']!='clean']
                assert len(cs)==75 and len({(r['corruption'],r['severity']) for r in cs})==75
                table.append(dict(method=method,seed=seed,checkpoint=checkpoint,epoch=int(clean['epoch']),clean_top1=float(clean['top1_error']),clean_top5=float(clean['top5_error']),corruption_top1=float(np.mean([float(r['top1_error']) for r in cs])),corruption_top5=float(np.mean([float(r['top5_error']) for r in cs]))))
                all_cells.extend(dict(method=method,seed=seed,**r) for r in cs)
    write_csv(out/'per_seed.csv',table); write_csv(out/'corruption_detail.csv',all_cells)
    stats=[]; paired=[]
    for checkpoint in ['best','last']:
        for method in ['baseline','augmix']:
            rs=[r for r in table if r['method']==method and r['checkpoint']==checkpoint]
            if rs:
                for metric in ['clean_top1','clean_top5','corruption_top1','corruption_top5']:
                    vals=[r[metric] for r in rs]; stats.append(dict(method=method,checkpoint=checkpoint,metric=metric,n=len(vals),mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None))
        for seed in range(3):
            b=next((r for r in table if r['checkpoint']==checkpoint and r['seed']==seed and r['method']=='baseline'),None)
            a=next((r for r in table if r['checkpoint']==checkpoint and r['seed']==seed and r['method']=='augmix'),None)
            if b and a: paired.append(dict(seed=seed,checkpoint=checkpoint,**{k:b[k]-a[k] for k in ['clean_top1','clean_top5','corruption_top1','corruption_top5']}))
        fig,ax=plt.subplots(figsize=(12,4)); found=False
        for method in ['baseline','augmix']:
            rs=[r for r in all_cells if r['method']==method and r['checkpoint']==checkpoint]
            if rs:
                found=True; ax.plot(CORRUPTIONS,[np.mean([float(r['top1_error']) for r in rs if r['corruption']==c]) for c in CORRUPTIONS],marker='o',label=method)
        if found:
            ax.legend(); ax.set_ylabel('Top-1 error (%)'); ax.set_title(f'{checkpoint}: available seeds, 5 severities'); ax.tick_params(axis='x',rotation=60); fig.tight_layout(); fig.savefig(out/f'corruption_{checkpoint}.png',dpi=150)
        plt.close(fig)
    save_json(out/'statistics.json',dict(summary=stats,paired_improvement_pp=paired))
    status=json.loads((ROOT/'progress.json').read_text(encoding='utf-8')) if (ROOT/'progress.json').exists() else {}
    text='# AugMix 復現報告（繁體中文，依實際紀錄更新）\n\n'
    text+='## 完成狀態\n\n目前有 '+str(len(table)//2)+' 個 run 完成 best/last 全部 clean 與 corruption 評估（目標 6）。未完成的實驗不填入推測數字。\n\n'
    text+='進度：`'+str(status.get('stage','preparation'))+'`。即時資訊见 progress.json；原始 epoch/exception 日誌在 runs 與 logs。\n\n'
    text+='## 已核對事實與設定差異\n\n詳見 SOURCE_AUDIT.md 的論文／官方／本次設定對照。WRN-40-2、100 epochs、batch 128、完整 CE+12×JSD；三視圖串接，沒有額外增強技巧。主比較為 last；best 根據每輪 clean test Top-1 選取，因此 clean test 並非未使用的獨立選模外測試。Top-1/Top-5 同列使用同 checkpoint。\n\n'
    text+='## 實測結果\n\n|方法|seed|checkpoint|epoch|clean Top-1|clean Top-5|CIFAR-C Top-1|CIFAR-C Top-5|\n|---|---|---|---|---|---|---|---|\n'
    for r in table: text+='|'+'|'.join(str(r[k]) if k in ['method','seed','checkpoint','epoch'] else f'{r[k]:.4f}' for k in ['method','seed','checkpoint','epoch','clean_top1','clean_top5','corruption_top1','corruption_top5'])+'|\n'
    if not table: text+='\n尚無完成的正式評估。\n'
    text+='\n平均與樣本標準差（ddof=1）、成對改善量（Baseline−AugMix，百分點）保存於 results/statistics.json；n<3 時不得稱為三次結果，n=1 不計樣本標準差。\n\n'
    for r in stats: text+=f"- {r['method']} / {r['checkpoint']} / {r['metric']}: n={r['n']}, mean={r['mean']:.4f}, sample SD={r['sample_sd']}\n"
    text+='\n## 論文比較與公式\n\n論文 Table 1 CIFAR-100-C／WRN-40-2：Standard 53.3%，AugMix 35.9%。無完全對應 clean Top-1/Top-5 數字可填。CIFAR-C 平均錯誤率為 (1/75)ΣcΣs[100×errors(c,s)/10000]，納入 SOURCE_AUDIT.md 的 15 類×5 severity，沒有 AlexNet 正規化。\n\n'
    text+='## 時間、顯存與可重現性\n\n環境见 environment.json；顯存完整 preflight 见 results/preflight.json；穩定五輪測速估算见 results/budget.json（不存在即尚未完成）；resume 實測见 results/resume_test.json（不存在即尚未通過）。逐 run 耗時與顯存在 epochs.csv。FP32 为正式預設，AMP 功能有獨立開關測試。固定 seed 不保證所有 CUDA 操作確定；續訓一致性只對已測環境與 epoch 邊界範圍有效。\n\n'
    text+='## 尚未證實的可能原因\n\n版本差異、隨機抽樣及 checkpoint 選模可能影響與論文差距；目前不將任何差距歸因於這些因素，需依實測與診斷支持。三個 seed 同方向不等於統計上證明與論文等同。\n\n'
    text+='## 已知限制與未完成項目\n\n未達六個完整 run 及 corruption 評估之前，本報告屬進行中。所有失敗與中斷保留，不可將修正前後 run 混成同一組三次。硬體斷電、作業系統睡眠或背景程序被終止時不會繼續訓練；重新執行 pipeline.py 會依 checkpoint 恢復。沒有執行 ImageNet、calibration 或額外消融。\n'
    (ROOT/'reproduction_report.md').write_text(text,encoding='utf-8')

if __name__=='__main__': main()
