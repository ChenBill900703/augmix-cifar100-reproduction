"""Audit completed experiments and build the final Traditional Chinese report."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from core import CORRUPTIONS, save_json, write_csv

def read(path):
    with path.open(encoding='utf-8') as f: return list(csv.DictReader(f))

def table(headers,rows):
    return '|'+ '|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('|'+ '|'.join(map(str,r))+'|\n' for r in rows)+'\n'

def main():
    protocol=json.loads((ROOT/'results/formal_protocol.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in protocol['code'].items())
    verified=json.loads((ROOT/'results/evaluation_alignment_verified.json').read_text())
    assert verified['all_12_clean_top1_top5_match_epoch_records']
    summaries=read(ROOT/'results/per_seed.csv'); assert len(summaries)==12
    resources=[]; cells=[]; audits=[]; discrepancies=[]
    for seed in range(3):
        for method in ['baseline','augmix']:
            run=ROOT/'runs'/f'{method}_seed{seed}'; epochs=read(run/'epochs.csv'); ev=read(run/'evaluation.csv')
            assert [int(r['epoch']) for r in epochs]==list(range(1,101))
            assert len(ev)==152
            assert {(r['checkpoint'],r['corruption'],int(r['severity'])) for r in ev}=={(cp,c,s) for cp in ['best','last'] for c,s in [('clean',0)]+[(c,s) for c in CORRUPTIONS for s in range(1,6)]}
            assert all(int(r['n'])==10000 and 0<=float(r['top5_error'])<=float(r['top1_error'])<=100 for r in ev)
            assert all(math.isfinite(float(r[k])) for r in epochs for k in ['train_loss','ce','jsd','clean_loss','clean_top1','clean_top5'])
            identity=json.loads((run/'evaluation_identity.json').read_text())
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            assert last['epoch']==100 and last['scheduler']['last_epoch']==39100 and len(last['rows'])==100
            assert last['config']==dict(protocol['configs'][method],seed=seed)
            best_epoch=int(min(epochs,key=lambda r:float(r['clean_top1']))['epoch'])
            for cp in ['best','last']:
                path=run/f'{cp}.pt'
                with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
                assert digest==identity[cp]
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['epoch']==(best_epoch if cp=='best' else 100)
                assert all(torch.isfinite(v).all() for v in ck['model'].values())
                if cp=='best': assert all(torch.equal(v,last['best_model'][k]) for k,v in ck['model'].items())
                row=next(r for r in ev if r['checkpoint']==cp and r['corruption']=='clean')
                assert float(row['top1_error'])==ck['rows'][-1]['clean_top1']
                assert float(row['top5_error'])==ck['rows'][-1]['clean_top5']
                old=read(run/'evaluation_initial_default_backend/evaluation.csv')
                oldrow=next(r for r in old if r['checkpoint']==cp and r['corruption']=='clean')
                discrepancies.append(dict(run=run.name,checkpoint=cp,old_top1=float(oldrow['top1_error']),aligned_top1=float(row['top1_error']),old_top5=float(oldrow['top5_error']),aligned_top5=float(row['top5_error'])))
            cells.extend(dict(method=method,seed=seed,**r) for r in ev if r['corruption']!='clean')
            resources.append(dict(method=method,seed=seed,train_hours=sum(float(r['train_seconds']) for r in epochs)/3600,validation_hours=sum(float(r['val_seconds']) for r in epochs)/3600,epoch_hours=sum(float(r['epoch_seconds']) for r in epochs)/3600,peak_allocated_GiB=max(float(r['allocated_GiB']) for r in epochs),peak_reserved_GiB=max(float(r['reserved_GiB']) for r in epochs),evaluation_hours=sum(float(r['seconds']) for r in ev)/3600))
            audits.append(dict(run=run.name,epochs=100,eval_rows=152,checkpoint_hashes=identity,best_epoch=best_epoch,all_clean_match=True))
    write_csv(ROOT/'results/runtime_summary.csv',resources); write_csv(ROOT/'results/evaluation_backend_comparison.csv',discrepancies)
    paired=[]; aggregate=[]; corruption=[]
    metrics=['clean_top1','clean_top5','corruption_top1','corruption_top5']
    for cp in ['last','best']:
        for method in ['baseline','augmix']:
            rows=[r for r in summaries if r['checkpoint']==cp and r['method']==method]
            aggregate.append([method,cp]+[f"{st.mean(float(r[k]) for r in rows):.3f} ± {st.stdev(float(r[k]) for r in rows):.3f}" for k in metrics])
        for seed in range(3):
            b=next(r for r in summaries if r['checkpoint']==cp and r['method']=='baseline' and int(r['seed'])==seed)
            a=next(r for r in summaries if r['checkpoint']==cp and r['method']=='augmix' and int(r['seed'])==seed)
            paired.append(dict(checkpoint=cp,seed=seed,**{k:float(b[k])-float(a[k]) for k in metrics}))
        for c in CORRUPTIONS:
            means={m:st.mean(float(r['top1_error']) for r in cells if r['checkpoint']==cp and r['method']==m and r['corruption']==c) for m in ['baseline','augmix']}
            corruption.append(dict(checkpoint=cp,corruption=c,**means,improvement_pp=means['baseline']-means['augmix']))
    write_csv(ROOT/'results/paired_improvements.csv',paired); write_csv(ROOT/'results/corruption_means.csv',corruption)
    grid=np.array([[st.mean(float(r['top1_error']) for r in cells if r['method']=='baseline' and r['checkpoint']=='last' and r['corruption']==c and int(r['severity'])==s)-st.mean(float(r['top1_error']) for r in cells if r['method']=='augmix' and r['checkpoint']=='last' and r['corruption']==c and int(r['severity'])==s) for s in range(1,6)] for c in CORRUPTIONS])
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(grid,aspect='auto',cmap='YlGnBu'); ax.set_yticks(range(15),CORRUPTIONS); ax.set_xticks(range(5),range(1,6)); ax.set_xlabel('Severity'); ax.set_title('Baseline - AugMix Top-1 error (pp), 3-seed mean, last')
    for i in range(15):
        for j in range(5): ax.text(j,i,f'{grid[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if grid[i,j]>grid.max()*.6 else 'black')
    fig.colorbar(im,ax=ax,label='Percentage points'); fig.tight_layout(); fig.savefig(ROOT/'results/corruption_severity_improvement.png',dpi=160); plt.close(fig)
    allhours=sum(r['epoch_hours'] for r in resources)
    evalhours=sum(r['evaluation_hours'] for r in resources)
    wallhours=(verified['completed']-1789417552)/3600
    lastpaired=[r for r in paired if r['checkpoint']=='last']; benefit=st.mean(r['corruption_top1'] for r in lastpaired)
    lastmeans={m:st.mean(float(r['corruption_top1']) for r in summaries if r['method']==m and r['checkpoint']=='last') for m in ['baseline','augmix']}
    text='# AugMix：CIFAR-100／WRN-40-2 復現報告\n\n'
    text+='六次正式訓練與一致設定的完整評估已完成：Baseline／AugMix，各 seeds 0、1、2，各100 epochs。共600輪、12個best/last checkpoint、912列clean或corruption/severity評估。沒有ImageNet訓練，也沒有另加Mixup、CutMix、EMA或預訓練。\n\n'
    text+=f"主要結果（last）：CIFAR-100-C 平均Top-1錯誤率由 **{lastmeans['baseline']:.3f}% 降至 {lastmeans['augmix']:.3f}%**，改善 **{benefit:.3f}百分點**。三個seed皆改善，但不能因此宣稱統計等同或超越論文。\n\n"
    text+='## 1. 已核對的來源、設定與公平比較\n\n'
    text+='論文為arXiv:1912.02781v2（2020-02-17，ICLR 2020），已閱讀主文與附錄並目視核對Table 1。官方commit為`9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4`。Google程式為Apache-2.0；第三方WRN為MIT，原始授權與來源保留於vendor/augmix。完整論文／官方／本次對照見 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)。\n\n'
    text+='WRN-40-2原始模型與初始化直接匯入，參數2,255,156；batch=128、100 epochs、SGD lr=.1、Nesterov momentum=.9、weight decay=.0005，每step後cosine排程至1e-6。基本處理為左右翻轉、padding4隨機裁切、各channel mean/std=.5。AugMix採官方九操作、severity3、width3、depth均勻1–3、Dirichlet/Beta α=1。\n\n'
    text+='AugMix將基本增強後原圖與兩個獨立AugMix視圖串接，一次forward，CE只計原圖；loss=CE+12×JSD，mixture clamp[1e-7,1]後log，所有分支保留梯度。Baseline無AugMix與JSD；兩者其餘訓練条件相同，沒有梯度累積或拆分forward。\n\n'
    text+='主比較固定使用last；best依每輪clean test Top-1嚴格改善選取（同分保留先前）。test因此參與選模，不能宣稱完全獨立未使用。資料未參與梯度更新，也未根據測試數字調參、換seed或挑較佳run。\n\n'
    text+='## 2. 主要結果：三次平均與樣本標準差\n\n數字皆為錯誤率百分比；標準差使用ddof=1，n=3。\n\n'
    text+=table(['方法','checkpoint','clean Top-1','clean Top-5','CIFAR-C Top-1','CIFAR-C Top-5'],aggregate)
    text+='## 3. 每個seed與checkpoint\n\n同列Top-1／Top-5來自同一checkpoint，沒有分別選指標最佳值。\n\n'
    text+=table(['方法','seed','checkpoint','epoch','clean Top-1','clean Top-5','CIFAR-C Top-1','CIFAR-C Top-5'],[[r['method'],r['seed'],r['checkpoint'],r['epoch']]+[f'{float(r[k]):.4f}' for k in metrics] for r in summaries])
    text+='成對改善定義為Baseline−AugMix，正值代表AugMix錯誤率較低。\n\n'
    text+=table(['checkpoint','seed','clean Top-1改善','clean Top-5改善','CIFAR-C Top-1改善','CIFAR-C Top-5改善'],[[r['checkpoint'],r['seed']]+[f'{r[k]:.4f}' for k in metrics] for r in paired])
    text+='## 4. Corruption定義與明細\n\nE[c,s]=100×錯誤樣本數/10000；平均錯誤率=(1/15)Σc(1/5)Σs E[c,s]。不是ImageNet-C以AlexNet正規化的mCE。各severity使用[10000(s−1):10000s]，五段標籤已逐項確認等於原始test labels。\n\n'
    text+='[corruption_detail.csv](results/corruption_detail.csv)包含全部900列corruption明細（六run×兩checkpoint×75）；各run/evaluation.csv另含clean，共912列。官方15類如下，其餘archive額外類型未納入平均。\n\n'
    text+=table(['corruption（last，三seed與五severity平均）','Baseline','AugMix','改善百分點'],[[r['corruption'],f"{r['baseline']:.3f}",f"{r['augmix']:.3f}",f"{r['improvement_pp']:.3f}"] for r in corruption if r['checkpoint']=='last'])
    text+=f"三seed平均下，75個corruption/severity cell 中有{int((grid>0).sum())}個改善。這是描述統計，不是75個獨立實驗。\n\n![各corruption比較](results/corruption_last.png)\n\n![severity改善](results/corruption_severity_improvement.png)\n\n"
    text+='## 5. 與論文比較及限制\n\n'
    text+=table(['方法','論文Table 1','本次last平均','差異百分點'],[[m,p,f'{lastmeans[m]:.4f}',f'{lastmeans[m]-p:+.4f}'] for m,p in [('baseline',53.3),('augmix',35.9)]])
    text+='本次改善方向及幅度與論文接近。論文數字只列至小數一位，沒有同條件的三seed原始分布可直接做等效檢定，因此不宣稱重現了相同機率分布。未找到相同模型CIFAR-100 clean Top-1/Top-5完整表格，不挪用CIFAR-10或其他模型。\n\n'
    text+='已知設定差異：現代Python/PyTorch/torchvision/Pillow；官方cudnn.benchmark=True，本次False且deterministic=True、TF32=False；每sample由seed/epoch/index決定隨機流，與官方抽樣分布一致但序列不同；修正官方test loss樣本加權及不完整resume。上述差異可能影響數值，並未證明是任何差距的原因。\n\n'
    text+='## 6. 實際時間、顯存與環境\n\n'
    text+=table(['方法','seed','訓練h','每轮驗證合計h','epoch含checkpoint合計h','peak allocated GiB','peak reserved GiB'],[[r['method'],r['seed']]+[f'{r[k]:.3f}' for k in ['train_hours','validation_hours','epoch_hours','peak_allocated_GiB','peak_reserved_GiB']] for r in resources])
    text+=f"正式600輪紀錄合計 **{allhours:.3f}小時**（含正常驗證與checkpoint主要開銷）；一致設定重評純評估 **{evalhours:.3f}小時**。從初始環境檢查到重評完成約 **{wallhours:.2f}小時**，包含下載、使用者暫停、停滯、恢復與第一版評估，低於168小時。epoch計時不包含停滯等待與epoch中断未提交的工作，也略去CSV與第二次checkpoint寫入尾端開銷；不得將epoch合計當全部wall time。\n\n"
    text+='硬體為RTX3070Ti 8GB、i7-11700K（8核16執行緒）、約32GB RAM、Windows11；driver591.86、Python3.12.14、PyTorch2.11.0+cu128、torchvision0.26.0+cu128、NumPy2.5.2、CUDA runtime12.8、cuDNN91900。driver顯示CUDA13.1為驅動支援上限，並非本次PyTorch runtime。詳見environment.json與requirements-lock.txt。\n\n'
    text+='顯存上限7.19956GiB。正式FP32保留官方batch128，AugMix每forward串接384張；preflight另測AMP開關但正式不用AMP。兩方法實測peak均低於上限。\n\n'
    text+='## 7. 測試、異常與可重現性\n\n'
    text+='9項測試實際通過，涵蓋模型、混合/獨立增強、JSD公式及梯度、Top-1/Top-5與不完整batch加權、corruption切片、AMP兩路徑，以及兩方法真實小資料的checkpoint/resume。不中斷對恢復的smoke在目前環境具有相同後續資料順序、loss、權重（最大權重差0）。未啟用全域deterministic_algorithms；這不是任何硬體或任意batch邊界的精確恢復保證。\n\n'
    text+='保存model、optimizer、scheduler、GradScaler、epoch、best狀態、設定、RNG與完整CSV rows；中斷時由最後完整epoch恢復，該epoch之後未提交工作重新執行。\n\n'
    text+='- 首次smoke的NumPy bool JSON序列化失敗已在正式訓練前修正，原始失敗日誌保留。\n- Baseline seed1在epoch68後停滯，一小時無進度觸發終止。原因未確認，沒有找到對應休眠證據。checkpoint/CSV/scheduler驗證完整，以相同程式及設定恢復epoch69–100，沒有挑結果或另換seed。\n- 早中期clean error曾短暫上升，例如Baseline seed0 epoch22增加7.54百分點；訓練loss持續下降且隨後恢復，沒有NaN/Inf。BatchNorm或學習率只屬可能原因，未完成專門因果診斷，不把尖峰直接當成已證明正常。\n- 健康監控替換狀態JSON曾遇Windows存取錯誤，已僅修正監控重試，不改訓練。\n- 首版獨立評估遺漏訓練的cuDNN/TF32設定，clean最多差0.03百分點。所有第一版明細保留，已在一致後端下完整重評12個checkpoint；最終12組clean Top-1/Top-5均與原epoch紀錄完全一致。本報告只使用一致後端結果，不混合兩版。差異表見results/evaluation_backend_comparison.csv。\n\n'
    text+='## 8. 交付與未完成項目\n\n主要範圍全部完成；未做calibration或無JSD消融，這些是選做項目。停滯根因及早期誤差尖峰機制尚未查明；沒有額外完整不中斷seed1對照。原始權重在各runs目錄，未上傳或推送GitHub。\n\n'
    text+='程式凍結版本在results/formal_source；一致後端重評入口為validation/evaluate_aligned.py；本報告核查／生成入口為validation/finalize.py。不要用原run.py eval重新產生未對齊後端的正式表格。\n\n'
    text+='![Baseline seed0訓練曲線](results/baseline_seed0_curves.png)\n\n![AugMix seed0訓練曲線](results/augmix_seed0_curves.png)\n\n其餘四個run曲線在results/*_curves.png。詳細來源：https://arxiv.org/abs/1912.02781v2 、https://github.com/google-research/augmix 、https://zenodo.org/records/3555552 。\n'
    text=text.replace('条件','條件').replace('每轮','每輪').replace('中断','中斷')
    (ROOT/'reproduction_report.md').write_text(text,encoding='utf-8')
    save_json(ROOT/'results/final_audit.json',dict(audited=time.time(),runs=audits,training_source_hashes_unchanged=True,epochs=600,evaluation_rows=912,corruption_rows=900,aligned_clean_verified=True,recorded_epoch_hours=allhours,aligned_evaluation_hours=evalhours,wall_hours_to_aligned_evaluation=wallhours,all_75_mean_cells_improve=bool((grid>0).all())))
    print(json.dumps(dict(last_corruption_means=lastmeans,improvement_pp=benefit,epoch_hours=allhours,evaluation_hours=evalhours,wall_hours=wallhours,all_cells_improve=bool((grid>0).all()))),flush=True)

if __name__=='__main__': main()
