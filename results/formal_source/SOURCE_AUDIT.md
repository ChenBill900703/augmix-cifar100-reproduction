# 來源與設定核對（繁體中文）

論文為 arXiv:1912.02781v2，2020-02-17，ICLR 2020；使用工作區原始 PDF，全文（含附錄 A–E）文字保存在 paper.txt。Table 1 第 6 頁另已渲染並目視核對。

官方程式：https://github.com/google-research/augmix

鎖定 SHA：`9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4`。完整來源保存在 vendor/augmix，Google 部分為 Apache-2.0，第三方 WRN 為 MIT（Copyright 2019 xternalz），授權原文保留於原路徑。本次不重新宣稱原始研究或模型作者權利。論文與資料不因程式授權而自動取得 Apache 授權。

|項目|論文／補充資料|官方 CIFAR 程式|本次固定設定|
|---|---|---|---|
|模型|§4.1 WRN-40-2|40 layers，widen 2，dropout 0|原檔直接匯入；100 類別，2,255,156 參數|
|初始化|未詳列|Conv normal std=sqrt(2/(k²*out_channels))；BN weight 1/bias 0；Linear bias 0，weight 保留 PyTorch 預設|沿用；現代 PyTorch 隨機實現不保證與舊版逐位相同|
|epochs|WRN 100；其他架構有 200|100|兩方法各 100，seeds 0/1/2|
|batch|主文未詳列|128，eval 1000|128，eval 1000；無累積、無拆分三视圖|
|optimizer|SGD、Nesterov；非 Mixup wd .0005|lr .1，momentum .9，wd .0005，Nesterov|相同|
|排程|cosine|每個 optimizer step 後 scheduler.step；lr .1→1e-6；100×391 steps|相同公式及更新順序|
|基本增強|先左右翻轉與裁切|RandomHorizontalFlip → RandomCrop(32,padding=4)|相同，兩方法共用|
|正規化|未詳列|ToTensor，再各 channel mean/std=.5|採用 cifar.py；不混入 NumPy 示範檔中的 CIFAR-10 mean/std|
|操作|附錄 C；不使用 contrast/color/brightness/sharpness/Cutout|autocontrast/equalize/posterize/rotate/solarize/shear_x/shear_y/translate_x/translate_y|直接使用官方 9 操作；all_ops=False|
|混合|width=3，depth 1–3；Dirichlet/Beta α|severity=3，width=3，depth=-1，α=1|相同；各操作有自身幅度映射與隨機符號|
|附錄 α 差異|附錄 A 的 ImageNet ablation α=.5、90 epochs|CIFAR α=1、100 epochs|不挪用 ImageNet ablation 設定|
|JSD|原圖與兩 AugMix 的三分布，CE+λJSD|λ=12；mixture clamp [1e-7,1] 再 log；KL batchmean|相同，所有分支及 mixture 均無 detach；CE 只用基本增強後的原圖|
|BN|模型含 BN|三视圖 concat，一次 forward|相同；AugMix 有 384 張而 Baseline 128 張，此為方法必要差異|
|AMP|未採用|FP32|正式 FP32；AMP 僅獨立功能與顯存測試|
|選模|未詳列 checkpoint 規則|每 epoch 看 test accuracy，嚴格改善才存 best；最後 corruption 評估使用 last|本次同規則存 best，主比較 last；額外完整報 best。test 參與選模，非完全獨立 holdout|
|eval 指標|CIFAR-C 平均錯誤率|15 corruption，每種全部 50,000；clean 只報 Top-1|另報 Top-5、severity；同一列來自同一 checkpoint|

## 已核對的程式差異及影響

1. 官方 test loss 把各 batch 的 mean loss 加總後除以樣本數，未乘 batch size；本次改成 CE sum / N，Top-1 邏輯無此問題。訓練 loss 由官方 EMA 改記樣本平均，並拆 CE/JSD，不改反向傳播 loss。
2. 官方 resume 未保存 scheduler／RNG，且會重設 best_acc 與覆寫 CSV。本次保存完整狀態，last checkpoint 內的 rows 為 CSV 恢復依據；提供 epoch 邊界恢复，epoch 中斷則重跑該 epoch。
3. 官方 torch.manual_seed(1)、np.random.seed(1)；本次公平跑 0、1、2，採每 seed/epoch/sample 獨立隨機流，使 worker 分派不影響增強抽樣。分布相同但不追求官方隨機序列逐步相同。
4. 官方 cudnn.benchmark=True；本次 False、cudnn.deterministic=True、TF32=False。未啟用全域 torch.use_deterministic_algorithms，因此固定 seed 本身不代表所有 CUDA 操作皆確定；精確續訓只依實測範圍陳述。
5. 單 GPU 不包 DataParallel；保持原模型算子、串接與 BN batch 組成。現代 PyTorch、torchvision、Pillow 可能產生版本數值差異，不能斷言為任何結果差距的原因。
6. PyTorch 1.2.0 原始 autograd derivatives.yaml 的 kl_div 同時註冊 self 與 target 導數；保存核對原檔於 results/pytorch120_derivatives.yaml。本次未將 target detach，測試 JSD 公式與三支梯度。

## 可直接比較的論文數字

Table 1：CIFAR-100-C／WideResNet Standard **53.3%**，AugMix **35.9%**，差 **17.4 百分點**。這些是論文值，絕不是本次實測。未找到相同模型的 CIFAR-100 clean Top-1/Top-5 表格，因此標記未提供，不以 CIFAR-10 或其他模型補值。

## 資料與平均定義

CIFAR-100：https://cave.cs.toronto.edu/kriz/cifar.html ，Python archive MD5 `eb9058c3a382ffc7106e4002c42a8d85`。

CIFAR-100-C：https://zenodo.org/records/3555552 ，archive MD5 `11f0ed0f1191edbf9fa23466ae6021d3`。每 corruption 為 50,000×32×32×3；severity s 使用 `[10000(s−1):10000s]`；labels.npy 應為 50,000 且每段等於原 test labels，實際通過與否以 data_manifest.json 為準。

納入 gaussian_noise、shot_noise、impulse_noise、defocus_blur、glass_blur、motion_blur、zoom_blur、snow、frost、fog、brightness、contrast、elastic_transform、pixelate、jpeg_compression。其餘 archive 類型明列但不納入。

`E[c,s] = 100 × 錯誤樣本數 / 10000`；`mean corruption error = (1/15) Σc (1/5) Σs E[c,s]`。這是不經參考模型正規化的百分比，與 ImageNet-C AlexNet-normalized mCE 不同。論文正文 uCE 的文字稱 average，但顯示公式寫 sum；本次依官方實作與 Table 1 的百分比採 75 個 cell 的平均，明列 1/5 因子。

跨 seed 標準差使用 ddof=1；每 seed 改善 = Baseline error − AugMix error（百分點）。未完成 3 seeds 時明列 n，不作三次結果或統計等同宣稱。
