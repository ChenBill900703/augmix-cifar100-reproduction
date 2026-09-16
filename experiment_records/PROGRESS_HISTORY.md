# 實際進度與續接

## 最終狀態：主要復現完成

2026-09-15晚間完成全部六次100 epochs、12個checkpoint一致後端重評、912列評估、三seed統計、圖表及繁體中文報告。validation/finalize.py核查通過，見results/final_audit.json；12個checkpoint clean Top1/Top5均與原epoch紀錄一致。

最終last CIFAR-100-C：Baseline53.6551%、AugMix35.9216%，改善17.7334百分點。正式epoch合計8.9993小時，一致後端評估0.6112小時；起始到重評完成wall time17.8769小時。剩餘主要工作：無。未完成選配項目與已知原因不明事件見正式報告。

正式報告為reproduction_report.md，README已更新；程序已結束，暫時防休眠已釋放。定時檢查已停用（automation augmix，PAUSED）。下方保留歷史進度，不應將舊的「進行中」當成最新狀態。重建報告請使用validation/finalize.py；不要重新啟動pipeline來覆蓋已審查的報告。

## 六次訓練與第一版評估完成；統一後端重評進行中

最終核查發現 core.evaluate_run 沒有呼叫訓練 setup，獨立評估因此使用預設 cudnn.deterministic=False、cudnn.allow_tf32=True；訓練則為 True/False。少數 clean 指標與 epoch CSV 差0.01–0.03百分點。Baseline seed0 last 已實测：統一後端後 clean Top1=24.58%、Top5=6.98%、loss=1.1080062377929687，完全重現epoch100，定位了後端設定遺漏。

沒有修改模型、checkpoint選取或任何訓練；新 validation/evaluate_aligned.py 正在以訓練相同FP32/cuDNN/TF32設定重評全部12個checkpoint，每個run的舊評估保留在 evaluation_initial_default_backend。results/per_seed_initial_default_backend.csv 和 statistics_initial_default_backend.json 保留第一版彙總。不得混合兩種後端的數字。

目前執行程序已改為 evaluate_aligned.py，以 progress.json 的PID、logs/evaluation_aligned.log及evaluation_aligned_error.log判斷狀態，不要重啟pipeline。完成後階段為evaluation_complete_report_pending，必須再審查最終報告、README、圖表與此評估修正的揭露，才可宣稱完整交付。此程序自帶暫時防休眠，結束會釋放。

## 2026-09-15 約20:20：最後一次訓練進行中

5/6 次正式訓練完成，AugMix seed2 已超過64輪，主控制器正常。獨立健康監控曾在替換 health_latest.json 時遭遇 WinError 5 而退出；只確認存取失敗，具體是哪個程序占用未證實。已在 monitoring/health_monitor.py 加入 PermissionError 重試，最終仍失敗則保留 append-only health.jsonl 並繼續監控。正式訓練程式／參數未變。監控已重啟，最新監控錯誤檔為 logs/health_monitor_recovery_error.log，原始失敗記錄保留。

## 停滯預防與診斷補強

已新增獨立 monitoring/health_monitor.py，不修改凍結的訓練程式、模型或資料流程。每30秒保存控制器子程序CPU累積時間、RAM、I/O、GPU狀態與epoch新鮮度；超過3分鐘未完成epoch時保留診斷JSON，並未自動重啟或改參數。日誌見 logs/health_latest.json 與 logs/health.jsonl。此工具是提前取證，尚不保證能消除根因不明的停滯。

監控存活期間以 Windows SetThreadExecutionState 暫時抑制自動休眠，控制器結束／停止時釋放；不改全域電源設定，不阻止使用者手動休眠。不是認定此次原因為休眠。若將來恢復控制器，需確認健康監控也重新啟動。

## 2026-09-15 約15:40：停滯後恢復

Baseline seed1 在第68輪後無新紀錄，控制器於約15:06因一小時無進度停止；原因尚未確認，未找到對應休眠事件。先前僅用程序存活判斷正常不足，後續需同時檢查epoch紀錄新鮮度。停止資料保存在 logs/incident_seed1_epoch68，事件在 runs/baseline_seed1/events.jsonl。

已驗證last checkpoint epoch=68、CSV=68列、scheduler=26588步、權重有限、正式程式雜湊未變，以完全相同設定恢復。第69輪已實際完成並保存。恢復控制器的錯誤日誌為 logs/controller_recovery1_error.log；舊 logs/controller_error.log 保留歷史錯誤，不應當成新程序正在失敗。即時狀態以 progress.json 與最新epoch為準。

## 最新：2026-09-15 約 10:16，使用者要求繼續，已恢復

兩份資料下載及校驗完成，data_manifest.json 確認 15 corruption、五段標籤均正確。完整測試 9 passed，兩方法 epoch 邊界 resume 的後續順序、loss、權重完全相同（實測範圍限定 smoke／目前環境）。首次 smoke 發現 NumPy bool JSON 序列化錯誤，已修正，失敗日誌保留在 logs/tests_full_initial_failure.log，未影響正式run。

Baseline 五輪測速完成，epochs 2–5 平均 40.5658 秒；AugMix 五輪測速進行中。背景 pipeline 控制器運作中，真實 PID、目前子程序與剩餘時間見 progress.json；不要重複啟動。通過預算門檻後會順序執行六次正式訓練、評估和報告。正式完整结果尚未產生。

本機每30秒狀態檢查不使用模型token；另已建立每2小時的簡短 heartbeat 檢查，automation id為 augmix，正常不變時不通知，完成後停用。原始七天期限未重置，續接時約剩162小時。

下面保留前一輪暫停紀錄供追溯；其中「暫停」描述已被本節取代。

2026-09-15：使用者要求先讓資料下載，結束對話以節省 token。**暫停後續代理工作；未啟動 pipeline、正式訓練或定時自動喚醒。**

已完成：工作區與來源檢查、官方程式下載並固定 SHA、論文及附錄核對、獨立 Python/CUDA PyTorch 環境、程式初版、兩方法 FP32/AMP 顯存 preflight、7 項單元測試通過。完整 resume 測試尚未執行。

仍在下載：
- download_clean.py：官方 CIFAR-100 八段下載，完成後組合、驗 MD5、解壓並產生 clean_ready.json。
- curl：官方 Zenodo CIFAR-100-C，目標 data/CIFAR-100-C.tar.part；完成後仍須 MD5 校驗、改名與解壓。

原始紀錄：logs/download_clean.log、logs/corruption_download.log、logs/preflight.log、logs/tests_unit.log。

下載為本機程序，電腦關機／睡眠／程序被終止會影響進度。不承諾對話結束後主動監控；未建立 automation。

續接時先檢查下載程序與檔案，避免重複下載。核對 clean_ready.json，對 CIFAR-C .part 計算 MD5，正確才改名 CIFAR-100-C.tar，執行 prepare.py。然後重新審閱程式、執行完整測試（含中斷恢復）、兩方法五完整 epochs 測速，再固定正式 protocol。尚未執行任何正式訓練、corruption 評估或生成實測結果。pipeline.py 與 report.py 是待驗證初版，不可將檔案存在視為已通過。

時間預算起點：2026-09-15 約 04:25 Asia/Taipei；原定 168 小時包含等待時間。續接時計算剩餘時間，必要時向使用者說明暫停影響，不自行重置七天期限。

已停止兩個慢速重複 CIFAR 下載；日誌與部分檔保留。環境與資料全在 augmix_reproduction，不改動原 PDF、不推送或發布。
