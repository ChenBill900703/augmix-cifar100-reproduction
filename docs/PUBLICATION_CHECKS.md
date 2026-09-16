# 公開版核查

2026-09-16，在原實驗電腦的既有獨立環境執行，沒有重跑正式訓練。

- `python tools/audit_published_results.py` 通過：600 輪、912 列評估、12 組 clean 指標與保存輪次吻合；重新計算平均、樣本標準差與成對改善量；核對正式程式 SHA-256。
- `python -m pytest test_core.py -q -k 'not resume'`：7 passed、2 deselected，12.69 秒。這次公開版未下載資料，因此沒有再跑兩項需要真實資料的 resume 測試；原實驗九項完整測試的紀錄保留在 `experiment_records/logs/tests_full.log`。
- `python reproduce.py --help` 正常；實際呼叫發布入口的後端設定，確認 cuDNN benchmark 關閉、deterministic 開啟、cuDNN／matmul TF32 關閉。
- 公開副本未附資料集、虛擬環境、模型權重；保留權重雜湊清單，不宣稱僅靠公開檔案已重新驗證權重。

GitHub Actions 另在每次 push／pull request 執行標準函式庫的紀錄核查。它不下載資料、不訓練，也不代表 Linux GPU 訓練已測試。

新實驗的 `prepare.py` 會記錄當前環境並更新環境、依賴及資料清單；測試與報表程式也會產生輸出。建議使用獨立 clone，保留 Git 中的原實驗版本供比較。新實驗完成後可用 `python report.py` 從 `runs/` 產生自己的圖表與統計；不要把新結果當成本專案已發布的六次實驗。
