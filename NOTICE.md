# 來源、授權與紀錄範圍

本專案是獨立復現，不是 AugMix 作者官方儲存庫。

- 論文：Hendrycks 等，AugMix，ICLR 2020，arXiv:1912.02781v2。提供連結，不散布完整 PDF 或擷取全文。
- 官方程式：google-research/augmix，commit `9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4`。`vendor/augmix/` 保留原始檔案，排除 `.git` 和 Python 快取。
- Google 部分為 Apache-2.0，原文在 `vendor/augmix/LICENSE`。
- 第三方 WideResNet 為 MIT（Copyright 2019 xternalz）；ResNeXt/DenseNet 為 MIT（Copyright 2017 Xuanyi Dong），各自 LICENSE 保留原位。
- 本次新增復現程式、發布工具與說明依根目錄 Apache-2.0 LICENSE 提供，不改變第三方授權。
- 資料不是本專案創作；程式授權不自動改變資料授權。資料不隨儲存庫發布，請由原始來源取得。

公開副本不改實驗數值。文字事件中的本機私人路徑改為 `<USER_HOME>`／`<PROJECT_ROOT>`；環境紀錄省略不相關的桌面程序清單。未遮蔽原始紀錄及模型權重保留在原電腦。

正式程式雜湊在 `results/formal_protocol.json`，權重雜湊在 `checkpoint_manifest.json`。`results/final_audit.json` 是在具有權重的原電腦完成的稽核證據。僅靠公開副本不能重新驗證未附的權重內容；可以核對保存紀錄，或自行重新訓練。
