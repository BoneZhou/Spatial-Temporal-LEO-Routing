# 本機整理版審閱說明

此資料夾已取代先前同名的核心程式版本，尚未上傳或公開 GitHub。

## 已完成

- 保留 Proposed 的 GAT、LSTM、DQN、逐封包 replay 與訓練程式。
- 保留 GraphPR-adapted、FDR-MARL-adapted、DQN-IR-adapted (fixed-link)，
  補上來源文獻、觀測／獎勵／佇列適配差異，圖表名稱也保留 adapted。
- 用 config.py 集中模擬與學習參數，用 paper_config.json 固定種子、
  訓練預算、評估時窗與輸出位置。
- 補齊主比較、嚴格消融、跨區配對、FTP burst／recovery 的 evaluation、
  aggregation、95% CI 與 plotting code。
- 加入 313 份精簡數據檔及 SHA-256 來源索引，可直接重算表格、重畫圖。
- README 提供安裝、smoke、完整訓練、正式評估與圖表重現命令。
- 清除整理包內的個人電腦路徑；不打包虛擬環境、暫存、舊版啟動腳本、
  debug 輸出、論文私人稿件或大型 checkpoint。

## 驗證結果

- 39 項測試通過，包含封包守恆、可行動作、模型消融與 checkpoint 恢復。
- 七種方法／變體的小型訓練通過；GraphPR 另以一槽更新的小型設定，
  實際確認 optimizer 有執行更新。
- 四個主方法的 tracked evaluation 小型整合流程通過。
- 主比較及消融重新產生的 mean／SD CSV 與原正式結果逐位元一致。
- 30 秒 burst 的逐 seed recovery 與注入統計逐位元一致；
  recovery 彙總最大差異約 1e-14，屬浮點尾差。
- 已重畫並檢視主要比較、消融、burst 與訓練曲線。
- 本次未重新執行完整五種子正式訓練；重畫使用既有完成實驗的測量資料。

## 審閱時特別留意

本機較早 PDF 的 queue=640、1300 episodes、120/180/240 Mbps 曲線，
與最終逐封包程式的 queue=100、一千萬決策預算及 240 Mbps 四場景
並不相同。整理版已把可重現範圍及差異列在 REPRODUCIBILITY.md，
不把舊稿全部數字標成已重現。

現有 FDR-MARL-adapted 未通過低負載行為 sanity check；為了保留原結果
的可追溯性，沒有在整理時修改獎勵再冒用原 checkpoint。公開前須決定
論文如何呈現此限制，並確認最終稿與程式設定一致。

程式與數據的公開授權尚未指定，因此未自行加入 MIT／Apache 等授權。
