# ICC 2027：四方法逐封包訓練與公平預算鎖定

日期：2026-10-20；進度更新：2026-10-21。此協定在啟動正式五個 seeds 前固定；seed 11 的四方法長曲線已檢查，末段不足以證明四方法均收斂，故目前只稱**等互動預算比較**。統計術語以括號保留英文。

| 項目 | 固定做法 |
|---|---|
| 方法 | Proposed GAT-LSTM-DQN、GraphPR-adapted、FDR-MARL-adapted、DQN-IR-adapted (fixed-link)；均在共同逐封包環境從頭訓練。舊 Proposed 單焦點 checkpoint 不進公平主表。 |
| 計數單位 | 一個實際進入 replay／學習的封包 hop 決策＝一個 trainable transition。30-slot 預熱的路由不計。每回合預熱 30 slots，訓練 100 slots；四方法同一 seed、load 與外生 trace。 |
| 共同上限 | **10,000,000 trainable transitions／方法／seed**，不准超過；於完整回合邊界停下，因此各方法實際數可略低於上限，須逐項報告。任何方法不能單獨加碼。 |
| 探索率（epsilon） | 保留各方法來源／適配規格的起點與下限；因新逐封包回合與舊單焦點回合長度不同，本研究統一按累積 trainable transitions 衰減，在 **8,000,000 transitions** 達各自下限，餘下最多 2,000,000 transitions 作低探索訓練。此為**本研究適配設定，不是三篇作者原始超參數**。 |
| 模型／更新 | Proposed 保留原六維節點特徵、GAT＋LSTM、reward 係數、batch 128、replay 100,000、target 每 200 updates；每 slot 一次 optimizer update。GraphPR 保留 batch 2,048／agent、replay 30,000／agent；未報告的更新間隔本研究設 C=10 slots。FDR／DQN-IR 依適配規格。 |
| 公平調參 | 在五 seeds 前只用 seed 11 與獨立驗證流量檢查穩定性；四方法同等調參試驗數。此協定的 10M 上限不因單一方法曲線不佳而私自增加。若 10M 時未收斂，主表只稱「等互動預算（fixed-budget）」；收斂比較須另行**對四方法一起**預先定義較長上限並重跑／續跑。 |
| 儲存／恢復 | 每 5 回合完整保存模型、target、optimizer、replay 與亂數狀態；每回合寫曲線 CSV。中斷後從最近 checkpoint 恢復，可能重算最多 4 回合，並以 checkpoint 截斷較新的未提交 CSV。正式輸出記錄程式來源 SHA-256、每回合 trace hash、實際 transitions、optimizer updates、耗時。 |

**設定依據與限制**：seed 11、240 Mbps 的六回合預檢查顯示四方法各約 95.6–96.6 萬 trainable transitions，約每回合 15.9–16.1 萬；reward／loss／delivery ratio 尚有趨勢或波動，不能以六回合宣稱收斂。10M 上限約為預檢查的十倍，並留出探索率到下限後的訓練區間；這是可負擔且事先共同固定的**互動預算**，不是從已確認收斂點回推的門檻。GraphPR 第六回合完整 checkpoint 約 4.76 GB，長訓可能接近 replay 上限而達約 8 GB；需監看儲存、RAM 與 GPU。正式評估仍要使用獨立、配對的 traffic seeds 與四個主指標。

**2026-10-21 執行紀錄**：seed 11 四方法已完成，獨立驗證流量 seeds 601–603 已通過配對與封包帳務檢查。使用者要求啟動五個訓練 seeds，故把 seed 11 計入五組，新增 seeds 22、33、44、55。四方法沿用上述超參數與共同決策上限；本輪各方法**追加調參試驗數均為 0**，不以驗證或正式測試流量挑選新設定。新增四組分成兩路互不重疊的 GPU 訓練：A 路 seeds 22／44，B 路 seeds 33／55；大型 GraphPR 訓練以互斥鎖避免兩路同時執行。seed 22 Proposed 原先單路執行至第 10 回合完整 checkpoint 後切換，從該 checkpoint 續跑。新 seeds 的 FDR-MARL／DQN-IR 使用 GPU；已完成的 seed 11 兩方法是在 CPU 訓練，這是裝置差異，後續須保留在再現性紀錄中。這一輪仍只支持等互動預算比較，不因此宣稱全部收斂。
