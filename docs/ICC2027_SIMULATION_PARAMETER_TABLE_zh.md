# ICC 2027 稿件用：星座、佇列、流量與評估設定表

更新：2026-10-22。本表描述**目前四方法逐封包主比較已實際使用**的統一模擬器與 240 Mbps 正式評估。120 Mbps 舊單焦點消融／FTP 與 180 Mbps 先前預檢查是其他實驗軌，不得混成同一負載曲線或模型排名。統計術語於括號保留英文。

## 表 1：物理網路與封包處理

| 類別 | 參數 | 本研究固定值／實作含義 |
|---|---|---|
| 星座 | Walker-Delta | `70°:45/3/1`：45 顆衛星、3 個軌道平面、每平面 15 顆、phasing factor 1；高度 570 km、傾角 70°。 |
| 軌道 | 地球半徑／重力參數／週期 | 6,371 km；`398600.4418 km³/s²`；圓軌道週期約 5,754.98 s（95.916 min）。RAAN 分別 0°、120°、240°，平面間相位差 8°。 |
| 時間 | slot 與拓樸更新 | 每 slot **20 ms**；每一個 slot 依當前軌道位置更新可用 ISL。 |
| 同平面 ISL | 連接規則 | 每星固定連接環狀軌道的前後相鄰衛星，至多 2 條同平面鄰邊。 |
| 跨平面 ISL | 連接規則 | 需有視線（line of sight, LoS）且兩端星下緯度符合 `|latitude|≤60°`；候選按距離排序，使用確定性的最大基數一對一匹配。每星至多 2 條跨平面邊，總節點度數至多 4；**不是**每個 slot 每星必有 4 個鄰居。 |
| 封包 | 大小／跳數上限 | 1,500 bytes；最多 30 hops，超出按 TTL 丟棄。 |
| 佇列 | 共用容量與紀律 | **每衛星一條** FIFO shared queue、容量 **100 packets**；不是每條 ISL 各有 100，也不是原論文的 640。 |
| 服務 | service rate | 每衛星每 slot **合計至多 36 packets**，所有出口共用；不是 `36×ISL 數`。 |
| ISL 容量 | transmission delay | 每條可用 ISL 固定 **300 Mbps**，僅用於 1,500-byte 封包的傳輸時間 **0.04 ms**；**不**替代前列 `36 packets/slot` 的佇列服務。傳播延遲由當前幾何距離決定；另設節點處理時間 **0.10 ms**。 |

**新增研究假設**：phasing factor 1、跨平面 `±60°` 限制與確定性最大基數匹配，是本次修稿補足原稿未明示的設定；不能描述為原投稿已完整規定。`100 packets` 共享佇列與 `36 packets/slot` 是目前實驗設定；舊稿／其他論文中的 `640` 或 `500 packets/slot` 不可當作已執行值。

## 表 2：外生流量與四場景

| 類別 | 參數 | 本研究固定值／配對規則 |
|---|---|---|
| 背景負載 | Offered load | **240 Mbps 全網總負載**用於目前四方法訓練與正式四場景評估，不是每個 source 240 Mbps。按 1,500-byte 封包換算，名目平均約 **400 packets/slot 全網**。120／180 Mbps 也定義為全網總負載，對應約 200／300 packets/slot，但**不是**目前這批四方法正式結果的測試負載。 |
| 背景到達 | 非齊次 Poisson（NHPP） | 45 個 source 各自抽 Poisson 到達，平均分配全網 offered load，強度依 source phase 起伏：`λ_i(t)=λ̄[1+0.35 sin(2πt/30+2πi/45)]` packets/slot，週期 30 slots；不同方法使用預產生的相同外生 arrival trace。 |
| Region | 定義 | 三個 orbital-plane groups：0＝衛星 0–14，1＝15–29，2＝30–44。這是**軌道平面分組**，不是地理行政區。 |
| Source–destination | 45 條固定 flow | 每衛星一條 source flow；同區目的地於同平面其他 14 星等機率抽樣；跨區目的地從其他平面均衡選擇，再於該平面 15 星等機率抽樣。同一次 traffic realization 的每條 flow 目的地固定。 |
| 四場景 | S1–S4 | S1 同區＋正常；S2 跨區＋正常；S3 同區＋FTP burst；S4 跨區＋FTP burst。同一配對 seed 中，同／跨區只更換目的地映射；正常／FTP 只更換 FTP 額外注入，背景 arrivals 相同。 |
| FTP 文件 | 檔案大小 | 名目 2 MB；模擬時按 1,500-byte 封包離散化為 **1,334 packets＝2,001,000 bytes**，比名目值多 0.05%。 |
| FTP 工作階段 | Arrival 與注入 | burst 期間全網 Poisson session 到達率 **3.75 sessions/s**；同 source 同時至多一個 session，每 session 注入上限 **20 packets/source/slot**。名目長期等效增量約 **60 Mbps**，但 5 s burst 中實際 session 數與注入量是隨機的，檔案尾部可能延至 burst 後；不能稱 FTP 場景總負載恆為 300 Mbps。 |
| FTP 來源 | 規格界線 | 2 MB 與 Poisson session 概念受 3GPP FTP Model 1 啟發；此處 `3.75 sessions/s`、5 s burst、注入 pacing 與恢復門檻均為**本研究自訂**。 |

## 表 3：共同訓練、獨立測試與指標

| 項目 | 本研究執行規格 |
|---|---|
| 訓練時窗 | 每回合先 30 slots（0.6 s）預熱，再 100 slots（2 s）逐封包訓練；預熱不計入 1,000 萬可學習 transitions 上限。訓練 seeds＝11、22、33、44、55；所有方法共同採該上限與 800 萬 transitions 的 epsilon 衰減尺度。此處**不是**舊版 `1,300/2,000 episodes` 設定。 |
| 正式測試分割 | checkpoint seeds 11／22／33／44／55 分別配對**未參與訓練**的 traffic seeds 101／202／303／404／505；四方法在每一對 seed 與每場景用同一預產生的 trace、凍結權重、`epsilon=0`。每方法／場景 **n=5**；四方法 × 四場景 × 五對＝80 場。 |
| 正式測試時間軸 | 30-slot 預熱（不列主要量測）；接著正常 baseline 250 slots（5 s）、burst 或對應空白段 250 slots（5 s）；已啟動 FTP 可繼續注入直到檔案結束；以**最後一個 FTP 封包注入後的下一 slot**為恢復起點，再觀察至多 500 slots（10 s）。正常場景使用配對的等長時間軸。 |
| 恢復判定（recovery） | 基準 `Q0` 是 baseline 最後 100 slots 的每衛星平均 queue。20-slot queue 移動平均需先高於 `Q0+max(0.1Q0,2 packets/衛星)` 才算有明顯偏離；之後需降回門檻以下並維持 50 slots。無偏離則不計恢復時間；500 slots 內未恢復標為右設限（right-censored），不強填 10 s。 |
| 吞吐量（throughput） | 量測期成功交付封包的位元數／量測總時長，單位 Mbps。FTP 場景的量測總時長包含實際檔案注入尾段。 |
| 封包損失率（packet loss rate） | 量測期出生且確定丟棄數／量測期生成數；量測終點仍在網路中的封包另列 `in_flight`，**不**算損失。逐組核對 `generated=delivered+dropped+in_flight`。 |
| 端到端延遲（end-to-end delay） | 僅對成功交付封包計算離散時槽駐留時間：`(delivery_slot−birth_slot+1)×20 ms`，再取平均。逐跳傳播、傳輸、處理時間只作診斷，**不**再加到此值以免重複計算；此口徑與舊單焦點實驗的分項加總延遲不同。 |
| 平均佇列長度（average queue length） | 所有量測 slots、所有衛星之**服務後**共享 queue 封包數的平均，單位 packets/衛星。 |
| 統計彙整 | 每方法／場景對五對 seeds 的原始四項指標等權計算平均值（mean）± 樣本標準差（sample standard deviation）；圖中誤差棒（error bars）以同一口徑繪製，不把 80 場當 80 個獨立 seeds。 |

正式四項指標與逐封包帳務見[正式評估報告](REPRODUCIBILITY.md)；跨區／FTP 的原始規格與版本差異見[跨區 FTP 規格](REPRODUCIBILITY.md)。120 Mbps 舊消融結果可獨立呈現為探索性／補充結果；若要聲稱**新四方法**在 120 Mbps 的表現，仍須另用凍結 checkpoint 做獨立配對評估。180 Mbps 的五種子比較亦尚未成為本表所述正式結果。
