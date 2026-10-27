# ICC 2027：三篇外部基線在統一環境的定稿適配規格

狀態：**2026-10-21 更新：Proposed 與三基線的逐封包介面及共同 1,000 萬次決策上限已固定；五個訓練 seeds 與 80／80 場獨立配對正式評估均完成。末段曲線不足以證明四方法全數收斂，只能稱等互動預算比較。** 這是本研究的移植規格，不是三篇作者原始模擬設定或官方數值重現；原文、原文歧義及本研究新決定分別標示。最新狀態見公開前審閱文件。統計術語以括號保留英文。


> 2026-10-23 sanity check 更新：原式壅塞分支分母為傳播延遲 τη；目前適配程式兩分支都用 queue 項，先前來源描述已更正。120 Mbps 凍結模型低負載測試 loss 75.24%，而配對最短路徑控制約 0.0063%；目前 FDR-MARL 未通過行為 sanity check，舊結果暫不作本文優越性的證據。詳見 outputs/teacher_revision_20261023/FDR_SANITY_REPORT_zh.md。本次尚未修改 reward 或重新訓練。

## 1. 不變的物理環境與比較邊界

| 項目 | 本研究固定決定 | 對外說法 |
|---|---|---|
| 星座與動態鏈路 | 3 planes × 15 satellites＝45 顆；Walker-Delta phasing factor 1；高度 570 km、傾角 70°；每 20 ms 更新軌道位置與可用 ISL；同平面前後鄰居、跨平面一對一匹配且受 LoS 與 ±60° 緯度限制。 | 三篇原文各自的星座／鏈路規則**不**在主比較中重建。 |
| 封包與佇列 | 1500 bytes；每衛星**一條** FIFO shared queue，上限 100 packets；每顆衛星每 slot 合計最多服務 36 packets，所有出口共用，不因有四條 ISL 乘四。TTL＝30 hops。 | GraphPR／FDR-MARL 原文「receiving queue＋四條 per-ISL sending queues」只保留為來源差異，不創造四條虛擬實體佇列。 |
| 鏈路容量與延遲 | 每條可用 ISL 固定 300 Mbps，**只**計算 1500-byte 封包的 transmission delay（0.04 ms）；不替代 36 packets/slot。傳播延遲依當前幾何距離；物理服務、排隊、丟棄由同一模擬器決定。 | 不捏造 SNR 或可用頻寬的動態變化。 |
| 流量 | 120／180／240 Mbps 都是**全網總負載**；訓練與測試使用預先產生、跨方法配對的 source／destination／arrival trace。跨區與 FTP 四場景沿用 `ICC2027_CROSS_REGION_FTP_SPEC.md`，只在指定正式評估中啟用。 | 路由不同可造成不同隊列與結果，但外生注入封包必須相同。 |
| 可行動作 | 當前 slot 的真實一跳 ISL 鄰居；不存在的鏈路一律遮罩（action mask）。最多四個方向，排序固定為同平面前、後與跨平面前一個、後一個 plane（mod 3）；缺邊遮罩。背景正式評估不額外施加「已拜訪節點」遮罩，交由 TTL 限制迴圈。 | 不能給某一基線額外 ISL 或事後才知道的拓樸。 |
| 評估與資訊 | 物理環境、外生 trace、時窗、佇列服務及四個正式指標相同；各方法依原意使用**不同的政策觀測**，並公開其資訊範圍。控制訊息及全域狀態取得的通訊開銷目前皆未從 300 Mbps 扣除；另記錄訊息量，不能宣稱通訊成本公平。 | 「相同模擬器」不等於「相同觀測權限」或「原論文數值重現」。 |

正式逐封包路由介面需包含 packet ID、目前節點、目的地、當前 hop、可用鄰居與當前 slot；同一 `(node, destination)` 的不同封包不可被強制合成一個訓練樣本。所有方法都經過同一介面，且評估時 epsilon＝0；若改寫此介面，須用相同 trace 檢查既有 Proposed 輸出是否等價。**既有 240 Mbps 消融結果保留其原用途，不直接與新基線結果拼成公平主表。**

## 2. 逐方法適配差異表（adaptation ledger）

### 2.1 GraphPR-adapted

來源：GraphPR 原文 PDF 第 2–4 頁，§II-A／C、§III-A–C、式 (10)–(14)、Algorithm 1；第 4 頁 §IV-A。

| 要件 | 作者原方法 | 本研究固定適配 |
|---|---|---|
| 代理與網路 | 每衛星獨立 Q agent；一層 GAT 讀入本地特徵及一跳鄰居先前 hidden state；串接目的地／封包特徵後經四層 FCNN。 | 保留 **45 組獨立政策／target／replay**、一層 GAT、hidden width 256、四個 affine FCNN layers。GAT heads 原文未報告，固定 **1 head**；鄰居 hidden state 傳遞時先 detach，不把別的 agent 梯度當成本地觀測。不得改用整張 45 節點圖直接讀全網 queue 的現有 `gat_dqn`。 |
| 佇列特徵 | 本地四條 sending queue 的即時值及移動平均，另有鄰居 hidden states。 | 本地改為**一個真實 shared queue** 的即時 `q_i/100` 及最近 50 slots 的移動平均；鄰居 hidden state 可包含該鄰居自己的同樣兩值。**不**把同一條 queue 複製四次並冒充四條出口。episode 起點重設 hidden history。 |
| 空間資訊與動作 | 本地／目的地軌道索引與位置；上、下、左、右四出口。 | 使用真實 3×15 軌道索引與當前位置，按第 1 節固定四個 action slots；缺少的跨平面 ISL 遮罩。只傳一跳 hidden message，不暴露全網即時 queue。每個 slot 更新一次 hidden message；這是由原文 1 分鐘相關狀態更新週期改為本研究短時窗的**明示適配**，另記錄訊息量。 |
| RSPH | `SH(i,d;G) − SH(j,d;G\{i})`，鼓勵避開返回當前節點的長路。 | 用**當前**公開拓樸快照、無 queue 權重的 BFS 計算；若移除 `i` 後 `j` 無路到 `d`，記 RSPH＝−30。不得用未來拓樸或未來流量。 |
| 獎勵與 loss | 非終止 `ω1·RSPH − ω2·τq − ω3·τη`；送達 Ψ＝1、丟棄 Φ＝−1；SmoothL1 loss。§IV-A 權重文字重複印出 ω2。 | 採文中三個依序數值 `ω1=0.6, ω2=0.55, ω3=0.45`，把第三個 ω2 視為排版誤植；`τq` 是實際等待時間、`τη` 是當跳傳播時間，均**除以 20 ms**化為無因次 slot 單位。成功／丟棄分別覆寫為 +1／−1。這個時間正規化是本研究選擇，不宣稱原文如此設定。 |

### 2.2 FDR-MARL-adapted

來源：FDR-MARL 原文 PDF 第 2–4 頁，§II-A、§III-A–C、式 (7)–(11)、Algorithm 1、Table I。

| 要件 | 作者原方法 | 本研究固定適配 |
|---|---|---|
| 代理與網路 | 每衛星獨立 DQN agent；三層 hidden FCNN、各 128 neurons；只知本地與一跳鄰居資訊。 | 保留 **45 組獨立政策／target／replay**，三層 128 ReLU FCNN；不使用 GAT、LSTM 或全網即時 queue。所有服務中的背景封包都可產生其所在 agent 的訓練 transition，不只採一顆 focal packet。 |
| 佇列與門檻 | 本地四 sending queues、鄰居 receiving queue；原文 `m=14, δ1=9, δ2=14`。式 (8) 卻以 `m` 作分支變數，與 Table I 的固定容量符號衝突。 | 本地與鄰居都改觀測**真實 shared queue**，不創造 sending／receiving queue。將原門檻比例映射為 `δ1=round(100×9/14)=64`、`δ2=100`。式 (8) 的分支變數明確解讀為**候選下一跳的當前 queue length `q_j`**：0–64 輕載、65–99 擁塞、100 已滿。滿佇列不預先移除 ISL 動作；依實際環境結果記 drop。 |
| ERPD 與資訊 | `ERPD(j,d)=||x_j−x_d||/c`；鄰居位置與 queue 通知。 | 以當前幾何座標計算，不看未來；每 slot 一跳分享位置與 shared queue 長度。現有軌道模型取代原文 SGP4/TLE，這是共同環境差異。控制訊息量另記但不扣 ISL throughput。 |
| 獎勵 | 式 (8) 輕載分支為 `ω1/τq + ω3/ERPD`，壅塞分支為 **`ω2/τη + ω3/ERPD`**；送達 `ω4·H`，其他 `Φ`。Table I：`0.5, 0.05, 1000, 50, Φ=0`。第 4 頁 Eq. (10) 要求最小化 MSE，但 Eq. (11) 寫成正梯度，兩者不一致。 | **既有版本尚未通過 sanity**：非終止 `r=[ω(q_j)/(1+q_j/36)+1000/(1+ERPD/20ms)]/1000`；兩分支都使用 queue，與原文壅塞分支分母不符。有界化與時間單位正規化也不是單純全域縮放；送達 `r=50×remaining_TTL/1000`、drop=0。程式以 Adam 最小化 MSE，與 Eq. (10) 一致，不機械照抄 Eq. (11) 的正梯度。此欄記錄舊 checkpoint 所用行為，不是通過驗收的定稿 reward；修改後須版本化並重新訓練。 |

### 2.3 DQN-IR-adapted（fixed-link）

來源：DQN-IR 原文 PDF 第 2–4 頁，§II、§III-B、式 (7)–(12)、Algorithm 1、Table I。

| 要件 | 作者原方法 | 本研究固定適配 |
|---|---|---|
| 網路與決策 | 一個可在節點本地使用的八層 ResNet，四個鄰居 Q 值；使用 quasi-static snapshots。 | 使用**同一組共享政策權重**在 45 節點本地決策；固定八個 affine layers＝輸入層、三個雙層 residual blocks、輸出層，hidden width **128**（原文未報告，屬本研究值）；ReLU＋Adam。改在每 slot 動態拓樸作決策，不聲稱是原 snapshot 重現。 |
| 候選特徵 | 每個鄰居的 SNR、可用頻寬、排隊延遲、曼哈頓距離、目的地旗標。 | **不產生假的 SNR**；因每條 ISL 固定 300 Mbps，可用頻寬對所有可行候選均為常數，從網路輸入刪除，並註明因此失去原文鏈路品質辨識能力。用候選下一跳當前 `q_j/36` slots 作等待時間估計；目的地旗標是 `1{j=d}`。曼哈頓距離採 3×15 軌道索引上的環狀 grid 距離：plane 最短環距＋同 plane 15 星最短環距；除以本星座最大值 **8** 正規化。缺邊遮罩。 |
| 獎勵 | 式 (11) 使用頻寬／SNR 產生的速率、queue delay、距離、hop 項及目的地旗標；式中 `F_j` 乘整個 cost，但式 (8) 定義 `F_i`，且直接乘旗標會使所有非終止步 reward 變零。 | 固定解讀為**目的地加分，不是非終止獎勵乘數**：非終止 `r=−[0.4×0.002+0.4×(q_j/100)+0.1×(grid_distance(j,d)/8)+0.1×H(h,18)]`；成功 `r=+1`，丟棄 `r=−1`。`0.002=0.04ms/20ms` 是所有 ISL 相同的傳輸成本；`H(h,18)=1/(1+exp(−(h−18)))`，18 僅為其內部 soft hop penalty，物理 TTL 仍是 30。這是公開的 fixed-link 獎勵改寫，不冒充作者原式。 |
| target 更新 | 每 500 次 learning updates，且新最佳完整路徑 reward 時額外同步；Algorithm 1 第 5 行衰減 ε、第 8 行卻寫 εmin 作探索門檻。 | 兩種 target 更新都保留；「新最佳」只比較同一訓練 traffic stream 中**已完成**路徑的累積 reward，不能看驗證／測試；探索門檻採**當前衰減後 ε**，而非永遠用 εmin。兩處都是對原文不一致處的明示解讀。 |

三個名稱在圖表、程式與正文一律加 `-adapted`；DQN-IR 再加 `(fixed-link)`。不得以舊 `main.py`／`train_dqn.py` 權重或來源不明 reward CSV 取代新適配版。四種 GAT/LSTM 消融仍只是**本文消融**，不是上述三篇外部基線。

## 3. 訓練與正式比較的執行鎖定

1. **訓練訊息與資料**：新增逐封包 transition hooks，但不改物理 queue／service；所有 agent 的經驗只來自自身路由決策。預先生成外生流量 trace，使政策不同不改變 source、destination、到達數；訓練、驗證、測試 traffic seeds 分開。相同 seed 是相同 trace，不是保證相同內生 queue。
2. **來源超參數**：GraphPR 先用原文 `lr=.001, γ=.90, replay=30000/agent, batch=2048/agent, ε0=.90, target 每 episode`；FDR-MARL 先用 `lr=.001, γ=.90, ε0=.95, εend=.001`；DQN-IR 先用 `lr=.01, γ=.90, replay=200, batch=32, ε=.60→.001（每外迴圈×.995）, target=500 updates`。原文未報告的值不標為「原論文值」；實作時先固定 `FDR replay=30000/agent, batch=128, target=500 updates` 與 `GraphPR εend=.01, decay=.995` 作**本研究初始值**。FDR ε 每 episode 以 .995 衰減至 .001，為本研究初始值。GraphPR 原文每 C slots 更新但未報告 C；本研究因 CPU 資源預檢查先明定 **C=10**，其他兩法每 slot 更新。此值不是作者原始超參數，正式定稿前仍需驗證。任何穩定性調整只能依訓練／驗證結果，記錄原值、改值、理由與等量調參機會。
3. **公平預算**：Proposed 已另建逐封包訓練版，四方法共同計數單位為**實際進入學習／replay 的 packet-hop transition**，30-slot 預熱不計。共同絕對上限固定 **10,000,000 transitions／方法／seed**；探索率按累積決策數衰減，在 8,000,000 次達各自下限。此為新回合定義下的本研究適配，不是作者原始探索超參數；完整理由、停訓規則及限制見 [逐封包公平預算](ICC2027_PACKET_FAIR_BUDGET_zh.md)。現有 Proposed 單焦點 2,000 episodes（seed 11、240 Mbps 約 4,594 hops）只列參考，**不進公平主表**。記錄實際總／每 agent transitions、optimizer updates、wall time、參數量與通訊量；若到上限仍未收斂，只報等互動預算結果，不能替單一方法加碼或宣稱收斂。
4. **正式評估**：固定權重後，先以獨立、配對的 traffic seeds 評估同一 240 Mbps 與同一四場景；如工作清單需要，再做 120／180 Mbps。主要指標只比較 throughput、packet loss rate、end-to-end delay、average queue length；每 seed 原始值、平均值（mean）± 樣本標準差（sample standard deviation）及誤差棒（error bars）均保存。跨方法 raw reward 不可直接排名，因獎勵函數不同。所有完成標記、checkpoint 與適配版本 hash 要可追溯。
5. **驗收門檻**：新增單元測試覆蓋 45 星 action mask、shared queue 上限／36 service、FDR 閾值 64／100、GraphPR RSPH 的可達／不可達情形、DQN 固定傳輸項、零延遲無除零、TTL／drop、逐封包 action、外生 trace 配對與統計帳務。通過後才啟動 seed 11 預檢查；**本文件定稿不授權立即啟動五種子訓練**。

## 4. 論文呈現用限制聲明

> GraphPR、FDR-MARL 與 DQN-IR 是依原論文演算法要件移植至相同 45 星動態拓樸、每衛星共享佇列及固定 300 Mbps ISL 的 **adapted baselines**。其原文 per-ISL queue、原星座與鏈路品質模型未在主實驗中重建；尤其 DQN-IR 的 SNR／可用頻寬特徵因固定鏈路而不可辨識。本文公開所有特徵、獎勵、閾值及訓練預算的修改，不主張重現三篇原論文的數值結果。


## Source papers

These implementations were written/adapted for this shared simulator from the
algorithm descriptions in the papers below. They are not official author code,
and their results do not reproduce the original papers' numerical results.

- GraphPR: Y. Ran, Y. Ding, S. Chen, J. Lei, and J. Luo,
  "Fully-distributed dynamic packet routing for LEO satellite networks:
  A GNN-enhanced multi-agent reinforcement learning approach,"
  IEEE Transactions on Vehicular Technology, vol. 74, no. 3,
  pp. 5229-5234, 2025.
- FDR-MARL: G. Xu, Y. Zhao, Y. Ran, R. Zhao, and J. Luo,
  "Spatial location aided fully-distributed dynamic routing for large-scale
  LEO satellite networks," IEEE Communications Letters, vol. 26, no. 12,
  pp. 3034-3038, 2022.
- DQN-IR: P. Zuo, C. Wang, Z. Yao, S. Hou, and H. Jiang,
  "An intelligent routing algorithm for LEO satellites based on deep
  reinforcement learning," IEEE VTC-Fall, pp. 1-5, 2021.

Implementation: `icc2027_experiments/baseline_methods.py`.
Machine identifiers retain their existing names for checkpoint compatibility;
display names must retain "-adapted", including "DQN-IR-adapted (fixed-link)".
The original FDR-adapted reward is retained to reproduce the saved experiments;
its failed low-load sanity check remains an unresolved research limitation.
