# 檔案清單（ASSETS）：Rank（競賽二）

2026-11-26 在原專案實測（大小與 sha256 由腳本計算）。`colab/assets_manifest.json` 是同一份清單的機器可讀版（`category` 欄分類）；`colab/reproduce_final.ipynb` 依它下載、驗 sha256。
本 repo **不含**下列任何檔案。每個檔案屬於三類之一：

| 類別 | 做法 |
|---|---|
| **自訓權重（公開 Release）** | 我們自己訓練出來的模型參數。放在這個 repo 的 GitHub Release，下載網址 `RELEASE_URL`：**待補**（Release 建好後填入 `colab/reproduce_final.ipynb` 與 README） |
| **官方資料衍生中間檔（僅提供主辦方）** | 由主辦方的官方棋譜算出來的特徵、向量、各模型對測試題／驗證題的機率與預測檔。**不放 repo、不放 Release**；可應主辦單位要求以私人連結另行提供。沒有它們時如何重新產生，見 README |
| **外部（不散布，附下載網址）** | KataGo 執行檔與網路、HumanSL 檢查點、Strength Estimator 權重。我們不再散布，只給官方網址與放置路徑（第 3 節） |

分類依據：只含「模型參數」（權重、投影矩陣、迴歸係數）者為自訓權重；凡含官方棋局的內容、由官方棋局逐盤算出的數值、對官方題目的預測，或內含官方玩家 id／標籤者，一律列為中間檔。
**Release 上的檔名**：Release 沒有資料夾，清單的「資產名」就是 Release 上的檔名（`/` 已換成 `__`）。**放回 repo 的位置**是相對 repo 根（`data/`、`.scratch/` 約定見 README）。

## 1. 自訓權重（公開 Release）

| 資產名 | 用途 | bytes | sha256 | 放回 repo 的位置 | 最後一步需要 |
|---|---|---:|---|---|:---:|
| `rank__stage2_fit.pkl` | Rank：E117 第二階段（v12）擬合結果（pickle，內含 e117_core.Stage2） | 1,778,533 | `c8a0ce87763d19f778ebe360c1be23597fd30dc0fa5594abe3fe80d81ad2bb3a` | `.scratch/e117/final/stage2_fit.pkl` | 是 |
| `finh_full__final_model.pt` | S 認人網路（測試端；7,365 類） | 15,930,521 | `fec071cad0cabc7aacbecbb81297dfc3ac87ca51e1b73bcbb01b0828252b2a59` | `.scratch/e90/colab/finh_full/final_model.pt` | 否 |
| `final_s0__last.pt` | Rank 讀法網路 ft4（交件版） | 74,724,967 | `d058726aed62c629782127977f3f061f1ef96b0ac6ef616373c642f2a618c199` | `.scratch/e98/runs/final_s0/last.pt` | 否 |
| `ft4_s0__last.pt` | Rank 讀法網路 ft4（驗證版；選用） | 74,724,175 | `d433ad6e1686c344a58cda922955392f836eb75171f1afdf2ead1d00e5f3aeba` | `.scratch/e98/runs/ft4_s0/last.pt` | 否 |
| `rank_up__c1_fit.npz` | C1 邏輯迴歸混合的擬合結果 | 2,508 | `4ea8bd3b86452caaf65a9a8100147f02f8e5f781f1407b2b8eed29dc3a262a3c` | `.scratch/e103/c1_fit.npz` | 否 |

小計：5 檔，167,160,704 bytes（167.2 MB）；最大單檔 `final_s0__last.pt`（74,724,967 bytes）。沒有單檔超過 GitHub Release 的 2 GB 上限。

**公開前的處理**：`final_s0__last.pt`、`ft4_s0__last.pt` 原檔的訓練參數（`args`／設定）裡記有作者本機或遠端主機的絕對路徑；公開版已把這些字串替換掉（帳號目錄前綴 → `<home>`、專案根目錄前綴移除），**所有張量與數值逐項相同**（上傳前以程式逐一比對）。因此這幾個檔的 sha256 與作者內部紀錄的不同，**以本表為準**。

**注意**
- `.pkl` 檔（pickle）載入時會執行程式碼；請只使用本 repo Release 的檔案，並先核對 sha256。
- 以 KataGo HumanSL 檢查點微調的權重（Rank 的 `final_s0`、`ft4_s0`）屬於衍生自 HumanSL 網路的模型，使用時請一併遵守 KataGo Neural Network License（見 README「外部資源」）。

## 2. 官方資料衍生中間檔（僅提供主辦方；不在 repo、不在 Release）

| 資產名 | 用途 | bytes | sha256 | 放回 repo 的位置 | 最後一步需要 |
|---|---|---:|---|---|:---:|
| `rank__val_preds.npz` | Rank：驗證側（fold 4，4,568 題）六組機率＋標籤，用來擬合第二階段 | 4,134,402 | `97244084a7a3b8054a31bac29b43653627636ef2de18d19d0dfd82ecd45bdc8b` | `.scratch/e103/val_preds.npz` | 是 |
| `rank__test_inputs.npz` | Rank：測試側 800 題六組機率 | 447,460 | `07e519b80f1804a9bf5dc4aad700cbc6fa6f8c7598d4dd3ab5690751fad2016e` | `.scratch/e117/final/test_inputs.npz` | 是 |
| `ref__rank_v13n.csv` | Rank 最終上傳檔 | 11,258 | `3a9525a3e7e6273727d0e5fd0a6fc7d5fc2921b17c7687b36f83e961c9d13919` | `ref/rank_v13n.csv` | 否 |
| `rank_up__final_s0_test12.npz` | Rank 讀法網路測試端輸出 | 6,387,014 | `24b22501fbc8d9128df819e89084f47982f9f4df12c997d6ba5c7ff9d3eaa20c` | `.scratch/e103/exports/final_s0_test12.npz` | 否 |
| `rank_up__S_bank_opp_full.npz` | Rank 對手側向量庫（C2 近鄰票） | 1,031,686,623 | `9f82fb6788c03af519bf062eae4c9d19bd89dea6edd66271648c88be019cf03e` | `.scratch/e103/bank/S_bank_opp_full.npz` | 否 |
| `rank_up__bank_meta_full.npz` | 向量庫的中繼資料 | 14,924,859 | `2bcb98ddea6e56d6730c06c7fec0129dd4631ee3379fb592e0760c7718698096` | `.scratch/e103/bank/bank_meta_full.npz` | 否 |
| `rank_up__head_b.npz` | Rank 認人網路分類頭（測試端） | 4,310,889 | `f68552f76faeb429e4a316bf1c1a4e40acfb46a58c7bac50c0df1a2ef92f12c8` | `.scratch/e91/b/head_b.npz` | 否 |
| `rank_up__S_test_own_b.npz` | Rank 測試盤己方向量 | 10,371,248 | `57b5728eecec10fb6022ce4ae317c1d269aca2c72c8d7860d3d65f46185b9021` | `.scratch/e91/b/S_test_own_b.npz` | 否 |
| `rank_up__S_test_opp_b.npz` | Rank 測試盤對手向量 | 10,371,248 | `e535cb3ff68d811658cb911d2cfba5bafb72c6e75bce7074961e5555176594f7` | `.scratch/e91/b/S_test_opp_b.npz` | 否 |
| `rank_up__test_m03.npy` | Rank 腿 M03T2 測試輸出 | 64,128 | `2e7bf261ed63cf853f5a11dd6e70fc9a309f8f43e1ee76ca32b08856bb67047b` | `.scratch/e66/final/test_m03.npy` | 否 |
| `rank_up__test_m04.npy` | Rank 腿 M04T 測試輸出 | 64,128 | `03aa0056e690411c4e487ff0cb2decc60238cdf34fce7f0b637fd9fa3ff365ec` | `.scratch/e66/final/test_m04.npy` | 否 |
| `rank_up__set_A1T2.npy` | Rank 腿 A1T2 測試輸出 | 960,128 | `6babdbabe0ce5326d55f32d6ce479ffd8d8a40fe4647ad5cc98a8b9c4a395e4e` | `.scratch/e66/final/set_A1T2.npy` | 否 |
| `rank_up__set_A1P.npy` | Rank 腿 A1P 測試輸出 | 960,128 | `ee93bc28af4f19f5981798fae0663f6a78ffc487d05005743cc676caad35bca3` | `.scratch/e79/final/set_A1P.npy` | 否 |
| `rank_up__aux.npz` | Rank 驗證框架輔助表（e105_core.load_val 讀取） | 45,203,639 | `0a5460e72ffe721c3f29540c7338154264125c6a3546b9ffa313b26c596938cf` | `.scratch/e29/bundle/aux.npz` | 否 |

小計：14 檔，1,129,897,152 bytes（1129.9 MB）；其中最後一步需要的 2 檔共 4,581,862 bytes（4.6 MB）。最大單檔 `rank_up__S_bank_opp_full.npz`（1,031,686,623 bytes）。

**取得方式**：這些檔案只提供給主辦方（私人連結）。放進同一個資料夾後，把 `INTERMEDIATE_DIR` 指到該資料夾；`colab/colab_utils.py` 的 `fetch_intermediate` 會依「資產名」或「放回 repo 的位置」找檔、複製到 repo、驗 sha256。

## 3. 外部（不散布；官方下載網址與放置路徑）

| 資源 | 官方下載網址 | bytes | sha256 | 放置路徑（相對 repo 根） | 授權 |
|---|---|---:|---|---|---|
| KataGo 執行檔 v1.18.2 | https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip | 51,350,638 | `2cb197fc9bc2050f29112abb5b83cdbad63936cb4605d6e199772aedc9806c02` | `data/external/katago/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip` | MIT 式（KataGo） |
| KataGo 主網路 | https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz | 97,898,094 | `9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d` | `data/external/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` | KataGo Neural Network License（MIT 式） |
| KataGo HumanSL 網路 | https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.bin.gz | 99,066,230 | `637746e44f0efe00ad1245a50aa9bbf0716efe364c43965ead97bd6835d84ab5` | `data/external/katago/b18c384nbt-humanv0.bin.gz` | KataGo Neural Network License（MIT 式） |
| HumanSL PyTorch 檢查點 | https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.ckpt | 323,752,318 | `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27` | `data/external/katago/b18c384nbt-humanv0.ckpt` | KataGo Neural Network License（MIT 式） |
| Strength Estimator 權重（中研院） | https://rlg.iis.sinica.edu.tw/papers/strength-estimator/assets/models/go.tar.gz | 448,813,641 | `885bea594a62956958de526f6a2073f48ecd181535b4d9e72d2c71a339172caf` | `data/external/se/download/go.tar.gz` | 程式庫 GPL-3.0；權重未見授權明示，我們只從權利人公開頁面下載使用，不再散布 |

`go.tar.gz` 內只用 `se_go.pt`（解出後放 `data/external/se/extracted/se_go.pt`，sha256 `95fc295137b1fd09a0ec4d5cc79ed888f8c12d0ba6a77fea58b24efc3f418126`）。`colab/colab_utils.py` 的 `fetch_external()` 會下載並驗證以上全部。

## 已不存在、無法提供的中間檔（要重算）

| 檔 | 用途 | 影響 |
|---|---|---|
| `ft4_s0_val12.npz`（`.scratch/e98/exports/`） | 驗證側 `val_preds.npz` 的上游 | 驗證側六組機率只能用保存版（中間檔） |
| `se_uid.parquet`（`.scratch/e21/`） | 訓練端 SE 表（685,231 盤，GPU 12.9 小時） | Rank 的 SE 腿重訓需要 |
| `X120_full.npy`、`G_full.npy`、`tok_full.npy`、各種 `pos_*_{trA,trB,val}.npz` | 訓練端 HSL W120 表、主幹向量、逐局面向量 | 要重訓任何網路都得先重新編碼（約 70 小時，估計值） |
| Rank 的 M03T2／M04T 模型 pkl 與 A1T2／A1P 各 15 個集合模型檔 | Rank 四條腿的訓練結果 | 只剩各腿的測試端輸出（中間檔 `test_m03.npy` 等）；要重算需從訓練端特徵重訓 |
| `humanv0.ckpt`、KataGo 網路與執行檔 | 外部資源 | 重新下載（第 3 節，附 sha256） |
