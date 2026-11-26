# AI CUP 2026 圍棋 競賽二：從棋譜預測玩家棋力等級（Rank）— 程式碼

AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽」競賽二的程式碼：資料處理、特徵／編碼、訓練、預測、最終組合，以及重現最終提交檔的 Colab 筆記本。
(English: code for AI CUP 2026 Go, competition 2, predict a player's rank level from games (Rank): data processing, features/encoders, training, inference, final assembly, and Colab notebooks to reproduce the final submission. License: GPL-3.0.)
另一題是獨立的 repo `aicup2026-go-player`；兩個 repo 各自能獨立安裝、跑測試、跑最後一步，共用程式庫（`src/gogame/`）兩邊各放一份。

| 題目 | 說明 | Private 名次 | Private 分數 |
|---|---|---|---|
| 競賽二 Rank | 預測一組棋譜的玩家段位（10 級） | 第 29 名 | 0.7286153 |

最終上傳檔：**Rank v13n**。本 repo 的 `final/` 能從保存的中間檔**逐位元組重現**這個檔（sha256 `3a9525a3e7e6273727d0e5fd0a6fc7d5fc2921b17c7687b36f83e961c9d13919`；見「驗證」）。


> **先講三件事**
> 1. **自訓權重**（5 個檔，167.2 MB，最大單檔 74.7 MB）放在這個 repo 的 GitHub Release，下載網址 `RELEASE_URL`：**待補**。
> 2. **由官方棋譜算出來的中間檔**（特徵、向量、各模型對測試題／驗證題的機率）**不公開**：不在 repo、不在 Release；可應主辦單位要求以私人連結另行提供（見「官方資料衍生的中間檔」）。
> 3. **外部模型**（KataGo 網路與執行檔、HumanSL、Strength Estimator 權重）**不散布**，只列官方下載網址與放置路徑（見「外部資源」）。
>
> 另外：從棋譜整條重跑需要先重新編碼約 70 小時（見「已知限制」）。能直接重跑的是「中間檔 → 最終 CSV」這一段；更上游的每一步都附程式與說明。

## 授權

本 repo 採 **GNU General Public License v3.0（GPL-3.0）**，全文見 [`LICENSE`](LICENSE)（取自 https://www.gnu.org/licenses/gpl-3.0.txt ）。

選用 GPL-3.0 的原因之一：`scripts/experiments/e21/se_encode.py` 是依 Strength Estimator（rlglab/strength-estimator，GPL-3.0）與 MiniZero（rlglab/minizero，Apache-2.0；GitHub 回報）公開原始碼的行為，以 Python 重寫的輸入平面轉換（非逐字複製；文件字串逐段引用其原始碼行號），與 GPL-3.0 相容。

第三方元件與其授權（本 repo 不含其程式碼或權重，僅下載連結與使用）：

- KataGo 程式：MIT 式（Copyright David J Wu (lightvector) and/or other authors）；`cpp/external` 內第三方元件各有授權。
- KataGo 網路（kata1 與 Human SL）：KataGo Neural Network License（MIT 式，https://katagotraining.org/network_license/ ）。
- Strength Estimator：**程式庫 GPL-3.0；權重未見授權明示，我們只從權利人公開頁面下載使用，不再散布。**
- MiniZero：Apache-2.0（GitHub 回報）。
- 使用的開源套件：numpy、scipy、pandas、pyarrow（BSD/Apache）、scikit-learn（BSD）、LightGBM（MIT）、PyTorch（BSD）、sgfmill（MIT）、PyYAML（MIT）、numba（BSD）。

## 目錄

```
README.md                  本檔
LICENSE                    GPL-3.0 全文
PIPELINE.md                管線：步驟、輸入輸出、耗時、硬體；末節「元件手冊」逐元件列訓練／推論指令、輸入、輸出
ASSETS.md                  檔案清單：自訓權重（Release）／官方資料衍生中間檔（僅主辦方）／外部（官方網址），含大小與 sha256
requirements.txt           CPU 環境（鎖版本）
requirements-gpu.txt       GPU 環境（torch 2.12.0 等）
src/gogame/                共用程式庫（SGF 解析、棋盤、L0 特徵、聚合、模型、提交檔寫入與檢查）
tests/                     單元測試（`python -m pytest -q`）；fixtures 是合成棋譜，不含官方資料
scripts/                   各階段腳本（保留原本目錄結構，腳本之間以相對位置互相 import；eNN 是實驗編號）
  predict.py, predict_hsl.py, predict_se.py, check_submission.py, make_split.py, build_l0_cache.py
  katago/                  KataGo HumanSL 掃描（scan_runner、scan_merge、scan_setup.sh、scan_pins.json…）
  experiments/eNN/、oneoff/  各階段模組（特徵／編碼／訓練／融合）
final/                     最後一步（六組機率 → 第二階段 → v13n）
colab/                     Colab 筆記本（reproduce_final、full_pipeline）、colab_utils.py、assets_manifest.json
```

`scripts/` 沒有再拆資料夾：腳本彼此用相對位置 import 與固定的 `scripts/experiments/eNN` 路徑，搬動會弄壞 import。「資料處理 → 特徵／編碼 → 訓練 → 推論 → 最終組合」的分段與每步指令寫在 [`PIPELINE.md`](PIPELINE.md)。

## 環境安裝

```bash
# 1) CPU 環境（特徵、LightGBM、sklearn、最終組合；Python 3.12）
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# 2) GPU 環境（只有 torch 的步驟需要）：系統 python3 + torch 2.12.0+cu130
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu130   # 或在 Colab 直接用其內建 torch
# 3) 測試
.venv/bin/python -m pytest -q
```

所有腳本以 repo 根目錄為工作目錄、`PYTHONPATH=src:scripts:.` 執行，例如 `PYTHONPATH=src:scripts:. .venv/bin/python scripts/predict.py ...`。
**scikit-learn 必須是 1.9.1**。

### 路徑約定

- `data/`：官方資料與掃描結果（不附）。環境變數 `DATA_DIR` 可指到別處（最終組合腳本支援）。
- `.scratch/`：工作目錄，所有中間檔與權重。環境變數 `WORK_DIR` 可指到別處（最終組合腳本支援）。
- 較上游的腳本直接使用 repo 根目錄下的 `data/` 與 `.scratch/`；若放在別的磁碟，請在 repo 根目錄建立同名符號連結（`ln -s /big/disk/data data`）。
- 遠端主機：`e79_items.py`、`e79_prep_local.py` 的 ssh/rsync 目標由環境變數 `REMOTE_HOST` 指定；Colab 控制腳本的 CLI 由 `COLAB_CLI` 指定。
- 測試用的棋譜（`tests/fixtures/golden/`）是 `tests/tools/make_synthetic_games.py` 產生的隨機合成棋譜，**不是**官方資料。

### 官方資料放置

官方資料（主辦方提供，不附）請放在 `data/` 下：

```
data/training/                 train_<rank>.csv … 訓練棋譜（1,000,000 盤；欄位 player_id, game_id, rank, color, sgf_content）
data/contest_1104/             測試集：rank_prediction_test_{public,private}.csv
```

## 外部資源（來源網址、版本、授權、放置路徑；不散布）

本專案**沒有使用任何外部棋譜或標註資料**；外部資源只有下列預訓練模型與開源套件。**這些檔案不在 repo、也不在 Release**：請自行從官方網址下載並遵守授權；下載後以 sha256 核對。
Colab 上可用 `colab/colab_utils.py` 的 `fetch_external()` 一次下載並驗證（`colab/reproduce_final.ipynb` 第 3 格，預設略過）；KataGo 另可用 `scripts/katago/setup_katago.sh`（Colab）或 `scan_setup.sh`。各網址於 2026-11-26 確認可連線，檔案大小與表中相同。

| 資源 | 官方下載網址 | 版本／檔名（大小；sha256） | 放置路徑 | 授權 | 使用方式 |
|---|---|---|---|---|---|
| KataGo 執行檔 | https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip | `katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip`（51,350,638 B；`2cb197fc9bc2050f29112abb5b83cdbad63936cb4605d6e199772aedc9806c02`） | `data/external/katago/katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip` | MIT 式（Copyright David J Wu (lightvector) and/or other authors）；`cpp/external` 內第三方元件各有授權 | **特徵抽取**：以 1 visit 對每盤棋逐手分析，取落子機率（HumanSL 掃描）。不微調、不當訓練資料 |
| KataGo 主網路 | https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz | `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz`（97,898,094 B；`9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d`） | `data/external/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` | KataGo Neural Network License（MIT 式，https://katagotraining.org/network_license/ ） | **特徵抽取**：KataGo 載入 HumanSL 時必須一併載入 |
| KataGo HumanSL 網路 | https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.bin.gz | `b18c384nbt-humanv0.bin.gz`（99,066,230 B；`637746e44f0efe00ad1245a50aa9bbf0716efe364c43965ead97bd6835d84ab5`） | `data/external/katago/b18c384nbt-humanv0.bin.gz` | 同上（Human SL 網路在該授權涵蓋範圍內）；訓練棋譜來源作者未公開說明 | **特徵抽取**：11 個段位 profile 的逐手人類落子機率（核心特徵） |
| HumanSL PyTorch 檢查點 | https://media.katagotraining.org/uploaded/networks/models_extra/b18c384nbt-humanv0.ckpt | `b18c384nbt-humanv0.ckpt`（323,752,318 B；`84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`） | `data/external/katago/b18c384nbt-humanv0.ckpt` | 同上 | **特徵抽取＋初始化**：抽主幹向量（固定 profile `rank_1d`、日本規則）；Rank 的 ft4 讀法網路以它為初始值、解凍最後 4 個區塊微調；載入前需用 KataGo 原始碼的 `migrate_defaultdict_to_dict.py` 轉換（torch 2.12） |
| Strength Estimator 權重（中研院） | https://rlg.iis.sinica.edu.tw/papers/strength-estimator/assets/models/go.tar.gz（專案頁 https://rlg.iis.sinica.edu.tw/papers/strength-estimator/ ；程式庫 https://github.com/rlglab/strength-estimator ） | `go.tar.gz`（448,813,641 B；`885bea594a62956958de526f6a2073f48ecd181535b4d9e72d2c71a339172caf`），使用其中 `se_go.pt`（`95fc295137b1fd09a0ec4d5cc79ed888f8c12d0ba6a77fea58b24efc3f418126`） | `data/external/se/download/go.tar.gz` | 程式庫 GPL-3.0；權重未見授權明示，我們只從權利人公開頁面下載使用，不再散布 | **特徵抽取**：每盤取 32 個己方局面，`score` 頭的平均與標準差；以 `torch.jit.load` 載入（`scripts/predict_se.py`、`scripts/experiments/e21/extract.py`）。只當黑盒特徵使用 |

## 自訓權重、中間檔、外部檔：三類檔案

逐檔的大小、sha256、放置位置見 **[ASSETS.md](ASSETS.md)**（`colab/assets_manifest.json` 是機器可讀版）。

| 類別 | 檔數 | 合計 | 在哪裡 |
|---|---:|---:|---|
| 自訓權重 | 5 | 167.2 MB（最大單檔 `final_s0__last.pt` 74.7 MB） | **GitHub Release**，下載網址 `RELEASE_URL`：**待補**（`colab/reproduce_final.ipynb` 第一格同名變數） |
| 官方資料衍生中間檔 | 14 | 1129.9 MB（最後一步需要的 2 檔共 4.6 MB） | **不公開**；僅提供主辦方 |
| 外部（KataGo、HumanSL、SE） | 5 | 約 1.0 GB | 不散布；官方網址見上表 |

自訓權重包含：Rank 讀法網路 `final_s0`（交件版）與 `ft4_s0`（驗證版）（HumanSL 微調）、S 認人網路 `finh_full`、C1 邏輯迴歸 `c1_fit.npz`、第二階段 `stage2_fit.pkl`（邏輯迴歸）。

## 官方資料衍生的中間檔（不公開）

最後一步與多數上游步驟的輸入，是由主辦方提供的官方棋譜算出來的：驗證題與測試題的六組機率（驗證側還含標籤）、對手側向量庫、各腿的測試端輸出。這些檔案**不放進 repo，也不放進 GitHub Release**。

- **提供方式**：可應主辦單位要求，以私人連結另行提供給主辦方（14 檔，1129.9 MB；清單、大小與 sha256 見 [ASSETS.md](ASSETS.md) 第 2 節）。
- **主辦方拿到之後**：放進一個資料夾（例如掛載的 Google Drive），開 `colab/reproduce_final.ipynb`，把第一格的 `INTERMEDIATE_DIR` 指到該資料夾，執行即可（筆記本會依資產名或放回位置找檔、複製、驗 sha256，再跑最後一步並比對 sha256）。本機也一樣：`INTERMEDIATE_DIR=<資料夾> jupyter nbconvert --execute colab/reproduce_final.ipynb`，或照「驗證」一節手動複製到 `.scratch/`。
- **沒有這些檔時，如何重新產生**（從官方資料＋我們的權重＋外部模型）。GPU 與時間為原作者實測；GPU 結果不會位元重現（見「已知限制」），所以重算的檔和主辦方那份不會位元組相同，但最後一步的算式相同：

| 中間檔 | 重新產生需要 | 腳本（參數見檔頭說明；步驟表見 PIPELINE.md） | 硬體與時間 | 備註 |
|---|---|---|---|---|
| `test_inputs.npz`（測試題六組機率） | 官方測試檔；L0／HSL／SE／主幹向量；四條腿；Release 的 `finh_full`、`final_s0`、`c1_fit.npz` | `scripts/experiments/e103/e103_make.py`、`e91/`、`e96/`、`e101/`、`e102/` | GPU 向量匯出＋CPU | **四條腿的模型檔沒有保存**，需從訓練端特徵重訓（訓練端編碼約 70 小時，估計值） |
| `val_preds.npz`（驗證題六組機率與標籤） | 官方訓練棋譜；`ft4_s0`（Release）對驗證題的輸出 | `scripts/experiments/e103/e103_val.py` 等 | GPU（學校 RTX8000 約 11 小時訓練 ft4） | 上游 `ft4_s0_val12.npz` 已不在；驗證側只能用保存版 |
| `final_s0_test12.npz`、`S_test_own/opp_b.npz`、`head_b.npz` | 官方測試棋譜；Release 的 `final_s0`、`finh_full` | `scripts/experiments/e103/e103_export.py`、`e91/e91_embed_b.py`、`e91_final_b.py` | GPU 推論 | 只做推論，不重訓 |
| `S_bank_opp_full.npz`、`bank_meta_full.npz`（對手側向量庫） | 官方訓練棋譜；Release 的 `finh_full` | `scripts/experiments/e102/e102_prep.py`、`e103/e103_data.py` | GPU 約 4,853 秒（本機 RTX 5070 Ti，測試版向量庫） | 約 1 GB |
| `test_m03.npy`、`test_m04.npy`、`set_A1T2.npy`、`set_A1P.npy` | 聚合特徵；四條腿重訓 | `scripts/experiments/e10/pipeline.py`、`scripts/oneoff/{e55,e63,e66,e79}_*.py` | CPU／GPU，小時級 | 模型檔沒有保存 |
| `aux.npz`（驗證框架輔助表） | 官方訓練棋譜與切分 | `scripts/experiments/e29/`（bundle）、`scripts/make_split.py` | CPU | |

## 在 Google Colab 上執行

| 筆記本 | 做什麼 | 硬體／耗時 |
|---|---|---|
| [`colab/reproduce_final.ipynb`](colab/reproduce_final.ipynb) | clone → 安裝鎖版本套件 → （選用）下載外部權重 → 從 `RELEASE_URL` 下載自訓權重並驗 sha256 → 從 `INTERMEDIATE_DIR` 讀入中間檔並驗 sha256 → 跑最後一步 → 印出 sha256 並與預期值比對 | CPU 即可，免費版可跑完；約 5 分鐘 |
| [`colab/full_pipeline.ipynb`](colab/full_pipeline.ipynb) | 沒有中間檔時，由官方棋譜重新產生的分段說明；每段標硬體、原作者耗時、免費版能否跑完；重段落預設不執行 | 免費版**跑不完**整條（訓練端編碼約 70 小時，估計值） |

使用方式（Colab）：File → Open notebook → GitHub，貼 repo 網址，開 `colab/reproduce_final.ipynb`，填第一格的 `REPO_URL`、`RELEASE_URL`（目前都是「待補」）與 `INTERMEDIATE_DIR`，Runtime → Run all。
官方資料不附；Rank 最後一步不需要官方原始棋譜。

筆記本以本機 `jupyter nbconvert --execute` 實跑過 `reproduce_final.ipynb`（Python 3.12、全新 venv 內以 `requirements.txt` 安裝；Release 下載改以本機 HTTP 伺服器代替；`INTERMEDIATE_DIR` 指到原專案保存的中間檔），sha256 全部相符；`full_pipeline.ipynb` 只跑了預設（不含重段落）。**尚未在真正的 Colab 上跑過**（Colab 的 Python／套件版本可能與鎖定版本不同；若 pip 提示重新啟動，照做後從頭重跑）。

## 驗證：最終檔的位元組重現

在原專案環境（保存了中間檔）驗證，輸出寫到 repo 外：

```bash
# Rank v13n（sha256 3a9525a3e7e6273727d0e5fd0a6fc7d5fc2921b17c7687b36f83e961c9d13919）
WORK_DIR=<含 e103/、e117/ 的目錄> .venv/bin/python final/make_rank_final.py --out /tmp/rank_v13n.csv --reference <上傳檔>
```

`--reference` 會逐位元組比對並印出 sha256。第三方若沒有中間檔，可閱讀程式了解最後一步的公式，並依 [`PIPELINE.md`](PIPELINE.md) 往上游重算。


## 已知限制

1. **全流程重跑成本高**：從棋譜重新編碼約 70 小時（SE 抽取 12.9 小時、主幹向量與逐局面向量 11 小時以上、S 認人網路約 42 GPU 小時（Rank 只用其中 `finh_full`，約 20.8 小時）、ft4 約 11 小時、HumanSL 掃描約 100 萬盤）。
2. **GPU 結果不會位元重現**：GPU 上 `index_add_` 順序不固定、使用 fp16；各平台 torch 版本不同。KataGo 掃描用 1 visit＋隨機對稱，重掃逐盤有雜訊。CPU 的最後組合步驟則是位元重現。
3. **sklearn 版本**：必須 1.9.1。LightGBM 4.7.0。
4. **上游中間檔不在 repo**：腳本多半以 `.scratch/…` 為輸入輸出；各階段腳本是由實驗過程中複製演進而來（每階段複製前階段程式），沒有單一入口腳本能從棋譜整條重跑。
5. 驗證側六組機率 `val_preds.npz` 只能引用保存版，其上游已不在；v13n 的離線數字來自同一批驗證題且後段決定與擬合都用這批題，偏樂觀（榜上 Public 0.7568 → Private 0.7286）。
6. 外部模型（HumanSL）的訓練棋譜來源與是否包含比賽棋局，作者未公開說明，無法排除。
7. 部分測試（約 13 個）需要官方資料，無資料時會被 skip；測試用棋譜為合成資料。
8. 四條腿（M03T2／M04T／A1T2／A1P）的模型檔沒有保存，只剩測試端輸出；`predict_hsl.py --backend school` 在公開版不可用。
9. **尚未在真正的 Colab 上跑過**：`colab/reproduce_final.ipynb` 只在本機以 `jupyter nbconvert --execute` 驗證。
10. 另一題見 `aicup2026-go-player`；兩邊的共用程式（`src/gogame/` 與部分腳本）各放一份，彼此可能略有出入（只放該題用到的腳本）。

## 生成式 AI 使用

開發過程使用 Claude Code（Anthropic）協助撰寫與審查程式、實驗與文件。
