# Rank（競賽二）管線

這份文件給想重現或讀懂整條管線的人。最終上傳的版本（內部代號 v13n）在 Private 拿到第 29 名、0.7286153。

指令預設以 repo 根目錄為工作目錄，並設定 `PYTHONPATH=src:scripts:.`。腳本頂端的 docstring 有完整參數。

## 代號對照

程式與紀錄用實驗編號命名。腳本放在 `scripts/experiments/eNN/` 和 `scripts/oneoff/eNN_*.py`。下表是這份文件會用到的編號。

| 代號 | 是什麼 |
|---|---|
| `e10` | 四條腿裡 LightGBM 與 CORAL 的訓練流程 |
| `e13` | KataGo 主幹向量的編碼器 |
| `e21` | Strength Estimator（SE，中研院的棋力估計模型）特徵抽取 |
| `e29` | 訓練資料包，以及集合神經網路的模型定義 |
| `e34`、`e35`、`e36` | HumanSL 後半盤（第 61 到 120 手）掃描與 W120 表 |
| `e55`、`e63`、`e79` | 主幹向量三代：前半盤、後半盤、24 個局面 token |
| `e89`、`e90`、`e96` | S 認人網路的資料、訓練，以及驗證版 |
| `e91` | 對手線索：用認人網路讀對手側 |
| `e98`、`e103` | ft4 微調網路的驗證版與交件版，`e103` 也負責六組機率的產生 |
| `e101` | 定先盤換算（C3） |
| `e102` | 近鄰線索（C2）的驗證版建庫 |
| `e117` | 第二階段（邏輯迴歸版，v12） |
| `e124` | 第二階段（平移不變版，v13） |

其他名稱：

- 「四條腿」是 M03T2（LightGBM）、M04T（CORAL 序數迴歸）、A1T2（DeepSets）和 A1P（DeepSets 加逐局面注意力池化）四個基礎模型。
- ft4 是 HumanSL 解凍最後 4 個區塊的微調網路。驗證版叫 `ft4_s0`，交件版叫 `final_s0`。
- C1、C2、C3 是三個可各自開關的附加項：C1 是微調網路的混入，C2 是近鄰線索，C3 是定先盤換算。
- W60、W120 是 HumanSL 掃描的視窗，分別取前 60 手和前 120 手。
- v9 是四條腿的融合，v12 是邏輯迴歸版第二階段，v13 是平移不變版第二階段，v13n 是最終版。

## 環境與路徑

CPU 環境用來做特徵、LightGBM、sklearn 和最終組合，Python 3.12。GPU 環境只有 torch 的步驟需要。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# GPU 環境：系統 python3 加 torch 2.12.0+cu130，在 Colab 可以直接用內建的 torch
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pytest -q
```

所有腳本以 repo 根目錄為工作目錄，用 `PYTHONPATH=src:scripts:.` 執行，例如：

```bash
PYTHONPATH=src:scripts:. .venv/bin/python scripts/predict.py ...
```

scikit-learn 必須是 1.9.1，LightGBM 用 4.7.0。

路徑約定：

- `data/` 放官方資料與掃描結果，不附。環境變數 `DATA_DIR` 可以指到別處，最終組合腳本支援。
- `.scratch/` 是工作目錄，放所有中間檔與權重。環境變數 `WORK_DIR` 可以指到別處，最終組合腳本支援。
- 較上游的腳本直接用 repo 根目錄下的 `data/` 和 `.scratch/`。放在別的磁碟時，在 repo 根目錄建同名符號連結，例如 `ln -s /big/disk/data data`。
- `e79_items.py` 和 `e79_prep_local.py` 的 ssh／rsync 目標由環境變數 `REMOTE_HOST` 指定。Colab 控制腳本的 CLI 由 `COLAB_CLI` 指定。
- `tests/fixtures/golden/` 的棋譜是 `tests/tools/make_synthetic_games.py` 產生的隨機合成棋譜，不是官方資料。部分測試（約 13 個）需要官方資料，沒有資料時會被 skip。

官方資料由主辦方提供，不附。請放在 `data/` 下：

```
data/training/       train_<rank>.csv …，訓練棋譜 1,000,000 盤（欄位 player_id, game_id, rank, color, sgf_content）
data/contest_1104/   測試集：rank_prediction_test_{public,private}.csv
```

## 管線總覽與步驟表

下圖由上往下讀，一共七層。最上面是逐盤特徵，最下面是最終答案。

```
層 1  逐盤／逐局面特徵（棋譜 → 向量，不看標籤）
      L0 手工特徵 │ KataGo HumanSL 掃描（W60/W120，104 欄）│ SE 強度分數 │ KataGo 主幹向量（前半 tk48／後半 lk48、24 局面 token）
      │ S 認人網路 finh_full（用來「看對手是誰」）│ 讀法網路 ft4（HumanSL 解凍最後 4 區塊，用段位標籤微調）
層 2  四條腿（每題輸出 10 類機率）：M03T2 LightGBM、M04T CORAL 序數迴歸、A1T2 DeepSets、A1P DeepSets＋逐局面注意力池化
層 3  v9 融合 F_E79 = nrm((1−w)·nrm(v·M03T2+(1−v)·M04T) + w·nrm((1−u)·A1T2+u·A1P))，(w,v,u)=(.9,.5,.75)
層 4  C1：F_E79 與 ft4 讀法的邏輯迴歸混合 → FF
層 5  對手票（C2 近鄰票、C3 定先盤站規換算）→ 六組機率 P_ALL、P_REF、FF、P_C1、P_C2、P_C3
層 6  第二階段（E124）：六組取 log → 沿段位軸平移不變的核分類器（70 參數）→ 與 P_ALL 幾何平均 ρ=.5 → EV 解碼 = v13
層 7  N=5 的 200 題改用 v12（E117 第二階段邏輯迴歸）的答案 → v13n（最終）
```

步驟表依序列出每一步的腳本、輸入輸出和硬體。

| 步 | 階段 | 腳本 | 輸入 → 輸出 | 硬體／耗時 |
|---|---|---|---|---|
| 0a | 資料處理：切分 | `scripts/make_split.py`、`src/gogame/split.py` | 玩家表 → `data/splits/split.parquet`、`questions_fold*.parquet` | CPU |
| 0b | 資料處理：L0 | `scripts/build_l0_cache.py`、`src/gogame/features_l0.py` | SGF → `data/cache/l0/`（dense.parquet、sparse.npz） | CPU，數十分鐘 |
| 1b | 特徵：HumanSL 掃描 | `scripts/katago/{scan_runner,scan_merge,scan_setup.sh,scan_pins.json}`；後半盤 `scripts/experiments/e34/scan_runner_w2.py`、`e35/`、`e36/build_w120_full.py` | SGF → `data/scan*` → W120 表 `X120_full.npy` | GPU；訓練端 100 萬盤，Colab L4 約 2.35 盤／秒；後半盤 E35 255,464 盤 28.9 小時 |
| 1c | 特徵：SE 強度 | `scripts/experiments/e21/{extract,pack_upload,merge_check}.py`、`se_encode.py` | SGF（每盤 32 個己方局面）→ SE score 平均 | GPU 12.9 小時（685,231 盤）；測試端約 1 小時 |
| 1d | 特徵：主幹向量 | `scripts/oneoff/{e55_prep,e55_trunk,e55_pca,e63_prep,e63_trunk,e63_pca,e79_items,e79_gpu,e79_prep_local,e79_tok}.py` | 主幹啟動值 → `G_full.npy`、後半盤、`tok_full.npy` | GPU；E79 約 11 小時（兩 GPU 並行） |
| 1e | 訓練：S 認人網路 | `scripts/experiments/e89/*`、`e90/{make_data_h,train_e90h,driver_e90h,embed}.py`、`e96/{e96_roster,train_e96,e96_embed}.py` | 約 100 萬盤 → `finh_full/final_model.pt`、`vec_*.npz` | GPU；Colab L4 20.8 小時 |
| 1f | 訓練：ft4 讀法網路 | `scripts/experiments/e98/{e98_data,e98_train,e98_export}.py`、`e103/{e103_data,e103_train,e103_export,e103_final_data}.py`、`e89/encode_k.py` | 全部非 H 盤每個己方手（約 7,900 萬局面）→ `final_s0/last.pt`、`final_s0_test12.npz` | GPU（學校 RTX8000）約 11 小時；K 線編碼約 10 小時量級 |
| 2a | 訓練＋推論：M03T2、M04T | `scripts/experiments/e10/pipeline.py`、`scripts/oneoff/{e55_dse,e63_dse,e66_final,e57_dse}.py`、`src/gogame/{aggregate,models,coral}.py` | 聚合特徵 → `test_m03.npy`、`test_m04.npy` | CPU |
| 2b | 訓練＋推論：A1T2、A1P | `scripts/experiments/e29/setmodel.py`、`scripts/oneoff/{e55_a1,e63_a1,e63_predict,e79_a1p,e79_predict}.py` | 集合模型 15＋15 個 → `set_A1T2.npy`、`set_A1P.npy` | GPU／CPU |
| 3 | 融合 | `scripts/oneoff/{e66_fuse,e79_fuse}.py` | 四條腿 → v9 機率 | CPU |
| 4–5 | 對手票、C1／C2／C3 | `scripts/experiments/e91/{e91_prep,e91_embed_b,e91_final,e91_final_b}.py`、`e96/e96_score.py`、`e101/e101_score.py`、`e102/{e102_prep,e102_score}.py`、`e103/{e103_core,e103_make}.py` | → `e117/final/test_inputs.npz`（800×10 六組機率） | GPU 向量匯出，加 CPU |
| 6 | 第二階段（驗證框架） | `scripts/experiments/e117/{e117_core,e117_make}.py`、`e124/{e124_core,e124_make,e124_run}.py` | `e103/val_preds.npz`（4,568 題六組機率）→ 第二階段擬合 | CPU |
| 7 | 最終組合（v13、v12、v13n） | `final/make_rank_final.py`（核心數學取自 `e117_core.py`、`e124_core.py`：`final/stage2_core.py`） | `val_preds.npz`、`test_inputs.npz`、`stage2_fit.pkl` → `rank_v13n.csv` | CPU，約 1 分鐘 |
| 8 | 格式檢查 | `scripts/check_submission.py`、`src/gogame/submission.py` | CSV → OK／FAILED | CPU |

逐元件的訓練與推論指令、輸入、輸出，在本文件最後的「元件手冊」。權重與中間檔的大小、sha256 和放置位置見 [ASSETS.md](ASSETS.md)。Colab 版的最後一步見 `colab/reproduce_final.ipynb`。

## 最後一步

這一步把六組機率經第二階段組成最終提交檔，只用 CPU，約 1 分鐘。

```bash
.venv/bin/python final/make_rank_final.py --work-dir <WORK_DIR> --out out/rank_v13n.csv [--reference 上傳檔.csv]
```

`--reference` 會逐位元組比對上傳檔並印出 sha256。沒有中間檔的人，可以讀程式了解最後一步的公式，再依本文件往上游重算。

最後一步需要的輸入，都在 `<WORK_DIR>` 下：

| 輸入 | 內容 |
|---|---|
| `e103/val_preds.npz` | 驗證側（fold 4，4,568 題）：`P_ALL、P_REF、FF、P_C1、P_C2、P_C3`（n×10）、`nq`、`yq_not_for_scoring_here`（標籤，只用來擬合第二階段） |
| `e117/final/test_inputs.npz` | 測試側 800 題：`qid、nq、qsplit`，以及同樣的六組機率 |
| `e117/final/stage2_fit.pkl` | E117 第二階段邏輯迴歸（v12，內含選定的 LR、C、ρ） |

輸出是 800 列的 `question_id,pred_rank`，並印出 sha256，以及與 v13（差 21 題）、v12（差 25 題）的差異數。

## 沒有中間檔時怎麼重新產生

最後一步和多數上游步驟的輸入，是由官方棋譜算出來的：驗證題與測試題的六組機率（驗證側還含標籤）、對手側向量庫、各條腿的測試端輸出。這些檔案不放在 repo 或 Release，只能向主辦單位索取。

沒有這些檔時，可以從官方資料、我們的權重和外部模型重新產生。下表的硬體和時間是原作者實測。GPU 的結果不會位元重現，所以重算出來的檔和主辦方那份不會位元組相同，但最後一步的算式一樣。

| 中間檔 | 重新產生需要 | 腳本（參數見檔頭說明） | 硬體與時間 | 備註 |
|---|---|---|---|---|
| `test_inputs.npz`（測試題六組機率） | 官方測試檔；L0、HSL、SE、主幹向量；四條腿；Release 的 `finh_full`、`final_s0`、`c1_fit.npz` | `scripts/experiments/e103/e103_make.py`、`e91/`、`e96/`、`e101/`、`e102/` | GPU 向量匯出，加 CPU | 四條腿的模型檔沒有保存，需從訓練端特徵重訓（GPU 約 71 小時以上，另加微調網路輸入編碼在學校 CPU 約 70 小時，分項相加） |
| `val_preds.npz`（驗證題六組機率與標籤） | 官方訓練棋譜；`ft4_s0`（Release）對驗證題的輸出 | `scripts/experiments/e103/e103_val.py` 等 | GPU（學校 RTX8000 約 11 小時訓練 ft4） | 上游 `ft4_s0_val12.npz` 已不在，驗證側只能用保存版 |
| `final_s0_test12.npz`、`S_test_own/opp_b.npz`、`head_b.npz` | 官方測試棋譜；Release 的 `final_s0`、`finh_full` | `scripts/experiments/e103/e103_export.py`、`e91/e91_embed_b.py`、`e91_final_b.py` | GPU 推論 | 只做推論，不重訓 |
| `S_bank_opp_full.npz`、`bank_meta_full.npz`（對手側向量庫） | 官方訓練棋譜；Release 的 `finh_full` | `scripts/experiments/e102/e102_prep.py`、`e103/e103_data.py` | GPU 約 4,853 秒（本機 RTX 5070 Ti，測試版向量庫） | 約 1 GB |
| `test_m03.npy`、`test_m04.npy`、`set_A1T2.npy`、`set_A1P.npy` | 聚合特徵；四條腿重訓 | `scripts/experiments/e10/pipeline.py`、`scripts/oneoff/{e55,e63,e66,e79}_*.py` | CPU／GPU，小時級 | 模型檔沒有保存 |
| `aux.npz`（驗證框架輔助表） | 官方訓練棋譜與切分 | `scripts/experiments/e29/`（bundle）、`scripts/make_split.py` | CPU | |

## 外部模型的用途

本專案沒有使用任何外部棋譜或標註資料，外部資源只有下列預訓練模型和開源套件。這些檔案不在 repo，也不在 Release，請自行從官方網址下載並遵守授權，下載後用 sha256 核對。網址、大小、sha256 和放置路徑在 [ASSETS.md](ASSETS.md) 第 3 節。各網址於 2026-11-26 確認可連線，檔案大小與表中相同。

在 Colab 上，`colab/colab_utils.py` 的 `fetch_external()` 可以一次下載並驗證，也就是 `colab/reproduce_final.ipynb` 的第 3 格，預設略過。KataGo 另可用 `scripts/katago/setup_katago.sh`（Colab）或 `scan_setup.sh`。

各資源怎麼用：

- KataGo 執行檔：以 1 visit 對每盤棋逐手分析，取落子機率（HumanSL 掃描）。不微調，也不當訓練資料。
- KataGo 主網路：KataGo 載入 HumanSL 時必須一併載入。
- KataGo HumanSL 網路：提供 11 個段位 profile 的逐手人類落子機率，是核心特徵。授權和上面一致，但訓練棋譜的來源作者沒有公開說明。
- HumanSL PyTorch 檢查點：抽主幹向量（固定 profile `rank_1d`、日本規則）。Rank 的 ft4 讀法網路以它為初始值，解凍最後 4 個區塊微調。載入前要用 KataGo 原始碼的 `migrate_defaultdict_to_dict.py` 轉換（torch 2.12）。
- Strength Estimator 權重（中研院）：每盤取 32 個己方局面，取 `score` 頭的平均與標準差，用 `torch.jit.load` 載入（`scripts/predict_se.py`、`scripts/experiments/e21/extract.py`）。只當黑盒特徵使用。權重沒有明確授權，我們只從公開頁面下載使用。

## 注意事項

- 驗證側六組機率 `val_preds.npz` 只能引用保存版，無法從上游重算，因為產生它所需的 `ft4_s0_val12.npz` 已不在。
- 驗證框架（`e117_core.py`、`e124_core.py`）依賴原專案的資料切分與評分模組。公開版的最後一步改用 `final/stage2_core.py`，數學相同，只是移除了框架的 import。
- v13n 的離線數字來自同一批驗證題（fold 4），後段的決定與擬合也都用這批題，所以偏樂觀。榜上 Public 0.7568 到 Private 0.7286，也能看出這個落差。

## 元件手冊

上面的步驟表是由下到上的流程，這一節則是逐元件的指令。

- 指令以 repo 根目錄為工作目錄。CPU 腳本用 `PYTHONPATH=src:scripts:. .venv/bin/python`。標「GPU」的用系統 `python3` 加 torch（`requirements-gpu.txt`）。
- 完整參數看各腳本頂端的 docstring。`.scratch/` 和 `data/` 的位置見上面「環境與路徑」。
- 「報告名稱」是報告正文用的描述性名稱，「內部代號」是程式與紀錄裡的編號。
- 原作者用本機 GPU 排隊工具和 `systemd-run` 的記憶體上限包住重工作，這兩者不是必要的。

### 資料與特徵（多半無訓練）

#### 1. 手工特徵（L0）

無訓練。

```bash
.venv/bin/python scripts/build_l0_cache.py [--jobs 14] [--chunk 20000]
```

輸入是官方棋譜，輸出是 `data/cache/l0/{dense.parquet,sparse.npz}`。切分用 `scripts/make_split.py`，以 player_id 分組。

#### 2. 訓練資料包（E29 bundle）

無訓練。

```bash
.venv/bin/python -m scripts.experiments.e29.prep_bundle
```

輸出 `.scratch/e29/bundle/{X.npy,aux.npz,manifest.json}`，內容是非保留玩家 712,945 盤的逐盤特徵與折角色。後面的主幹向量、集合模型和各腳本都讀它。

#### 3. 棋力剖面特徵（HumanSL 落子統計，前 60、120 手，104 欄）

用外部的 KataGo HumanSL，無訓練。

- 抽取：`bash scripts/katago/scan_setup.sh` 會以 sha256 釘選並下載 KataGo 與網路。接著跑 `scripts/katago/scan_runner.py`（前 60 手）和 `scripts/experiments/e34/scan_runner_w2.py`（第 61 到 120 手），參數見兩者的 docstring。
- 彙整成逐盤 104 欄：訓練端用 `python -m scripts.experiments.e36.build_w120_full old|new|train`，輸出 `.scratch/e36/bundle/X120_full.npy`，先前階段是 `e34.build_x120`。測試端用 `python -m scripts.experiments.e35.build_x120 old|new|train|test`，其中 `test` 輸出 `test_X120.npy`。
- 輸入：SGF。輸出：逐手 log p、名次、熵，再彙總成逐盤特徵。

#### 4. 強度估計器特徵（SE，外部預訓練模型，只推論）

- 訓練端（GPU，約 12.9 小時）：先 `.venv/bin/python scripts/experiments/e21/pack_upload.py` 打包棋局，再用 GPU 跑 `scripts/experiments/e21/extract.py`，每盤均勻抽 32 個己方局面，取 score 頭的平均，輸入編碼在 `se_encode.py`。然後 `merge_check.py`，最後 `make_se_full.py`，輸出 `.scratch/e21/se_full.parquet`，其中 HD≥2 的盤為 NaN。
- 測試端：`python3 scripts/predict_se.py extract ...`，需要設定 `GOGAME_TORCH_PATH` 或自行安裝 torch，權重見 ASSETS.md 的外部清單。輸出 `se_test_games.parquet`（`uid,se_mean,se_std,se_n`）。

#### 5. 中間層向量特徵（KataGo 主幹向量）

包含前半盤 tk48、後半盤 lk48，以及 24 個局面 token。這是外部預訓練網路的內部啟動值，無訓練，PCA 在每折內擬合。

- 前半盤：`.venv/bin/python scripts/oneoff/e55_prep.py`，然後 GPU `python3 scripts/oneoff/e55_trunk.py`，再 `.venv/bin/python scripts/oneoff/e55_pca.py`，輸出 `G_full.npy` 和每折 PCA-48。
- 後半盤：`e63_prep.py`，然後 GPU `e63_trunk.py`，再 `e63_pca.py`，輸出 `L_full.npy` 和每折 PCA-48。
- 逐局面 token：`e79_items.py`，`e79_prep_local.py`，GPU `e79_gpu.py`（PCA-64 轉 fp16），最後 `e79_tok.py`，輸出 `tok_full.npy`。
- 輸入：SGF 和 KataGo 的 `b18c384nbt-humanv0.ckpt`。編碼器在 `scripts/experiments/e13/encode.py`。

### 神經網路

#### 6. 認人網路（S，`finh_full`，10 區塊×128 通道）

Rank 用它「看對手是誰」。

- 資料：先 `PYTHONPATH=src:. .venv/bin/python scripts/experiments/e90/make_data.py union|test|meta`，得到 `u_games.npz`、`rosters.npz`、`t_games.npz`、`export_spec*`。再用 `scripts/experiments/e90/make_data_h.py build` 和 `python scripts/experiments/e90/make_data_h.py concat [--dir DIR]` 加入保留玩家的盤，得到 `uh_games.npz`、`rosters_h.npz`。自寫的棋盤引擎在 `scripts/experiments/e89/fastboard.py`（numba）。
- 訓練（GPU，可續跑）：`python scripts/experiments/e90/driver_e90h.py --roster finh_full --root DIR --accum 2 --workers 8 --export auto`，20 輪、種子 0。DIR 內要放 `uh_games.npz rosters_h.npz t_games.npz export_spec_h.json export_spec_h.npz`。驗證版 `finh_v4` 排除了驗證折的玩家，先用 `.venv/bin/python scripts/experiments/e96/e96_roster.py` 建名單，再用 `scripts/experiments/e96/train_e96.py` 訓練。
- 推論（把一盤棋嵌入成 128 維，再對庫內玩家算機率）：`/usr/bin/python3 scripts/experiments/e91/e91_embed_b.py head|embed GAMES.npz OUT.npz --net-dir D --head F`。測試棋局用 `e91_final.py prep-test` 和 `embed-test`，驗證版用 `e96/e96_embed.py`。
- 輸入：約 100 萬盤的己方局面，共 18 個 19×19 平面。輸出：`finh_full/final_model.pt`（權重），以及 `.scratch/e91/b/{head_b.npz,S_test_own_b.npz,S_test_opp_b.npz}`。

#### 7. 微調網路（ft4）

驗證版是 `ft4_s0`，交件版是 `final_s0`。做法是把 KataGo humanv0 解凍最後 4 個區塊，接上集合網路和序數頭，用段位標籤微調。

- 訓練資料：K 線局面編碼用 `python scripts/experiments/e89/encode_k.py plan|run ...`。驗證版 bundle 用 `scripts/experiments/e98/e98_data.py extract`。交件版用 `scripts/experiments/e103/e103_final_data.py select|merge|verify`，它會加入驗證折的玩家，抽局面仍用 `e98_data.py extract`。測試端 bundle 用 `e103_data.py build|check|prep-test`。
- 訓練（GPU，可續跑，12,000 步、種子 0）：驗證版 `python3 scripts/experiments/e98/e98_train.py --arm ft4 --seed 0 ...`，交件版 `python3 scripts/experiments/e103/e103_train.py --arm ft4 --seed 0 --run-name final_s0 ...`。後者是 e103 的複本，只改了訓練玩家集合。
- 推論（每題 10 類機率 `q_probs`）：驗證題用 `python3 scripts/experiments/e98/e98_export.py --arm ft4 --seed 0 --steps 12000 --val bundle_val12 --out ft4_s0_val12.npz`，測試題用 `python3 scripts/experiments/e103/e103_export.py --arm ft4 --seed 0 --steps 12000 --val ... --out final_s0_test12.npz`。
- 輸入：外部的 `b18c384nbt-humanv0.ckpt` 和局面 bundle。輸出：`runs/{ft4_s0,final_s0}/last.pt`（權重）、`ft4_s0_val12.npz`、`final_s0_test12.npz`。

### 四個基礎模型與融合

#### 8. 梯度提升樹模型（LightGBM 多分類，M03T2）與序數迴歸模型（CORAL，M04T）

訓練與推論在同一支腳本，用 CPU。

- 訓練：`.venv/bin/python scripts/oneoff/e66_final.py train --out DIR [--threads 4]`，輸出 `DIR/rank_hsl_lgbm_se.pkl`（兩個模型）和 `train_info.json`。M03T2 固定 63 輪、3,199 欄，M04T 是 762 欄（`src/gogame/coral.py`）。
- 推論：`.venv/bin/python scripts/oneoff/e66_final.py predict --model-dir DIR --out-prefix PFX`，輸出 `PFX_m03.npy` 和 `PFX_m04.npy`（800×10）。
- 輸入：元件 1 到 5 的聚合特徵（`src/gogame/aggregate.py`）。輸出：兩個模型檔和測試機率。

#### 9. 集合神經網路（DeepSets 加 CORAL，A1T2）

共 15 個模型，5 折乘 3 個種子。

- 訓練（GPU，可一次只跑一折）：`python3 scripts/oneoff/e63_a1.py [--folds 0 1 2 3 4] [--seeds 0 1 2]`。網路定義與取樣在 `scripts/experiments/e29/setmodel.py`。
- 推論：`python3 scripts/oneoff/e63_predict.py infer`，CPU 也能跑。測試特徵要先用 `python -m scripts.experiments.e29.predict_test features` 或 `e35.build_x120 test` 產生。輸出 `set_A1T2.npy`（[5,3,800,10]）。
- 輸入：X235，內容是 HumanSL 104 欄、L0 31 欄、SE 2 欄加缺值旗標、is_white、tk48、lk48。輸出：模型檔（每折每個種子一個）和 `set_A1T2.npy`。

#### 10. 注意力集合神經網路（A1P，15 個模型）

- 訓練（GPU）：`python3 scripts/oneoff/e79_a1p.py [--folds 0 1 2 3 4]`。每折先用 4 格網格選 p_in 和 tdrop，再訓練種子 0、1、2。
- 推論：`python3 scripts/oneoff/e79_predict.py infer --out set_A1P.npy`。`valcheck` 可以先驗證能不能重現驗證折的機率。
- 輸入：X139 加 24 個局面 token。輸出：模型檔和 `set_A1P.npy`。

#### 11. 基礎模型融合（v9 融合）

```bash
.venv/bin/python scripts/oneoff/e79_fuse.py --m03 NPY --m04 NPY --set-a1t2 NPY --set-a1p NPY --out CSV --report J --ref v8 CSV
```

(w,v,u)=(.9,.5,.75)。`e66_fuse.py` 是前一代。輸入是元件 8 到 10 的四個機率檔，輸出是 800 列 CSV 和機率。

### 對手線索、第二階段與最終版

#### 12. 微調網路混入（C1）

```bash
PYTHONPATH=src:scripts:. .venv/bin/python scripts/experiments/e103/e103_make.py fit-c1
```

這會在 4,568 題驗證題的 `q_probs` 上擬合多類別邏輯迴歸，C=1、z=.2，輸出 `c1_fit.{json,npz}`。套用時用 `e103_make.py make ... --c1 final_s0_test12.npz`。

#### 13. 對手線索（庫內票 V）

不需要額外訓練，用元件 6 的認人網路與分類頭。`scripts/experiments/e91/e91_prep.py` 準備對手側棋局檔，`e91_embed_b.py` 做嵌入，`e91_final.py make` 和 `e91_final_b.py make` 把票併進基礎機率（α=.1、λ=2），`e96/e96_score.py` 做驗證。

#### 14. 近鄰線索（C2）

無訓練。

- 建庫：`/usr/bin/python3 scripts/experiments/e103/e103_bank_prep.py` 產生全部訓練棋局的對手側棋局檔與列表，再用元件 6 的網路嵌入（`e91_embed_b.py embed`），得到 `S_bank_opp_full.npz`。
- 查詢：`e103_make.py make --c2 .scratch/e103/bank`，取前 50 近鄰，β=40、w=.5。
- 驗證版的建庫用 `scripts/experiments/e102/e102_prep.py` 和 `e102_score.py`。

#### 15. 定先盤換算（C3）

無訓練。兩個轉移核直接寫在程式裡：`scripts/experiments/e101/e101_score.py`。套用時用 `e103_make.py make --c3`。

#### 16. 六組機率（測試端與驗證側）

- 測試端（v11b）：先 `e103_make.py make --name v11_retrain --c1 .scratch/e103/exports/final_s0_test12.npz --c2 .scratch/e103/bank --c3 --allow-other-ckpt`，再用 `.venv/bin/python scripts/experiments/e117/e117_make.py inputs` 整理成六組並存成 `.scratch/e117/final/test_inputs.npz`。
- 驗證側：`E103_FORMAL=1 .venv/bin/python scripts/experiments/e103/e103_val.py formal`，輸出 `.scratch/e103/val_preds.npz`（4,568 題六組機率）。它需要驗證版 `ft4_s0_val12.npz` 和驗證版對手線索，本 repo 只能引用保存版，見上面的「注意事項」。

#### 17. 第二階段（邏輯迴歸版 E117，平移不變版 E124，70 參數）

- 訓練：邏輯迴歸版用 `.venv/bin/python scripts/experiments/e117/e117_make.py fit`，輸出 `stage2_fit.pkl`。平移不變版在 `e124_make.py final` 內擬合，核心在 `e124_core.py` 和 `e117_core.py`，驗證用 `e124_run.py`。
- 推論：`.venv/bin/python scripts/experiments/e117/e117_make.py make`（v12），以及 `.venv/bin/python scripts/experiments/e124/e124_make.py final`（v13，輸出 `.scratch/e124/final/rank_v12_shift.csv`）。
- 輸入：`val_preds.npz` 和 `test_inputs.npz`。輸出：`stage2_fit.pkl`、核與偏置的 json、兩個 CSV。公開版的同一份數學在 `final/stage2_core.py`。

#### 18. 最終版（v13n）

N=5 的 200 題用邏輯迴歸版的答案，其餘用平移不變版。指令見上面的「最後一步」：

```bash
.venv/bin/python final/make_rank_final.py --work-dir <WORK_DIR> --out out/rank_v13n.csv
```

格式檢查用 `scripts/check_submission.py`。

### 本 repo 找不到程式的部分

- v13n 的 N=5 拼接：原專案當時是臨時拼接，沒有存檔。`final/make_rank_final.py` 是事後補的等價實作，已驗證與上傳檔逐列相同。
- 驗證側六組機率 `val_preds.npz` 無法從上游重算，`ft4_s0_val12.npz` 等已刪。但產生它的程式（`e103_val.py formal`、`e98_*`、`e96_*`、`e102_*`）都在。
- 局面抽取（`e98_data.py extract`、`e103_data.py build`）原本在學校主機執行。平台的排程與監看腳本不公開，平台無關的執行本體都在上面的表裡。

## 已知限制

1. 全流程重跑成本高。從棋譜重新編碼與重訓，GPU 約 71 小時以上。已知分項有強度估計器 12.9、中間層向量約 11、認人網路兩版約 29、微調網路兩版約 18.5 小時。另外微調網路輸入編碼在學校 CPU 以 4 個行程跑，約 70 小時。沒有整條量過，棋力剖面掃描 100 萬盤另計。
2. GPU 結果不會位元重現。GPU 上 `index_add_` 的順序不固定，又用 fp16，各平台的 torch 版本也不同。KataGo 掃描用 1 visit 加隨機對稱，重掃逐盤有雜訊。CPU 的最後組合步驟則是位元重現。
3. sklearn 必須是 1.9.1，LightGBM 是 4.7.0。
4. 上游中間檔不在 repo。腳本多半以 `.scratch/…` 為輸入輸出。各階段腳本是實驗過程中複製演進而來，每個階段都複製了前一階段的程式，沒有單一入口腳本能從棋譜整條重跑。
5. 驗證側六組機率 `val_preds.npz` 只能引用保存版，其上游已不在。v13n 的離線數字來自同一批驗證題，後段決定與擬合都用這批題，偏樂觀（榜上 Public 0.7568 到 Private 0.7286）。
6. 外部模型（HumanSL）的訓練棋譜來源，以及是否包含比賽棋局，作者沒有公開說明，無法排除。
7. 部分測試（約 13 個）需要官方資料，沒有資料時會被 skip，測試用棋譜是合成資料。
8. 四條腿（M03T2、M04T、A1T2、A1P）的模型檔沒有保存，只剩測試端輸出。`predict_hsl.py --backend school` 在公開版不可用。
9. `colab/reproduce_final.ipynb` 只在本機以 `jupyter nbconvert --execute` 驗證過，還沒有在真正的 Colab 上跑過。
10. 另一題見 `aicup2026-go-player`。兩邊的共用程式（`src/gogame/` 與部分腳本）各放一份，彼此可能略有出入，只放該題用到的腳本。
