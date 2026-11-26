# Rank（競賽二）管線

最終上傳 **Rank v13n**（Private 第 29 名 0.7286153）。`eNN` 是實驗編號；腳本在 `scripts/experiments/eNN/` 與 `scripts/oneoff/eNN_*.py`。指令以 repo 根目錄為工作目錄、`PYTHONPATH=src:scripts:.`；參數見各腳本頂端 docstring。

## 管線（由下到上）

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

## 步驟

| 步 | 階段 | 腳本 | 輸入 → 輸出 | 硬體／耗時 |
|---|---|---|---|---|
| 0a | 資料處理：切分 | `scripts/make_split.py`、`src/gogame/split.py` | 玩家表 → `data/splits/split.parquet`、`questions_fold*.parquet` | CPU |
| 0b | 資料處理：L0 | `scripts/build_l0_cache.py`、`src/gogame/features_l0.py` | SGF → `data/cache/l0/`（dense.parquet、sparse.npz） | CPU，數十分鐘 |
| 1b | 特徵：HumanSL 掃描 | `scripts/katago/{scan_runner,scan_merge,scan_setup.sh,scan_pins.json}`；後半盤 `scripts/experiments/e34/scan_runner_w2.py`、`e35/`、`e36/build_w120_full.py` | SGF → `data/scan*` → W120 表 `X120_full.npy` | GPU；訓練端 100 萬盤，Colab L4 約 2.35 盤／秒；後半盤 E35 255,464 盤 28.9 小時 |
| 1c | 特徵：SE 強度 | `scripts/experiments/e21/{extract,pack_upload,merge_check}.py`、`se_encode.py` | SGF（每盤 32 個己方局面）→ SE score 平均 | GPU 12.9 小時（685,231 盤）；測試端約 1 小時 |
| 1d | 特徵：主幹向量 | `scripts/oneoff/{e55_prep,e55_trunk,e55_pca,e63_prep,e63_trunk,e63_pca,e79_items,e79_gpu,e79_prep_local,e79_tok}.py` | 主幹啟動值 → `G_full.npy`、後半盤、`tok_full.npy` | GPU；E79 約 11 小時（兩 GPU 並行） |
| 1e | 訓練：S 認人網路 | `scripts/experiments/e89/*`、`e90/{make_data_h,train_e90h,driver_e90h,embed}.py`、`e96/{e96_roster,train_e96,e96_embed}.py` | 約 100 萬盤 → `finh_full/final_model.pt`、`vec_*.npz` | GPU；Colab L4 20.8 小時 |
| 1f | 訓練：ft4 讀法網路 | `scripts/experiments/e98/{e98_data,e98_train,e98_export}.py`、`e103/{e103_data,e103_train,e103_export,e103_final_data}.py`、`e89/encode_k.py` | 全部非 H 盤每個己方手（約 7,900 萬局面）→ `final_s0/last.pt`、`final_s0_test12.npz` | GPU（學校 RTX8000）約 11 小時；K 線編碼約 10 小時量級 |
| 2a | 訓練＋推論：M03T2／M04T | `scripts/experiments/e10/pipeline.py`、`scripts/oneoff/{e55_dse,e63_dse,e66_final,e57_dse}.py`、`src/gogame/{aggregate,models,coral}.py` | 聚合特徵 → `test_m03.npy`、`test_m04.npy` | CPU |
| 2b | 訓練＋推論：A1T2／A1P | `scripts/experiments/e29/setmodel.py`、`scripts/oneoff/{e55_a1,e63_a1,e63_predict,e79_a1p,e79_predict}.py` | 集合模型 15＋15 個 → `set_A1T2.npy`、`set_A1P.npy` | GPU／CPU |
| 3 | 融合 | `scripts/oneoff/{e66_fuse,e79_fuse}.py` | 四條腿 → v9 機率 | CPU |
| 4–5 | 對手票、C1／C2／C3 | `scripts/experiments/e91/{e91_prep,e91_embed_b,e91_final,e91_final_b}.py`、`e96/e96_score.py`、`e101/e101_score.py`、`e102/{e102_prep,e102_score}.py`、`e103/{e103_core,e103_make}.py` | → `e117/final/test_inputs.npz`（800×10 六組機率） | GPU 向量匯出＋CPU |
| 6 | 第二階段（驗證框架） | `scripts/experiments/e117/{e117_core,e117_make}.py`、`e124/{e124_core,e124_make,e124_run}.py` | `e103/val_preds.npz`（4,568 題六組機率）→ 第二階段擬合 | CPU |
| 7 | **最終組合（v13／v12／v13n）** | **`final/make_rank_final.py`**（核心數學取自 `e117_core.py`、`e124_core.py`：`final/stage2_core.py`） | `val_preds.npz`＋`test_inputs.npz`＋`stage2_fit.pkl` → `rank_v13n.csv` | CPU，約 1 分鐘 |
| 8 | 格式檢查 | `scripts/check_submission.py`、`src/gogame/submission.py` | CSV → OK／FAILED | CPU |

**逐元件的訓練指令、推論指令、輸入、輸出見本頁最後一節「元件手冊」。**

權重與中間檔的大小、sha256 與放置位置見[`ASSETS.md`](ASSETS.md)；Colab 版最後一步見 `colab/reproduce_final.ipynb`。

## 最後一步：指令、輸入、輸出

```bash
.venv/bin/python final/make_rank_final.py --work-dir <WORK_DIR> --out out/rank_v13n.csv [--reference 上傳檔.csv]
```

| 輸入（`<WORK_DIR>` 下） | 內容 |
|---|---|
| `e103/val_preds.npz` | 驗證側（fold 4，4,568 題）：`P_ALL、P_REF、FF、P_C1、P_C2、P_C3`（n×10）、`nq`、`yq_not_for_scoring_here`（標籤，只用來擬合第二階段） |
| `e117/final/test_inputs.npz` | 測試側 800 題：`qid、nq、qsplit`、同樣六組機率 |
| `e117/final/stage2_fit.pkl` | E117 第二階段邏輯迴歸（v12；內含選定的 LR/C/ρ） |

輸出：800 列 `question_id,pred_rank`，並印 sha256、與 v13（差 21 題）、v12（差 25 題）的差異數。

## 注意

- 驗證側六組機率 `val_preds.npz` 只能引用保存版，無法從上游重算（其產生腳本需要的 `ft4_s0_val12.npz` 已不在）。
- 驗證框架（`e117_core.py`、`e124_core.py`）依賴原專案的資料切分與評分模組；公開版的最後一步改用 `final/stage2_core.py`（同一份數學，移除了框架 import）。
- v13n 的離線數字來自同一批驗證題（fold 4）且後段決定與擬合都用這批題，偏樂觀；榜上 Public 0.7568 → Private 0.7286。

## 元件手冊：每個元件的訓練、推論、輸入、輸出

上面的「步驟」表是由下到上的流程；這一節是逐元件的指令。慣例：

- 指令以 repo 根目錄為工作目錄；CPU 腳本用 `PYTHONPATH=src:scripts:. .venv/bin/python`，標「GPU」的用系統 `python3`＋torch（`requirements-gpu.txt`）。完整參數看各腳本頂端 docstring；`.scratch/`、`data/` 的位置見 README「路徑約定」。
- 「報告名稱」是報告正文用的描述性名稱；「內部代號」是程式與紀錄裡的編號。
- 原作者用本機 GPU 排隊工具與 `systemd-run` 記憶體上限包住重工作，這兩者不是必要的。

### 資料與特徵（多半無訓練）

**1. 手工特徵（L0）** — 無訓練。`.venv/bin/python scripts/build_l0_cache.py [--jobs 14] [--chunk 20000]`；輸入官方棋譜，輸出 `data/cache/l0/{dense.parquet,sparse.npz}`。切分：`scripts/make_split.py`（以 player_id 分組）。

**2. 訓練資料包（E29 bundle）** — 無訓練。`.venv/bin/python -m scripts.experiments.e29.prep_bundle` → `.scratch/e29/bundle/{X.npy,aux.npz,manifest.json}`（非保留玩家 712,945 盤的逐盤特徵、折角色；後面的主幹向量、集合模型、各腳本都讀它）。

**3. 棋力剖面特徵（HumanSL 落子統計，前 60／120 手，104 欄）** — 外部 KataGo HumanSL，無訓練。
- 抽取：`bash scripts/katago/scan_setup.sh`（以 sha256 釘選下載 KataGo 與網路）→ `scripts/katago/scan_runner.py`（前 60 手）／`scripts/experiments/e34/scan_runner_w2.py`（第 61–120 手），參數見兩者 docstring。
- 彙整成逐盤 104 欄：訓練端 `python -m scripts.experiments.e36.build_w120_full old|new|train`（→ `.scratch/e36/bundle/X120_full.npy`；先前階段 `e34.build_x120`）；測試端 `python -m scripts.experiments.e35.build_x120 old|new|train|test`（`test` → `test_X120.npy`）。
- 輸入：SGF。輸出：逐手 log p、名次、熵 → 逐盤特徵。

**4. 強度估計器特徵（SE，外部預訓練模型，只推論）**
- 訓練端（GPU，約 12.9 小時）：`.venv/bin/python scripts/experiments/e21/pack_upload.py`（打包棋局）→ GPU `scripts/experiments/e21/extract.py`（每盤均勻抽 32 個己方局面取 score 頭平均；輸入編碼 `se_encode.py`）→ `merge_check.py` → `make_se_full.py`（→ `.scratch/e21/se_full.parquet`，HD≥2 為 NaN）。
- 測試端：`python3 scripts/predict_se.py extract ...`（需 `GOGAME_TORCH_PATH` 或自行安裝 torch；權重見 ASSETS.md 外部清單）→ `se_test_games.parquet`（`uid,se_mean,se_std,se_n`）。

**5. 中間層向量特徵（KataGo 主幹向量；前半盤 tk48、後半盤 lk48、24 個局面 token）** — 外部預訓練網路的內部啟動值，無訓練（PCA 在每折內擬合）。
- 前半盤：`.venv/bin/python scripts/oneoff/e55_prep.py` → GPU `python3 scripts/oneoff/e55_trunk.py` → `.venv/bin/python scripts/oneoff/e55_pca.py`（→ `G_full.npy`、每折 PCA-48）。
- 後半盤：`e63_prep.py` → GPU `e63_trunk.py` → `e63_pca.py`（→ `L_full.npy`、每折 PCA-48）。
- 逐局面 token：`e79_items.py` → `e79_prep_local.py` → GPU `e79_gpu.py`（PCA-64 → fp16）→ `e79_tok.py`（→ `tok_full.npy`）。
- 輸入：SGF、KataGo `b18c384nbt-humanv0.ckpt`。編碼器 `scripts/experiments/e13/encode.py`。

### 神經網路

**6. 認人網路（S；`finh_full`，10 區塊×128 通道）** — Rank 用它「看對手是誰」。
- 資料：`PYTHONPATH=src:. .venv/bin/python scripts/experiments/e90/make_data.py union|test|meta`（`u_games.npz`、`rosters.npz`、`t_games.npz`、`export_spec*`）→ `scripts/experiments/e90/make_data_h.py build` 與 `python scripts/experiments/e90/make_data_h.py concat [--dir DIR]`（加入保留玩家的盤，得 `uh_games.npz`、`rosters_h.npz`）。自寫棋盤引擎 `scripts/experiments/e89/fastboard.py`（numba）。
- 訓練（GPU，可續跑）：`python scripts/experiments/e90/driver_e90h.py --roster finh_full --root DIR --accum 2 --workers 8 --export auto`（20 輪、種子 0；DIR 內放 `uh_games.npz rosters_h.npz t_games.npz export_spec_h.json export_spec_h.npz`）。驗證版 `finh_v4`（排除驗證折玩家）：`.venv/bin/python scripts/experiments/e96/e96_roster.py` 建名單，`scripts/experiments/e96/train_e96.py` 訓練。
- 推論（把一盤棋嵌入成 128 維、對庫內玩家算機率）：`/usr/bin/python3 scripts/experiments/e91/e91_embed_b.py head|embed GAMES.npz OUT.npz --net-dir D --head F`（測試棋局用 `e91_final.py prep-test`、`embed-test`；驗證版 `e96/e96_embed.py`）。
- 輸入：約 100 萬盤的己方局面 18 個 19×19 平面。輸出：`finh_full/final_model.pt`（權重）、`.scratch/e91/b/{head_b.npz,S_test_own_b.npz,S_test_opp_b.npz}`。

**7. 微調網路（ft4；驗證版 `ft4_s0`、交件版 `final_s0`）** — KataGo humanv0 解凍最後 4 個區塊＋集合網路＋序數頭，以段位標籤微調。
- 訓練資料：K 線局面編碼 `python scripts/experiments/e89/encode_k.py plan|run ...`；驗證版 bundle `scripts/experiments/e98/e98_data.py extract`；交件版 `scripts/experiments/e103/e103_final_data.py select|merge|verify`（加入驗證折玩家；抽局面仍用 `e98_data.py extract`）與 `e103_data.py build|check|prep-test`（測試端 bundle）。
- 訓練（GPU，可續跑，12,000 步、種子 0）：驗證版 `python3 scripts/experiments/e98/e98_train.py --arm ft4 --seed 0 ...`；交件版 `python3 scripts/experiments/e103/e103_train.py --arm ft4 --seed 0 --run-name final_s0 ...`（e103 的複本，只改訓練玩家集合）。
- 推論（每題 10 類機率 `q_probs`）：驗證題 `python3 scripts/experiments/e98/e98_export.py --arm ft4 --seed 0 --steps 12000 --val bundle_val12 --out ft4_s0_val12.npz`；測試題 `python3 scripts/experiments/e103/e103_export.py --arm ft4 --seed 0 --steps 12000 --val ... --out final_s0_test12.npz`。
- 輸入：`b18c384nbt-humanv0.ckpt`（外部）、局面 bundle。輸出：`runs/{ft4_s0,final_s0}/last.pt`（權重）、`ft4_s0_val12.npz`、`final_s0_test12.npz`。

### 四個基礎模型與融合

**8. 梯度提升樹模型（LightGBM 多分類，M03T2）與序數迴歸模型（CORAL，M04T）** — 訓練與推論同一支腳本，CPU。
- 訓練：`.venv/bin/python scripts/oneoff/e66_final.py train --out DIR [--threads 4]` → `DIR/rank_hsl_lgbm_se.pkl`（兩個模型）、`train_info.json`（M03T2 固定 63 輪、3,199 欄；M04T 762 欄，`src/gogame/coral.py`）。
- 推論：`.venv/bin/python scripts/oneoff/e66_final.py predict --model-dir DIR --out-prefix PFX` → `PFX_m03.npy`、`PFX_m04.npy`（800×10）。
- 輸入：元件 1–5 的聚合特徵（`src/gogame/aggregate.py`）。輸出：兩個模型檔與測試機率。

**9. 集合神經網路（DeepSets＋CORAL，A1T2；15 個模型＝5 折×3 種子）**
- 訓練（GPU，一次一折可選）：`python3 scripts/oneoff/e63_a1.py [--folds 0 1 2 3 4] [--seeds 0 1 2]`（網路定義與取樣 `scripts/experiments/e29/setmodel.py`）。
- 推論：`python3 scripts/oneoff/e63_predict.py infer`（CPU 亦可；測試特徵先用 `python -m scripts.experiments.e29.predict_test features` 或 `e35.build_x120 test`）→ `set_A1T2.npy`（[5,3,800,10]）。
- 輸入：X235（HumanSL 104＋L0 31＋SE 2 欄與缺值旗標＋is_white＋tk48＋lk48）。輸出：模型檔（每折每種子一個）、`set_A1T2.npy`。

**10. 注意力集合神經網路（A1P；15 個模型）**
- 訓練（GPU）：`python3 scripts/oneoff/e79_a1p.py [--folds 0 1 2 3 4]`（每折先 4 格網格選 p_in、tdrop，再訓種子 0／1／2）。
- 推論：`python3 scripts/oneoff/e79_predict.py infer --out set_A1P.npy`（`valcheck` 先驗證能重現驗證折機率）。
- 輸入：X139＋24 局面 token。輸出：模型檔、`set_A1P.npy`。

**11. 基礎模型融合（v9 融合）** — `.venv/bin/python scripts/oneoff/e79_fuse.py --m03 NPY --m04 NPY --set-a1t2 NPY --set-a1p NPY --out CSV --report J --ref v8 CSV`；(w,v,u)=(.9,.5,.75)（`e66_fuse.py` 是前一代）。輸入元件 8–10 的四個機率檔，輸出 800 列 CSV 與機率。

### 對手線索、第二階段與最終版

**12. 微調網路混入（C1）** — `PYTHONPATH=src:scripts:. .venv/bin/python scripts/experiments/e103/e103_make.py fit-c1`（在 4,568 題驗證題的 `q_probs` 上擬合多類別邏輯迴歸，C=1、z=.2；輸出 `c1_fit.{json,npz}`）；套用在 `e103_make.py make ... --c1 final_s0_test12.npz`。

**13. 對手線索（庫內票 V）** — 無額外訓練（用元件 6 的認人網路與分類頭）。`scripts/experiments/e91/e91_prep.py`（對手側棋局檔）、`e91_embed_b.py`（嵌入）、`e91_final.py make`／`e91_final_b.py make`（把票併進基礎機率，α=.1、λ=2）、`e96/e96_score.py`（驗證）。

**14. 近鄰線索（C2）** — 無訓練。建庫：`/usr/bin/python3 scripts/experiments/e103/e103_bank_prep.py`（全部訓練棋局的對手側棋局檔與列表）→ 用元件 6 的網路嵌入（`e91_embed_b.py embed`）→ `S_bank_opp_full.npz`；查詢：`e103_make.py make --c2 .scratch/e103/bank`（前 50 近鄰、β=40、w=.5）。驗證版建庫 `scripts/experiments/e102/e102_prep.py`、`e102_score.py`。

**15. 定先盤換算（C3）** — 無訓練，兩個轉移核寫在程式裡：`scripts/experiments/e101/e101_score.py`，套用 `e103_make.py make --c3`。

**16. 六組機率（測試端與驗證側）**
- 測試端（v11b）：`e103_make.py make --name v11_retrain --c1 .scratch/e103/exports/final_s0_test12.npz --c2 .scratch/e103/bank --c3 --allow-other-ckpt`；整理成六組並存檔：`.venv/bin/python scripts/experiments/e117/e117_make.py inputs` → `.scratch/e117/final/test_inputs.npz`。
- 驗證側：`E103_FORMAL=1 .venv/bin/python scripts/experiments/e103/e103_val.py formal` → `.scratch/e103/val_preds.npz`（4,568 題六組機率；需要驗證版 `ft4_s0_val12.npz` 與驗證版對手線索，本 repo 只能引用保存版，見下方「注意」）。

**17. 第二階段（邏輯迴歸版 E117；平移不變版 E124，70 參數）**
- 訓練：邏輯迴歸版 `.venv/bin/python scripts/experiments/e117/e117_make.py fit`（→ `stage2_fit.pkl`）；平移不變版在 `e124_make.py final` 內擬合（核心 `e124_core.py`、`e117_core.py`；驗證 `e124_run.py`）。
- 推論：`.venv/bin/python scripts/experiments/e117/e117_make.py make`（v12）；`.venv/bin/python scripts/experiments/e124/e124_make.py final`（v13；輸出 `.scratch/e124/final/rank_v12_shift.csv`）。
- 輸入：`val_preds.npz`、`test_inputs.npz`。輸出：`stage2_fit.pkl`、核與偏置 json、兩個 CSV。公開版的同一份數學在 `final/stage2_core.py`。

**18. 最終版（v13n：N=5 的 200 題用邏輯迴歸版答案，其餘用平移不變版）** — 見上節：`.venv/bin/python final/make_rank_final.py --work-dir <WORK_DIR> --out out/rank_v13n.csv`；格式檢查 `scripts/check_submission.py`。

### 本 repo 找不到程式的部分

- v13n 的 N=5 拼接：原專案當時是臨時拼接，沒有存檔；`final/make_rank_final.py` 是事後補的等價實作（已驗證與上傳檔逐列相同）。
- 驗證側六組機率 `val_preds.npz` 無法從上游重算（`ft4_s0_val12.npz` 等已刪）；其產生程式（`e103_val.py formal`、`e98_*`、`e96_*`、`e102_*`）都在。
- 局面抽取（`e98_data.py extract`、`e103_data.py build`）原在學校主機執行；平台排程與監看腳本不公開，平台無關的執行本體都在上表。
