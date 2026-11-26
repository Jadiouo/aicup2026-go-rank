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
