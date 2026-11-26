"""E98 匯出：用訓練完的最後一步檢查點（runs/<arm>_s<seed>/last.pt）對驗證 bundle（fold 4 驗證題的 41,589 盤）算輸出，給本機 e98_score.py 計分。
輸出 npz（鍵）：
  qid (4568,)、q_probs (4568,10) 以題為單位（該題的 N 盤當一袋，head.score → CORAL 類別機率）、q_s (4568,) 袋的純量分數、q_game_mean_probs (4568,10) 該題各盤「單盤袋」類別機率的平均、
  rid (41589,) 升冪（＝E29 bundle 列號，同 E92／E95 匯出檔的 rid）、game_s1 (41589,) 單盤袋純量分數、game_probs1 (41589,10)、game_phi (41589,256) float16（φ 後的盤向量）、
  thresholds (9,)、step、arm、seed、n_pos_per_game_max。
可續做（每 --chunk 盤存一個小檔於 <out>.chunks/）；不覆寫既有輸出。檢查點的 step 必須等於 --steps（只有 --allow-partial 的測試例外）。
  python e98_export.py --arm ft4 --seed 0 --steps N --val bundle_val12 --out exports/ft4_s0_val12.npz
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import e98_train as T


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["frozen", "ft4"], default="ft4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-q", default=str(T.W / "sel" / "val_questions.npz"))
    ap.add_argument("--run", default=None, help="run dir (default <W>/runs/<arm>_s<seed>)")
    ap.add_argument("--ckpt", default=str(T.W / "ckpt.ckpt"))
    ap.add_argument("--stats", default=str(T.W / "stats.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-partial", action="store_true", help="TEST ONLY: accept a checkpoint at step != --steps")
    ap.add_argument("--synth-val", action="store_true", help="TEST ONLY: synthetic questions on a partial validation bundle")
    ap.add_argument("--limit-games", type=int, default=0, help="TEST ONLY")
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists(), f"{out} exists - refusing to overwrite"
    out.parent.mkdir(parents=True, exist_ok=True)
    if T.free_gb(out.parent) < 1.0:
        raise SystemExit("disk free < 1GB: stop")
    dev = a.device
    run = Path(a.run) if a.run else T.W / "runs" / f"{a.arm}_s{a.seed}"
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    if not a.allow_partial:
        assert ck["step"] == a.steps, f"checkpoint is at step {ck['step']}, expected the last step {a.steps} - training not finished"
    assert ck["args"]["arm"] == a.arm and ck["args"]["seed"] == a.seed
    z = np.load(a.stats)
    head = T.Head(z["mu"], z["sd"]).to(dev)
    head.load_state_dict(ck["head"])
    trunk = T.Trunk(a.arm, dev, Path(a.ckpt))
    assert len(trunk.params) == len(ck["trunk_ft"])
    for p, v in zip(trunk.params, ck["trunk_ft"]):
        p.data.copy_(v.to(dev))
    head.eval()
    V = T.ValData(a.val, None if a.synth_val else a.val_q, limit_games=a.limit_games, synth=a.synth_val)
    print(f"export arm {a.arm} seed {a.seed} step {ck['step']} | val games {V.G} questions {V.nq} positions {V.nrows} | device {dev}", flush=True)
    cdir = Path(str(out) + ".chunks")
    cdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gv_parts = []
    for c0 in range(0, V.G, a.chunk):
        cf = cdir / f"c{c0:08d}.npy"
        if not cf.exists():
            gv = T.embed_games(V, trunk, head, dev, np.arange(c0, min(V.G, c0 + a.chunk))).cpu().numpy()
            np.save(str(cf) + ".tmp.npy", gv)
            Path(str(cf) + ".tmp.npy").rename(cf)
            print(f"chunk {c0}/{V.G} {time.time() - t0:.0f}s", flush=True)
        gv_parts.append(np.load(cf))
    gv = torch.from_numpy(np.concatenate(gv_parts)).to(dev)
    assert gv.shape == (V.G, 256) and torch.isfinite(gv).all()
    with torch.no_grad():
        phi = head.phi(gv)
        s1 = head.rho(phi)[:, 0]
        p1 = head.probs(s1)
    qs, qp = T.question_outputs(head, gv, V.q_gidx, dev)
    p1n, s1n = p1.cpu().numpy(), s1.cpu().numpy()
    gq = np.where(V.q_gidx >= 0, V.q_gidx, 0)
    mk = (V.q_gidx >= 0).astype(np.float64)
    qgm = (p1n[gq] * mk[..., None]).sum(1) / mk.sum(1, keepdims=True)
    o = V.order                                            # bundle 序 -> rid 升冪
    assert np.isfinite(qp).all() and np.allclose(qp.sum(1), 1, atol=1e-4) and np.allclose(qgm.sum(1), 1, atol=1e-4)
    np.savez(out, qid=V.q["qid"], q_probs=qp.astype(np.float64), q_s=qs.astype(np.float64), q_game_mean_probs=qgm,
             rid=V.rid[o].astype(np.int64), game_s1=s1n[o], game_probs1=p1n[o], game_phi=phi.cpu().numpy()[o].astype(np.float16),
             thresholds=head.thresholds().detach().cpu().numpy(), step=int(ck["step"]), arm=a.arm, seed=a.seed, n_pos_per_game_max=int(np.max(V.end - V.start)),
             ckpt_sha16=sha16(run / "last.pt"), positions=int(V.nrows))
    shutil.rmtree(cdir)
    Path(str(out) + ".json").write_text(json.dumps(dict(arm=a.arm, seed=a.seed, step=int(ck["step"]), val=str(a.val), n_games=int(V.G), n_questions=int(V.nq), positions=int(V.nrows),
                                                        seconds=round(time.time() - t0), q_top1_class_counts=np.bincount(qp.argmax(1), minlength=10).tolist(), ckpt_sha16=sha16(run / "last.pt")), indent=1))
    print("written", out, "q_probs", qp.shape, "q argmax counts", np.bincount(qp.argmax(1), minlength=10).tolist(), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
