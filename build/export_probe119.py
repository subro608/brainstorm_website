#!/usr/bin/env python3
"""Probe b73ef9cf (the 119th test probe) for the website, LFP-only methods only.

Inputs: /scratch/sd5963/ray_results/Neurips_2026/pathb_newmodels/fig3_fulltest_probe119_2026-11-05 (every chunk of
the probe, same checkpoints / recipe as the 118-probe full-test run). Same main protocol as export_chan.py.
Writes build/probe119/chan_<key>.npz (seed 1), rmse.json (per-probe kabsch / kabsch_s1 / pooled / raw,
mean over decoder seeds as in Fig 3c) and meta.json (subject, lab, long id, fine_id per channel).
"""
import json, os, sys
import numpy as np

SD = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/Lfp2vec_benchmarks/scripts"
E = "/scratch/sd5963/ray_results/Neurips_2026/pathb_newmodels/fig3_fulltest118_2026-11-02"
E119 = "/scratch/sd5963/ray_results/Neurips_2026/pathb_newmodels/fig3_fulltest_probe119_2026-11-05"
sys.path.insert(0, SD); sys.path.insert(0, f"{E}/source")
import fig3_main_protocol_prep as PP      # noqa: E402
import fig3_fulltest_prep as FP           # noqa: E402
K = PP.K
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe119")
SEEDS = (1, 2, 3, 4, 5, 6, 7, 42)
PROBE = "b73ef9cf"


def scores(r):
    """per-probe RMSE for the single probe: raw, pooled, kabsch (LH folded)."""
    out = {"raw": float(np.sqrt(r["raw_e2"].mean()))}
    for st in ("pooled", "kabsch"):
        e2 = ((K.fold_lr(r[st]) - K.fold_lr(r["true"])) ** 2).sum(1)
        out[st] = float(np.sqrt(e2.mean()))
    return out


def save(key, r):
    np.savez_compressed(f"{OUT}/chan_{key}.npz", c_pk=r["c_pk"].astype("U24"), c_ch=r["c_ch"].astype(np.int32),
                        pooled=r["pooled"].astype(np.float32), kabsch=np.asarray(r["kabsch"], np.float32),
                        true=r["true"].astype(np.float32))


def main():
    os.makedirs(OUT, exist_ok=True)
    fz = np.load(f"{E119}/fineid/fid_000.npz")
    keys = fz["shard"].astype(np.int64) * 5000 + fz["offset"]
    o = np.argsort(keys)
    FP.ROWKEY = (keys[o], fz["meta_str"].astype(str)[o])
    rmse = {}
    for key in ("ours_lin", "ours_tf"):
        per = []
        for i, s in enumerate(SEEDS):
            mu, sig, tr, meta = PP.load_ours(f"{E119}/preds/{key}_s{s}/predictions")
            pk, ch = PP.pk_ch(meta)
            assert set(pk.tolist()) == {PROBE} and len(mu) == 68_800
            r = PP.main_protocol(mu, sig, tr, pk, ch, "sigma")
            per.append(scores(r))
            if i == 0:
                save(key, r)
        rmse[key] = {st: float(np.mean([p[st] for p in per])) for st in ("raw", "pooled", "kabsch")}
        rmse[key]["kabsch_s1"] = per[0]["kabsch"]
        rmse[key]["kabsch_per_seed"] = [p["kabsch"] for p in per]
    srcs = {k: f"{E119}/baselines_vit_anynet/probe119_{k}.npz" for k in ("vit", "anynet")}
    srcs.update({k: f"{E119}/baselines_feat/probe119_{stem}.npz" for k, stem in FP.FEAT.items()})
    for key, src in srcs.items():
        if key in ("vit", "anynet"):
            z = np.load(src); meta = FP.meta_for(z["shard"], z["offset"])
            mu, sig, tr, mode = z["preds"].astype(np.float64), z["stds"].astype(np.float64), z["true"].astype(np.float64), "sigma"
        else:
            mu, sig, tr, meta, mode = FP.load_source((key, src))
        pk, ch = PP.pk_ch(meta)
        assert set(pk.tolist()) == {PROBE} and len(mu) == 68_800, key
        r = PP.main_protocol(mu, sig, tr, pk, ch, mode)
        sc = scores(r)
        rmse[key] = dict(sc, kabsch_s1=sc["kabsch"])
        save(key, r)
        print(key, json.dumps(sc), flush=True)
    for k in ("ours_lin", "ours_tf"):
        print(k, json.dumps({a: b for a, b in rmse[k].items() if a != "kabsch_per_seed"}), flush=True)
    # metadata: lab of subject SWC_042 from the 118-probe channel table (its other probe is in that set)
    zm = np.load(f"{E}/fig3_inputs/fig3_main_inputs_full118.npz")
    labs = set(zm["eh_lab"][zm["eh_subject"] == "SWC_042"].astype(str))
    assert len(labs) == 1, labs
    pk, ch = PP.pk_ch(fz["meta_str"].astype(str))
    fine = {}
    for c, f in zip(ch.tolist(), fz["fine_id"].tolist()):
        fine.setdefault(str(c), int(f))
    meta = {"probes": {PROBE: {"subject": "SWC_042", "lab": labs.pop(), "fine_id": fine,
                               "long_id": "f9860a11-24d3-452e-ab95-39e199f20a93_probe00"}},
            "source": E119}
    json.dump({k: {PROBE: {a: b for a, b in v.items() if a != "kabsch_per_seed"}} for k, v in rmse.items()},
              open(f"{OUT}/rmse.json", "w"), indent=1)
    json.dump(rmse, open(f"{OUT}/rmse_detail.json", "w"), indent=1)
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
