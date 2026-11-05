#!/usr/bin/env python3
"""Per-channel full-recording predictions for the website (2026-11-05).

For every Fig 3c method that has full-recording (every chunk) test predictions, run the unchanged
main protocol (fig3_main_protocol_prep.main_protocol: inverse-variance channel pooling -> unweighted
per-probe Kabsch, LH-folded scoring) on decoder seed 1 and save, per channel:
  c_pk, c_ch, pooled (AP, DV, ML um), kabsch (um), true (um)
to build/chan/<key>.npz.  Also checks the seed-1 per-probe Kabsch RMSE against the stored Fig 3 full-test
inputs (fig3_{main,fusion}_inputs_full118.npz, seed index 0).

Usage: export_chan.py <key> [<key> ...]   (keys below)
Sources are the 118-probe full-test run (E) and, when present, the 119th probe run (E119).
"""
import glob, json, os, sys
import numpy as np

SD = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/Lfp2vec_benchmarks/scripts"
E = "/scratch/sd5963/ray_results/Neurips_2026/pathb_newmodels/fig3_fulltest118_2026-11-02"
sys.path.insert(0, SD)
sys.path.insert(0, f"{E}/source")
import fig3_main_protocol_prep as PP      # noqa: E402
import fig3_fulltest_prep as FP           # noqa: E402
K = PP.K
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chan")
PRED = f"{E}/preds"

FEAT = FP.FEAT          # key -> file stem
SIGMA_FEAT = FP.SIGMA_FEAT


def set_rowkey():
    fz = [np.load(f) for f in sorted(glob.glob(f"{E}/fineid/fid_*.npz"))]
    sh = np.concatenate([z["shard"] for z in fz]); off = np.concatenate([z["offset"] for z in fz])
    meta_all = np.concatenate([z["meta_str"] for z in fz]).astype(str)
    keys = sh.astype(np.int64) * 5000 + off
    o = np.argsort(keys)
    FP.ROWKEY = (keys[o], meta_all[o])


def load(key):
    """(mu, sig, true, meta, mode) for decoder seed 1 of one method."""
    if key in ("ours_lin", "ours_tf", "fus_lin", "fus_tf"):
        mu, sig, tr, meta = PP.load_ours(f"{PRED}/{key}_s1/predictions")
        return mu, sig, tr, meta, "sigma"
    if key == "manip":
        tr, meta = FP.load_fusion_rows()
        mu = np.load(f"{PRED}/manip_xyz.npy").astype(np.float64)
        assert len(mu) == len(meta) and np.isfinite(mu).all()
        return mu, np.ones_like(mu), tr, meta, "mean"
    if key in ("vit", "anynet"):
        return FP.load_source((key, f"{E}/baselines_vit_anynet/full118_{key}.npz"))
    if key in FEAT:
        return FP.load_source((key, f"{E}/baselines_feat/full118_{FEAT[key]}.npz"))
    raise KeyError(key)


FUSION_KEYS = ("fus_lin", "fus_tf", "manip")


def stored_seed1(key, long_to_short):
    """Seed-1 per-probe Kabsch RMSE from the stored Fig 3 full-test inputs: {8-hex probe key: rmse}.

    The fusion file indexes probes by the long id (K.parse_pid_channel); long_to_short maps it to the
    8-hex key used by PP.pk_ch / main_protocol.
    """
    if key in FUSION_KEYS:
        z = np.load(f"{E}/fig3_inputs/fig3_fusion_inputs_full118.npz")
        probes = [long_to_short[p] for p in z["probes"].astype(str)]
    else:
        z = np.load(f"{E}/fig3_inputs/fig3_main_inputs_full118.npz")
        probes = z["probes"].astype(str).tolist()
    e2, n = z[f"{key}_kabsch_e2"][0], z[f"{key}_kabsch_n"][0]
    return {p: float(np.sqrt(a / b)) for p, a, b in zip(probes, e2, n) if b > 0}


def main():
    set_rowkey()
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for key in sys.argv[1:]:
        mu, sig, tr, meta, mode = load(key)
        pk, ch = PP.pk_ch(meta)
        r = PP.main_protocol(mu, sig, tr, pk, ch, mode)
        pooled, kab, true = r["pooled"], r["kabsch"], r["true"]
        # seed-1 per-probe Kabsch RMSE (LH folded), and the same for the stored inputs
        e2 = ((K.fold_lr(kab) - K.fold_lr(true)) ** 2).sum(1)
        mine = {}
        for p in np.unique(r["c_pk"]):
            m = r["c_pk"] == p
            mine[str(p)] = float(np.sqrt(e2[m].mean()))
        long_to_short = {}
        if key in FUSION_KEYS:
            _, first = np.unique(pk, return_index=True)
            for i in first:
                long_to_short[str(K.parse_pid_channel(meta[i])[0])] = str(pk[i])
        st = stored_seed1(key, long_to_short)
        common = sorted(set(mine) & set(st))
        diff = max(abs(mine[p] - st[p]) for p in common) if common else None
        report[key] = dict(rows=int(len(mu)), channels=int(len(pooled)), probes=len(mine),
                           matched_probes=len(common), max_abs_diff_um=diff,
                           mean_rmse_um=float(np.mean(list(mine.values()))))
        print(key, json.dumps(report[key]), flush=True)
        np.savez_compressed(f"{OUT}/{key}.npz", c_pk=r["c_pk"].astype("U24"), c_ch=r["c_ch"].astype(np.int32),
                            pooled=pooled.astype(np.float32), kabsch=np.asarray(kab, np.float32),
                            true=true.astype(np.float32), mode=mode, n_rows=len(mu))
        del mu, sig, tr, meta, r
    with open(f"{OUT}/report_{'_'.join(sys.argv[1:])[:60]}.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
