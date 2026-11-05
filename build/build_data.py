#!/usr/bin/env python3
"""Assemble the website data from build/chan/*.npz (+ optional probe-119 extras) -> site/data/.

Outputs
  site/data/probes.json        probes, methods, per-probe RMSE tables, regions, provenance
  site/data/gt.bin             int16 [N,3] true CCF um (AP, DV, ML; native hemisphere)
                               + uint16 [N] region index + uint16 [N] channel number
  site/data/methods/<key>.bin  int16 [N,3] pooled + int16 [N,3] Kabsch, display coordinates
                               (LH-folded prediction mirrored into the true hemisphere; -32768 = missing)
All RMSE values are left-hemisphere-folded 3D RMSE in um, per probe (Fig 3c statistic: per seed, then mean
over decoder seeds). Seed 1 is drawn for multi-seed methods.
"""
import glob, json, os, re
import numpy as np

W = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/website_lfpformer_2026-11-05"
E = "/scratch/sd5963/ray_results/Neurips_2026/pathb_newmodels/fig3_fulltest118_2026-11-02"
TREE = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/data/allen_ccf/structure_tree.json"
CHAN = f"{W}/build/chan"
EXTRA = f"{W}/build/probe119"          # optional: chan_<key>.npz + rmse.json for probe b73ef9cf
OUT = f"{W}/site/data"
MID = 5739.0          # ALLEN_MIDLINE_UM (recompute_refined_3d_metrics), the fold used by every Fig 3 score
MISSING = -32768

METHODS = [  # key, label, group, colour, n seeds   (Fig 3c rows with full-recording predictions)
    ("fus_lin", "LFPformer (linear) + manip.", "LFP + micromanipulator", "#C77DA3", 8),
    ("fus_tf", "LFPformer (TF) + manip.", "LFP + micromanipulator", "#882255", 8),
    ("manip", "Micromanipulator alone", "LFP + micromanipulator", "#4D4D4D", 1),
    ("ours_tf", "LFPformer (TF)", "LFPformer, LFP only", "#882255", 8),
    ("ours_lin", "LFPformer (linear)", "LFPformer, LFP only", "#C77DA3", 8),
    ("anynet", "AnyNet", "Deep baselines", "#33A02C", 1),
    ("vit", "ViT", "Deep baselines", "#D62728", 1),
    ("histgbm", "HistGBM", "Feature baselines", "#1F78B4", 1),
    ("knn_psd", "kNN (PSD)", "Feature baselines", "#6B6B6B", 1),
    ("knn_band", "kNN (band)", "Feature baselines", "#6B6B6B", 1),
    ("rf_band", "RF (band)", "Feature baselines", "#6B6B6B", 1),
    ("ridge_psd", "Ridge (PSD)", "Feature baselines", "#6B6B6B", 1),
    ("pcaridge_psd", "PCA-Ridge (PSD)", "Feature baselines", "#6B6B6B", 1),
    ("svm_band", "SVM (band)", "Feature baselines", "#6B6B6B", 1),
    ("ridge_band", "Ridge (band)", "Feature baselines", "#6B6B6B", 1),
]
FUSION = {"fus_lin", "fus_tf", "manip"}
MAJOR = [(315, "Isocortex"), (698, "OLF"), (1089, "HPF"), (703, "CTXsp"), (477, "STR"), (803, "PAL"),
         (549, "TH"), (1097, "HY"), (313, "MB"), (771, "P"), (354, "MY"), (512, "CB"),
         (1009, "fiber tracts"), (73, "VS")]
META_RE = re.compile(r"^([0-9a-f-]{36})_\d+_(probe\d+)_([0-9a-f]+)-ch(\d+)_")


def fold(x):
    y = np.asarray(x, np.float64).copy()
    y[:, 2] = MID - np.abs(y[:, 2] - MID)
    return y


def display(pred, true, pk):
    """Unfolded prediction, mirrored as a WHOLE probe into the true track's hemisphere.

    Mirroring per channel (the 2026-11-05 first version) bent the rigid Kabsch track into a Z wherever the true track
    crosses the midline (8 test probes, e.g. b33e4298): channels on the far side jumped across. Scores are unchanged
    (still per-channel LH-folded, as in the paper); only the drawing is per probe.
    """
    d = np.asarray(pred, np.float64).copy()
    for p in np.unique(pk):
        m = pk == p
        if (true[m, 2].mean() > MID) != (d[m, 2].mean() > MID):
            d[m, 2] = 2 * MID - d[m, 2]
    return d


def per_probe_seed_mean(e2, n):
    """(seeds, probes) e2 sums / counts -> per-probe RMSE averaged over seeds (Fig 3c statistic)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sqrt(e2 / n)
    return r.mean(0), r[0]


def main():
    os.makedirs(f"{OUT}/methods", exist_ok=True)
    tree = {int(n["id"]): n for n in json.load(open(TREE))}

    # ---- probe metadata (118 full-test probes) from the stored Fig 3 inputs + fineid meta strings
    zm = np.load(f"{E}/fig3_inputs/fig3_main_inputs_full118.npz")
    zf = np.load(f"{E}/fig3_inputs/fig3_fusion_inputs_full118.npz")
    subj = dict(zip(zm["eh_probe"].astype(str), zm["eh_subject"].astype(str)))
    lab = dict(zip(zm["eh_probe"].astype(str), zm["eh_lab"].astype(str)))
    fid = {(p, int(c)): int(f) for p, c, f in zip(zm["chan_ours_lin_c_pk"].astype(str),
                                                    zm["chan_ours_lin_c_ch"], zm["eh_fid"])}
    long_of = {}
    for f in sorted(glob.glob(f"{E}/fineid/fid_*.npz")):
        meta = np.load(f)["meta_str"].astype(str)
        for s in np.unique(np.char.partition(meta, "-ch")[:, 0]):
            m = META_RE.match(s + "-ch0_")
            long_of.setdefault(m.group(3), f"{m.group(1)}_{m.group(2)}")

    # ---- per-channel arrays
    chans = {}
    for key, *_ in METHODS:
        z = np.load(f"{CHAN}/{key}.npz")
        d = {k: z[k] for k in ("c_pk", "c_ch", "pooled", "kabsch", "true")}
        ex = f"{EXTRA}/chan_{key}.npz"
        if key not in FUSION and os.path.exists(ex):
            x = np.load(ex)
            for k in d:
                d[k] = np.concatenate([d[k], x[k]])
        chans[key] = d
    extra_meta = json.load(open(f"{EXTRA}/meta.json")) if os.path.exists(f"{EXTRA}/meta.json") else None
    if extra_meta:
        for p, info in extra_meta["probes"].items():
            subj[p], lab[p], long_of[p] = info["subject"], info["lab"], info["long_id"]
            for c, f in info["fine_id"].items():
                fid[(p, int(c))] = int(f)

    keys = sorted({(str(p), int(c)) for d in chans.values() for p, c in zip(d["c_pk"], d["c_ch"])})
    idx = {k: i for i, k in enumerate(keys)}
    N = len(keys)
    true = np.full((N, 3), np.nan)
    for key, d in chans.items():
        ii = np.array([idx[(str(p), int(c))] for p, c in zip(d["c_pk"], d["c_ch"])])
        prev = true[ii]
        ok = np.isfinite(prev[:, 0])
        assert np.allclose(prev[ok], d["true"][ok], atol=1.0), f"true mismatch in {key}"
        true[ii] = d["true"]
        d["ii"] = ii
    assert np.isfinite(true).all()

    probes = sorted({p for p, _ in keys})
    off = {p: keys.index((p, min(c for q, c in keys if q == p))) for p in probes}
    count = {p: sum(1 for q, _ in keys if q == p) for p in probes}

    # ---- regions
    fids = np.array([fid.get(k, 0) for k in keys])
    uniq = sorted(set(fids.tolist()))
    reg = []
    for f in uniq:
        n = tree.get(f)
        if n is None:
            reg.append(dict(id=f, acr="?", name="unassigned", color="#BBBBBB", major="")); continue
        path = [int(x) for x in n["structure_id_path"].strip("/").split("/")]
        major = next((nm for mid, nm in MAJOR if mid in path), "")
        reg.append(dict(id=f, acr=n["acronym"], name=n["name"], color="#" + n["color_hex_triplet"], major=major))
    rindex = {f: i for i, f in enumerate(uniq)}
    ridx = np.array([rindex[f] for f in fids], np.uint16)
    chn = np.array([c for _, c in keys], np.uint16)

    with open(f"{OUT}/gt.bin", "wb") as fh:
        fh.write(np.round(true).astype(np.int16).tobytes()); fh.write(ridx.tobytes()); fh.write(chn.tobytes())

    # ---- per-method binaries
    for key, d in chans.items():
        arr = np.full((2, N, 3), MISSING, np.int16)
        t = true[d["ii"]]
        arr[0, d["ii"]] = np.round(display(d["pooled"], t, d["c_pk"].astype(str))).astype(np.int16)
        arr[1, d["ii"]] = np.round(display(d["kabsch"], t, d["c_pk"].astype(str))).astype(np.int16)
        with open(f"{OUT}/methods/{key}.bin", "wb") as fh:
            fh.write(arr.tobytes())

    # ---- per-probe RMSE tables (seed mean; seed 1) from the stored full-test inputs
    pmain = zm["probes"].astype(str).tolist()
    short_of = {v: k for k, v in long_of.items()}
    pfus = [short_of[p] for p in zf["probes"].astype(str)]
    extra_rmse = json.load(open(f"{EXTRA}/rmse.json")) if os.path.exists(f"{EXTRA}/rmse.json") else {}
    rm = {}
    for key, label, group, col, ns in METHODS:
        z, plist = (zf, pfus) if key in FUSION else (zm, pmain)
        tab = {}
        for st in ("kabsch", "pooled", "raw"):
            mean, s1 = per_probe_seed_mean(z[f"{key}_{st}_e2"], z[f"{key}_{st}_n"])
            assert z[f"{key}_{st}_e2"].shape[0] == ns, (key, st)
            tab[st] = {p: float(v) for p, v in zip(plist, mean)}
            if st == "kabsch":
                tab["kabsch_s1"] = {p: float(v) for p, v in zip(plist, s1)}
        for p, vals in extra_rmse.get(key, {}).items():
            for st, v in vals.items():
                tab[st][p] = v
        rm[key] = {st: [round(tab[st][p], 1) if p in tab[st] else None for p in probes] for st in tab}
        # gate: drawn seed-1 Kabsch channels reproduce the stored seed-1 per-probe RMSE
        d = chans[key]
        e2 = ((fold(d["kabsch"]) - fold(d["true"])) ** 2).sum(1)
        pk = d["c_pk"].astype(str)
        worst = max(abs(np.sqrt(e2[pk == p].mean()) - tab["kabsch_s1"][p]) for p in set(pk) if p in tab["kabsch_s1"])
        assert worst < 0.01, (key, worst)

    methods = []
    for key, label, group, col, ns in METHODS:
        v = np.array([x for x in rm[key]["kabsch"] if x is not None])
        methods.append(dict(key=key, label=label, group=group, color=col, seeds=ns, n_probes=int(len(v)),
                            kabsch_mean=round(float(v.mean()), 1), kabsch_sd=round(float(v.std(ddof=1)), 1)))
    out = dict(
        probes=[dict(k=p, id=long_of.get(p, p), subject=subj.get(p, "?"), lab=lab.get(p, "?"),
                     off=off[p], n=count[p], fusion=all(rm[k]["kabsch"][i] is not None for k in FUSION))
                for i, p in enumerate(probes)],
        methods=methods, rmse=rm, regions=reg, n_channels=N,
        notes=dict(
            cohort="every chunk of every test probe (IBL compact_atlas_ibl_v2_full/test); "
                   "micromanipulator rows exclude probe b73ef9cf (no usable micromanipulator record)",
            statistic="per-probe left-hemisphere-folded 3D RMSE after inverse-variance channel pooling and "
                      "unweighted per-probe Kabsch; per decoder seed, then mean over seeds",
            drawn="decoder seed 1; display = folded prediction mirrored into the probe's true hemisphere",
            kabsch="known probe geometry fitted to the predictions; no true coordinates in the fit"))
    json.dump(out, open(f"{OUT}/probes.json", "w"), separators=(",", ":"))
    print(f"{len(probes)} probes, {N} channels, {len(reg)} regions")
    for m in methods:
        print(f"  {m['label']:28s} n={m['n_probes']:3d}  {m['kabsch_mean']:.1f} ± {m['kabsch_sd']:.1f}")


if __name__ == "__main__":
    main()
