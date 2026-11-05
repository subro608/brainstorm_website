#!/usr/bin/env python3
"""Chronic zero-shot recordings (collaborator data, mouse M05, NP 2.0) for the website player (2026-11-05).

Source (pl2820): /scratch/pl2820/ray_results/eval_v2/chronic_mishi_large30k_full/<session>/predictions.npz
  Wav2Vec2 Large ManipDistc step 30k encoder, zero-shot, 8-seed MLP readout + per-chunk rigid NP2.0 fit (pred_xyz_kabsch,
  fit_tip_xyz = channel 0, fit_base_xyz = top channel, 2865 um apart). This is the field of the paper figure
  (viz_chronic_mishi_v2_timechunk.py --field pred_xyz_kabsch -> ..._full.png).
Region labels use the SAME lookup as that figure (viz_session_inference_timechunk.build_atlas_lookup / relabel_xyz /
build_label_map over all Allen acronyms). Recording blocks and linear-maze tracking come from the collaborator's
MergePoints.events.mat and Tracking.Behavior.mat (/scratch/mc10168/mishi/5DAYS), same session clock as the .lfp.

Writes site/data/chronic/:
  sessions.json            sessions, blocks, palette (acronym, colour), per-session file names
  <i>_labels.bin           uint8 [192 depth rows (surface -> tip), n_chunks] palette indices (255 = outside atlas)
  <i>_probe.bin            int16 [n_chunks, 6] tip xyz, base xyz (CCF um: AP, DV, ML)
  <i>_track.bin            float32 [n, 3] t (s, session clock), x, y (cm), 10 Hz, linear-maze block only
"""
import json, os, re, sys
import numpy as np
import scipy.io as sio

sys.path.insert(0, "/scratch/pl2820/Alphabrain_staging/Lfp2vec_benchmarks/scripts")
from viz_session_inference_timechunk import build_atlas_lookup, relabel_xyz, build_label_map  # noqa: E402

ROOT = "/scratch/pl2820/ray_results/eval_v2/chronic_mishi_large30k_full"
MISHI = "/scratch/mc10168/mishi/5DAYS"
OUT = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/website_lfpformer_2026-11-05/site/data/chronic"
SESSIONS = ["TES_sResp_M05_20240729", "TES_sResp_M05_20240730", "TES_sResp_M05_20240731",
            "TES_sResp_M05_20241014", "TES_sResp_M05_20241015"]
LABELS = ["Jul 29", "Jul 30", "Jul 31", "Oct 14", "Oct 15"]
TRACK_HZ = 10.0


def blocks(session):
    mp = sio.loadmat(f"{MISHI}/{session}/{session}.MergePoints.events.mat", squeeze_me=True,
                     struct_as_record=False)["MergePoints"]
    out = []
    for (t0, t1), name in zip(np.atleast_2d(mp.timestamps), np.atleast_1d(mp.foldernames)):
        leaf = re.split(r"[\\/]", str(name))[-1]
        tag = re.sub(r"_g\d+.*$", "", leaf).lower()
        kind = "maze" if "maze" in tag else ("pre" if tag.startswith("pre") else ("post" if tag.startswith("post") else "other"))
        label = {"maze": "linear maze", "pre": "home cage (pre)", "post": "home cage (post)"}.get(kind, tag.replace("_", " "))
        out.append(dict(t0=round(float(t0), 3), t1=round(float(t1), 3), kind=kind, label=label))
    return out


def track(session, maze):
    b = sio.loadmat(f"{MISHI}/{session}/{session}.Tracking.Behavior.mat", squeeze_me=True,
                    struct_as_record=False)["behavior"]
    t = np.asarray(b.timestamps, float)
    # x/y in cm for every session: pixel coordinates x the session's cm-per-pixel factor
    # (session 1 stores position.x/y in metres around the centre, the others in cm).
    x = np.asarray(b.position.x_in_pixel, float) * float(b.convFact)
    y = np.asarray(b.position.y_in_pixel, float) * float(b.convFact)
    keep = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & (t >= maze["t0"]) & (t <= maze["t1"])
    t, x, y = t[keep], x[keep], y[keep]
    grid = np.arange(t[0], t[-1], 1.0 / TRACK_HZ)
    out = np.stack([grid, np.interp(grid, t, x), np.interp(grid, t, y)], 1).astype(np.float32)
    return out, dict(fs=float(b.samplingRate), n_frames=int(len(b.timestamps)), t_first=float(t[0]), t_last=float(t[-1]))


def main():
    os.makedirs(OUT, exist_ok=True)
    annotation, parent, acr_to_color, sid_to_acr = build_atlas_lookup()
    acr_to_id, id_to_acr = build_label_map(np.array([], dtype=str), sid_to_acr)
    used = {}
    per = []
    for si, (s, lab) in enumerate(zip(SESSIONS, LABELS)):
        z = np.load(f"{ROOT}/{s}/predictions.npz", allow_pickle=True)
        pred = z["pred_xyz_kabsch"]                       # (384, n_chunks, 3)
        axial = z["channel_axial_um"].astype(np.float32)
        assert float(z["chunk_dur_sec"]) == 3.0 and float(z["win_start_sec"]) == 0.0
        # one row per depth (two columns share each depth): first channel at each depth, surface -> tip
        depths = np.unique(axial)[::-1]
        rows = np.array([np.flatnonzero(axial == d)[0] for d in depths])
        assert len(rows) == 192
        n_ch, n_chunks, _ = pred.shape
        lab_full = relabel_xyz(pred[rows].reshape(-1, 3), annotation, parent, sid_to_acr, acr_to_id).reshape(192, n_chunks)
        # palette indices shared across sessions
        idx = np.full(lab_full.shape, 255, np.uint8)
        for lid in np.unique(lab_full[lab_full >= 0]):
            if int(lid) not in used:
                used[int(lid)] = len(used)
            idx[lab_full == lid] = used[int(lid)]
        assert len(used) < 255
        idx.tofile(f"{OUT}/{si}_labels.bin")
        tip, base = z["fit_tip_xyz"], z["fit_base_xyz"]
        assert np.abs(pred[0] - tip).max() < 1e-3
        np.round(np.concatenate([tip, base], 1)).astype(np.int16).tofile(f"{OUT}/{si}_probe.bin")
        blk = blocks(s)
        maze = next(b for b in blk if b["kind"] == "maze")
        tr, tinfo = track(s, maze)
        tr.tofile(f"{OUT}/{si}_track.bin")
        res = z["fit_residual_um"]
        per.append(dict(i=si, label=lab, date=s[-8:], n_chunks=int(n_chunks), chunk_s=3.0,
                        duration_s=float(n_chunks * 3.0), blocks=blk, n_track=int(len(tr)), track_hz=TRACK_HZ,
                        track_info=tinfo, fit_residual_median_um=round(float(np.median(res)), 1),
                        outside_atlas_frac=round(float((idx == 255).mean()), 4)))
        print(s, n_chunks, "chunks;", len(tr), "track samples;", "outside atlas %.3f" % (idx == 255).mean(), flush=True)
    palette = [None] * len(used)
    for lid, k in used.items():
        acr = id_to_acr[lid]
        palette[k] = dict(acr=acr, color=acr_to_color.get(acr, "#888888"))
    meta = dict(
        sessions=per, palette=palette, rows=192, row_pitch_um=15.0,
        model="Wav2Vec2 large, MaskPred + ManipDistc, step 30k; zero-shot; 8-seed MLP readout + per-chunk rigid NP2.0 fit",
        source=ROOT, region_lookup="viz_session_inference_timechunk.relabel_xyz (same as the paper figure)")
    json.dump(meta, open(f"{OUT}/sessions.json", "w"), separators=(",", ":"))
    print("palette", len(palette), "regions")


if __name__ == "__main__":
    main()
