# LFPformer project page (brainstorm-lfp2vec-core.vercel.app), built 2026-11-05

User request (2026-11-05): restyle https://brainstorm-lfp2vec-core.vercel.app/ like https://visinf.github.io/MARCO/ with all the
new camera-ready plots (`plots/neurips_2026_final/`), full-recording predictions vs ground truth for all 119 test probes
for the Fig 3c methods, and a 3D brain with predicted and true probe tracks; deploy.

Live: https://lfpformer-nyu.vercel.app (domain added 2026-11-05 at the user's request; also still https://brainstorm-lfp2vec-core.vercel.app). Vercel CLI project `brainstorm-lfp2vec-core`, no Git link; production deploys update both domains. The previous July site is still in
Vercel's deployment history (brainstorm-lfp2vec-core-159dosw42-…, 2026-07-18) and its source in `../vercel_deploy_noemb/`.

## Layout
- `site/` = everything that is deployed (static, no build): `index.html`, `assets/{site.css,app.js}` (three.js 0.160 from
  jsdelivr), `data/` (binary arrays + probes.json), `figures/` (webp + vector PDFs of the final figures).
- `build/` = scripts and intermediates (not deployed).

## Data pipeline
1. `build/export_chan.py` (+ `export_chan.sbatch`, array 21928577, 15 tasks): seed-1 per-channel pooled / Kabsch / true
   coordinates for each Fig 3c method with full-recording predictions, from the 118-probe full-test run
   `ray_results/Neurips_2026/pathb_newmodels/fig3_fulltest118_2026-11-02`. Gate: seed-1 per-probe Kabsch RMSE equals the stored
   `fig3_{main,fusion}_inputs_full118.npz` values (max diff 2e-12 um, all 15 methods).
2. Probe 119 (b73ef9cf, SWC_042, no usable micromanipulator record): every chunk (68,800) decoded for the LFP-only methods in
   `pathb_newmodels/fig3_fulltest_probe119_2026-11-05/` (README there, same checkpoints/recipe; gates on its 6,256 eval_v2 rows).
   `build/export_probe119.py` scores it (8 decoder seeds for LFPformer) -> `build/probe119/`.
3. `build/build_data.py` -> `site/data/` (gt.bin, methods/<key>.bin int16 display coordinates, probes.json with per-probe
   seed-mean Kabsch / pooled / raw RMSE). Asserts that the drawn seed-1 arrays reproduce the stored seed-1 per-probe RMSE.
   Fold midline = 5739 um (ALLEN_MIDLINE_UM), as in every Fig 3 score.
4. `build/make_static.py` -> brain mesh (`data/allen_ccf/ccf_full_mesh_step5.npz`, uint16) + figure webp/PDF copies.

Drawing vs scoring (fix 2026-11-05, user screenshot of b33e4298 with a Z-shaped predicted track): scores stay per-channel
left/right folded as in the paper, but each predicted track is drawn UNFOLDED and mirrored as a whole probe into the true
track's hemisphere (`display()` in build_data.py). The first version mirrored per channel, which bent the rigid Kabsch track
wherever the true track crosses the midline (8 test probes; b33e4298 true ML 5065-6155 um, midline 5739, 128/380 channels on
the right). After the fix every drawn Kabsch track of every method is straight to <= 25 um (Neuropixels column stagger).
For b33e4298 the folded score is 1054 um (TF + manip., seed 1) vs 1133 um unfolded, so folding does not inflate it.

Fusion rows (linear/TF + manip., manipulator alone) cover 118 probes; all LFP-only rows cover 119. The two wav2vec2 control
rows of Fig 3c were only scored on the eval_v2 sample (493,744 chunks), not on full recordings, so they are not in the explorer.
Full-recording per-probe means (this page) differ slightly from the paper's eval_v2 numbers (e.g. LFPformer TF 1015 vs 1012);
the Performance section quotes the paper numbers and says so.

## Deploy
Copy `site/` to a folder OUTSIDE any git repo, then
`python3 ~/.claude/skills/vercel-deploy/scripts/ship.py <copy> --cli brainstorm-lfp2vec-core --prod`.
Deploying `site/` in place picks up the Alphabrain_staging git commit (author sd5963@cs667.hpc.nyu.edu) and Vercel BLOCKS it
(TEAM_ACCESS_REQUIRED, "commit author doesn't have permission"); that happened once on 2026-11-05 (dpl_5U3hukuF5ZUPKCq19cADfydNuVhr).

Local render check: `python3 -m http.server 8765 --bind 127.0.0.1` in `site/`, then headless chromium (see memory
reference-headless-chrome-render-check); `?rotate=0` disables auto-rotation for screenshots.

## Chronic zero-shot player (added 2026-11-05, user: "like a recording" + linear-maze x-y)
`build/export_chronic.py` -> `site/data/chronic/` (sessions.json; per session `<i>_labels.bin` uint8 [192 depth rows x chunks],
`<i>_probe.bin` int16 tip/base per 3-s chunk, `<i>_track.bin` float32 t/x/y at 10 Hz). Source: pl2820
`ray_results/eval_v2/chronic_mishi_large30k_full/<session>/predictions.npz` (Large ManipDistc step 30k, zero-shot, 8-seed MLP +
per-chunk rigid NP2.0 fit; field pred_xyz_kabsch = the paper figure), regions via pl2820's `viz_session_inference_timechunk`
lookup (identical colouring), blocks + tracking from the collaborator's MergePoints / Tracking.Behavior .mat files
(`/scratch/mc10168/mishi/5DAYS`, same clock as the .lfp; x/y = pixel x convFact in cm, because session 1 stores position.x/y in m).
Front end: `site/assets/chronic.js`, two players with separate clocks (user 11-05): FULL = 3D probe + trail, regions along the probe, whole-session predictions-vs-time with blocks (10/60/300x); MAZE = mouse x-y, regions along the probe, maze-only predictions-vs-time + position-on-track trace (1/10/60x). Older single-clock version:
1x/10x/60x/300x). Sessions are labelled by date only (no TES study names on the page).
