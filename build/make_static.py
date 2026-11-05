#!/usr/bin/env python3
"""Website static assets: brain mesh (uint16 quantised) + web copies of the final NeurIPS figures."""
import json, os, shutil
import numpy as np
from PIL import Image
W = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/website_lfpformer_2026-11-05"
F = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/plots/neurips_2026_final"
MESH = "/scratch/sd5963/AlphaBrain/Alphabrain_staging/data/allen_ccf/ccf_full_mesh_step5.npz"
S = f"{W}/site"

# brain mesh: CCF um (AP, DV, ML); vertices quantised to 1 um in uint16, faces uint16
m = np.load(MESH)
v, f = m["v"].astype(np.float64), m["f"].astype(np.int64)
assert v.max() < 65535 and len(v) < 65535
vq = np.round(v).astype(np.uint16)
with open(f"{S}/data/brain_mesh.bin", "wb") as fh:
    fh.write(vq.tobytes()); fh.write(f.astype(np.uint16).tobytes())
mesh_meta = dict(n_vertices=int(len(v)), n_faces=int(len(f)), order="AP,DV,ML", unit="um",
                 source="Allen CCF v3 root, marching cubes step 5 (ccf_full_mesh_step5.npz)")

FIGS = [  # id, src png, src pdf, caption txt, title
    ("fig3", "fig3/fig3_new_composite.png", "fig3/fig3_new_composite.pdf", "fig3/fig3_new_composite_caption.txt", "Figure 3 · IBL 3D localization"),
    ("fig4", "fig4/fig4_final.png", "fig4/fig4_final.pdf", "fig4/fig4_final_caption.txt", "Figure 4 · Cross-dataset brain-region classification"),
    ("fig5", "fig5/fig5_base_linear_0_5k_115k.png", "fig5/fig5_base_linear_0_5k_115k.pdf", "fig5/fig5_base_linear_0_5k_115k_caption.txt", "Figure 5 · Coordinates emerge during pretraining"),
    ("fig6", "fig6/fig6_scaling.png", "fig6/fig6_scaling.pdf", "fig6/fig6_scaling_caption.txt", "Figure 6 · Data, model-size and compute scaling"),
    ("supp_datasets", "supp_fig_datasets/supp_fig6_datasets_overview.png", "supp_fig_datasets/supp_fig6_datasets_overview.pdf", "supp_fig_datasets/supp_fig6_caption.txt", "Supplementary · Dataset overview"),
    ("supp7a", "supp_fig7/fig7a.png", "supp_fig7/fig7a.pdf", "supp_fig7/fig7a_caption.txt", "Supplementary Fig. 7a · Per region, subject, probe and lab"),
    ("supp7b", "supp_fig7/fig7b.png", "supp_fig7/fig7b.pdf", "supp_fig7/fig7b_caption.txt", "Supplementary Fig. 7b · Depth-5 confusion matrices"),
    ("supp11", "supp_fig11/fig11_suppl.png", "supp_fig11/fig11_suppl.pdf", "supp_fig11/fig11_suppl_caption.txt", "Supplementary Fig. 11 · LFPformer readouts with the micromanipulator"),
]
figs = []
for fid, png, pdf, cap, title in FIGS:
    im = Image.open(f"{F}/{png}").convert("RGB")
    w = min(im.size[0], 2000)
    if im.size[0] > w:
        im = im.resize((w, round(im.size[1] * w / im.size[0])), Image.LANCZOS)
    im.save(f"{S}/figures/{fid}.webp", "WEBP", quality=90, method=6)
    sm = im.copy(); sm.thumbnail((900, 900 * 4), Image.LANCZOS)
    sm.save(f"{S}/figures/{fid}_sm.webp", "WEBP", quality=86, method=6)
    shutil.copyfile(f"{F}/{pdf}", f"{S}/figures/{fid}.pdf")
    caption = open(f"{F}/{cap}").read().strip()
    figs.append(dict(id=fid, title=title, w=im.size[0], h=im.size[1], caption=caption,
                     src_png=f"{F}/{png}", bytes_webp=os.path.getsize(f"{S}/figures/{fid}.webp")))
    print(fid, im.size, os.path.getsize(f"{S}/figures/{fid}.webp") // 1024, "KB")
json.dump(dict(mesh=mesh_meta, figures=figs), open(f"{W}/build/static_manifest.json", "w"), indent=1)
json.dump(figs, open(f"{S}/data/figures.json", "w"), indent=0, ensure_ascii=False)
print("mesh", mesh_meta, os.path.getsize(f"{S}/data/brain_mesh.bin") // 1024, "KB")
