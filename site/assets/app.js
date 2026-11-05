// LFPformer project page: 3D explorer + full-recording probe gallery + figure sections.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const $ = (s, r = document) => r.querySelector(s);
const MISSING = -32768;
const MID = 5700;
const MESH_NV = 39828, MESH_NF = 79364;

// ───────────────────────── state ─────────────────────────
const S = {
  data: null, N: 0, gt: null, ridx: null, chn: null,
  methods: {}, method: 'ours_tf', mode: 'kabsch', scope: 'all', sel: null,
  sort: 'err', proj: 'sag', listeners: [],
};
const cache = new Map();
async function loadMethod(key) {
  if (cache.has(key)) return cache.get(key);
  const p = fetch(`data/methods/${key}.bin`).then(r => r.arrayBuffer()).then(b => {
    const a = new Int16Array(b);
    return { pooled: a.subarray(0, S.N * 3), kabsch: a.subarray(S.N * 3) };
  });
  cache.set(key, p);
  return p;
}
const M = k => S.methods[k];
const rm = (k, i, st = 'kabsch') => S.data.rmse[k][st][i];
const fmt = v => (v == null ? 'n/a' : Math.round(v).toLocaleString('en-US'));
function emit() { for (const f of S.listeners) f(); }
function setMethod(k) { if (k !== S.method) { S.method = k; emit(); } }
function setSel(i) { if (i !== S.sel || !S.pinned) { S.sel = i; S.pinned = true; emit(); } }

// ───────────────────────── data ─────────────────────────
async function loadData() {
  const [d, g] = await Promise.all([
    fetch('data/probes.json').then(r => r.json()),
    fetch('data/gt.bin').then(r => r.arrayBuffer()),
  ]);
  S.data = d; S.N = d.n_channels;
  S.gt = new Int16Array(g, 0, S.N * 3);
  S.ridx = new Uint16Array(g, S.N * 6, S.N);
  S.chn = new Uint16Array(g, S.N * 8, S.N);
  for (const m of d.methods) S.methods[m.key] = m;
  // default selection: the median-error probe for LFPformer (TF)
  const order = d.probes.map((_, i) => i).filter(i => rm('ours_tf', i) != null)
    .sort((a, b) => rm('ours_tf', a) - rm('ours_tf', b));
  S.sel = order[Math.floor(order.length / 2)];
}
const regionColor = j => S.data.regions[S.ridx[j]].color;

// ───────────────────────── chips (shared method selector) ─────────────────────────
function buildChips(el, compact) {
  const groups = [];
  for (const m of S.data.methods) {
    let g = groups.find(x => x.name === m.group);
    if (!g) groups.push(g = { name: m.group, items: [] });
    g.items.push(m);
  }
  el.innerHTML = groups.map(g => `<div class="chip-group"><span class="chip-group-label">${g.name}</span>${
    g.items.map(m => `<button class="chip" data-k="${m.key}"><span class="dot" style="background:${m.color}"></span>${m.label}</button>`).join('')}</div>`).join('');
  el.addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) setMethod(b.dataset.k); });
  const sync = () => el.querySelectorAll('.chip').forEach(b => b.classList.toggle('is-active', b.dataset.k === S.method));
  S.listeners.push(sync); sync();
}
function segControl(el, attr, get, set) {
  el.addEventListener('click', e => { const b = e.target.closest('button'); if (b) set(b.dataset[attr]); });
  const sync = () => el.querySelectorAll('button').forEach(b => b.classList.toggle('is-active', b.dataset[attr] === get()));
  S.listeners.push(sync); sync();
}

// ───────────────────────── 3D explorer ─────────────────────────
const V = {};
const toV = (ap, dv, ml, out, o) => { out[o] = (ml - MID) / 1000; out[o + 1] = -(dv - 4000) / 1000; out[o + 2] = (ap - 6600) / 1000; };

async function initViewer() {
  const host = $('#viewer');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.autoRotate = !/[?&]rotate=0/.test(location.search); controls.autoRotateSpeed = 0.55;
  controls.addEventListener('change', () => { V.dirty = true; });
  Object.assign(V, { host, renderer, scene, camera, controls });
  resetView();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xdfe6f3, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, 0.9); dl.position.set(-6, 10, -8); scene.add(dl);

  // brain mesh
  const buf = await fetch('data/brain_mesh.bin').then(r => r.arrayBuffer());
  const vq = new Uint16Array(buf, 0, MESH_NV * 3), fq = new Uint16Array(buf, MESH_NV * 6, MESH_NF * 3);
  const pos = new Float32Array(MESH_NV * 3);
  for (let i = 0; i < MESH_NV; i++) toV(vq[3 * i], vq[3 * i + 1], vq[3 * i + 2], pos, 3 * i);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(new Uint16Array(fq), 1));
  geo.computeVertexNormals();
  const brain = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
    color: 0xbfc9de, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide, shininess: 12 }));
  brain.renderOrder = 0; scene.add(brain);

  // ground-truth tracks (static), one segment per neighbouring channel pair
  const segs = [], segProbe = [], segStart = [];
  S.data.probes.forEach((p, i) => { segStart.push(segs.length); for (let j = p.off; j < p.off + p.n - 1; j++) { segs.push(j); segProbe.push(i); } });
  V.segs = segs; V.segProbe = segProbe; V.segStart = segStart; V.gtScope = 'all';
  const gp = new Float32Array(segs.length * 6), gc = new Float32Array(segs.length * 6);
  const col = new THREE.Color();
  segs.forEach((j, s) => {
    toV(S.gt[3 * j], S.gt[3 * j + 1], S.gt[3 * j + 2], gp, 6 * s);
    toV(S.gt[3 * j + 3], S.gt[3 * j + 4], S.gt[3 * j + 5], gp, 6 * s + 3);
    col.set(regionColor(j)); col.toArray(gc, 6 * s);
    col.set(regionColor(j + 1)); col.toArray(gc, 6 * s + 3);
  });
  V.gtPos = gp; V.gtCol = gc;
  V.matGT = new LineMaterial({ linewidth: 2.2, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false });
  const gg = new LineSegmentsGeometry(); gg.setPositions(gp); gg.setColors(gc);
  V.gtLines = new LineSegments2(gg, V.matGT); V.gtLines.renderOrder = 4; scene.add(V.gtLines);

  V.matPred = new LineMaterial({ linewidth: 1.6, color: 0x882255, transparent: true, opacity: 0.6, depthWrite: false });
  V.predLines = new LineSegments2(new LineSegmentsGeometry(), V.matPred); V.predLines.renderOrder = 3; scene.add(V.predLines);
  V.matPts = new THREE.PointsMaterial({ size: 2.6, sizeAttenuation: false, color: 0x882255, transparent: true, opacity: 0.55, depthWrite: false });
  V.predPts = new THREE.Points(new THREE.BufferGeometry(), V.matPts); V.predPts.renderOrder = 3; scene.add(V.predPts);
  V.hl = new THREE.Group(); V.hl.renderOrder = 5; scene.add(V.hl);

  // interaction
  V.ray = new THREE.Raycaster(); V.ray.params.Line2 = { threshold: 6 }; V.ray.camera = camera;
  V.mouse = new THREE.Vector2(); V.hover = null;
  const stopRotate = () => { controls.autoRotate = false; $('#btn-rotate').classList.remove('is-on'); };
  renderer.domElement.addEventListener('pointerdown', e => { V.down = [e.clientX, e.clientY]; stopRotate(); });
  renderer.domElement.addEventListener('pointerup', e => {
    if (V.down && Math.hypot(e.clientX - V.down[0], e.clientY - V.down[1]) < 4) {
      const i = pick(e); if (i != null) { setSel(i); if (S.scope === 'all') focusProbe(i, false); }
    }
  });
  renderer.domElement.addEventListener('pointermove', e => {
    const i = pick(e); const tt = $('#tooltip');
    if (i == null) { tt.style.display = 'none'; renderer.domElement.style.cursor = ''; return; }
    const p = S.data.probes[i], r = host.getBoundingClientRect();
    tt.innerHTML = `${p.k} · ${p.subject} · ${fmt(rm(S.method, i))} µm`;
    tt.style.left = (e.clientX - r.left + 12) + 'px'; tt.style.top = (e.clientY - r.top + 12) + 'px'; tt.style.display = 'block';
    renderer.domElement.style.cursor = 'pointer';
  });
  renderer.domElement.addEventListener('pointerleave', () => { $('#tooltip').style.display = 'none'; });
  renderer.domElement.addEventListener('wheel', stopRotate, { passive: true });
  $('#btn-rotate').classList.toggle('is-on', controls.autoRotate);
  $('#btn-rotate').onclick = () => { controls.autoRotate = !controls.autoRotate; $('#btn-rotate').classList.toggle('is-on', controls.autoRotate); };
  $('#btn-reset').onclick = () => resetView();

  new ResizeObserver(resize).observe(host); resize(); resetView();
  $('#viewer-loading').remove();
  // render on demand: only while the camera moves (damping / auto-rotate / fly-to) or after a scene update
  V.dirty = true;
  renderer.setAnimationLoop(() => {
    controls.update();
    if (V.dirty || V.flying) { renderer.render(scene, camera); V.dirty = false; }
  });
}
function resetView() {
  const f = Math.max(1, 1.2 / (V.camera.aspect || 1.2));   // pull back on narrow (phone) viewers
  V.camera.position.set(-17 * f, 9.5 * f, -12.5 * f); V.controls.target.set(0, -0.3, 0.4); V.camera.lookAt(V.controls.target); V.dirty = true;
}
function resize() {
  const w = V.host.clientWidth, h = Math.max(420, V.host.clientHeight);
  V.renderer.setSize(w, h, false); V.camera.aspect = w / h; V.camera.updateProjectionMatrix(); V.dirty = true;
  for (const m of [V.matGT, V.matPred, ...(V.hlMats || [])]) m && m.resolution.set(w, h);
}
function pick(e) {
  const r = V.renderer.domElement.getBoundingClientRect();
  V.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  V.ray.setFromCamera(V.mouse, V.camera);
  const hit = V.ray.intersectObject(V.gtLines, false)[0];
  if (!hit) return null;
  return S.scope === 'all' ? V.segProbe[hit.faceIndex] : S.sel;
}
function focusProbe(i, fly = true) {
  const p = S.data.probes[i]; const c = [0, 0, 0], v = [0, 0, 0];
  for (let j = p.off; j < p.off + p.n; j++) { toV(S.gt[3 * j], S.gt[3 * j + 1], S.gt[3 * j + 2], v, 0); c[0] += v[0]; c[1] += v[1]; c[2] += v[2]; }
  const t = new THREE.Vector3(c[0] / p.n, c[1] / p.n, c[2] / p.n);
  if (!fly) return;
  const off = V.camera.position.clone().sub(V.controls.target).setLength(S.scope === 'cmp' ? 19 : 13);
  animateCamera(t, t.clone().add(off));
}
function animateCamera(target, pos) {
  const t0 = performance.now(), a = V.controls.target.clone(), b = V.camera.position.clone();
  V.controls.autoRotate = false; $('#btn-rotate').classList.remove('is-on');
  const step = now => {
    const u = Math.min(1, (now - t0) / 700), k = u * u * (3 - 2 * u);
    V.controls.target.lerpVectors(a, target, k); V.camera.position.lerpVectors(b, pos, k); V.dirty = true;
    if (u < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function trackSegments(arr, probeIdx) {
  // predicted-track segments for the given probes (skip probes the method does not cover)
  const out = [];
  for (const i of probeIdx) {
    const p = S.data.probes[i];
    if (arr[3 * p.off] === MISSING) continue;
    // join channels four apart (same Neuropixels column), so staggered sites do not zigzag
    for (let j = p.off; j + 4 < p.off + p.n; j += 4) out.push([j, j + 4]);
  }
  const pos = new Float32Array(out.length * 6);
  out.forEach(([a, b], s) => { toV(arr[3 * a], arr[3 * a + 1], arr[3 * a + 2], pos, 6 * s); toV(arr[3 * b], arr[3 * b + 1], arr[3 * b + 2], pos, 6 * s + 3); });
  return pos;
}
function pointCloud(arr, probeIdx) {
  const idx = [];
  for (const i of probeIdx) { const p = S.data.probes[i]; if (arr[3 * p.off] === MISSING) continue; for (let j = p.off; j < p.off + p.n; j++) idx.push(j); }
  const pos = new Float32Array(idx.length * 3);
  idx.forEach((j, s) => toV(arr[3 * j], arr[3 * j + 1], arr[3 * j + 2], pos, 3 * s));
  return pos;
}
function probeGT(i) {
  // GT segment positions/colours of one probe (its segments are contiguous in V.gtPos)
  const s0 = V.segStart[i], n = S.data.probes[i].n - 1;
  return [V.gtPos.slice(6 * s0, 6 * (s0 + n)), V.gtCol.slice(6 * s0, 6 * (s0 + n))];
}
function disposeGroup(g) { for (const o of [...g.children]) { g.remove(o); o.geometry?.dispose(); } }

let viewToken = 0;
async function updateViewer() {
  if (!V.scene) return;
  const tok = ++viewToken;
  const m = M(S.method), arrs = await loadMethod(S.method);
  if (tok !== viewToken) return;
  const all = S.data.probes.map((_, i) => i);
  const scope = S.scope, sel = S.sel;
  const showIdx = scope === 'all' ? all : [sel];
  const arr = S.mode === 'kabsch' ? arrs.kabsch : arrs.pooled;
  const c = new THREE.Color(m.color);
  V.matPred.color.copy(c); V.matPts.color.copy(c);
  $('#legend-pred').style.background = m.color;
  $('#legend-pred-label').textContent = m.label;

  // base layers
  V.predLines.visible = S.mode === 'kabsch' && scope !== 'cmp';
  V.predPts.visible = S.mode === 'pooled' && scope !== 'cmp';
  if (V.predLines.visible) { const g = new LineSegmentsGeometry(); g.setPositions(trackSegments(arr, showIdx)); V.predLines.geometry.dispose(); V.predLines.geometry = g; }
  if (V.predPts.visible) { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pointCloud(arr, showIdx), 3)); V.predPts.geometry.dispose(); V.predPts.geometry = g; }
  // GT layer: every probe in "all" scope, only the selected probe otherwise
  const gtScope = scope === 'all' ? 'all' : `p${sel}`;
  if (V.gtScope !== gtScope) {
    const g = new LineSegmentsGeometry();
    if (scope === 'all') { g.setPositions(V.gtPos); g.setColors(V.gtCol); }
    else { const [pos, colr] = probeGT(sel); g.setPositions(pos); g.setColors(colr); }
    V.gtLines.geometry.dispose(); V.gtLines.geometry = g; V.gtScope = gtScope;
  }
  const dim = scope === 'all' && S.pinned;
  V.matGT.opacity = dim ? 0.3 : 0.95; V.matGT.linewidth = scope === 'all' ? 2.4 : 5;
  V.matPred.opacity = dim ? 0.25 : 0.6; V.matPred.linewidth = scope === 'all' ? 1.6 : 3.5;
  V.matPts.opacity = dim ? 0.18 : 0.55; V.matPts.size = scope === 'all' ? 2.6 : 5;

  // highlight layer
  disposeGroup(V.hl); V.hlMats = [];
  const w = V.host.clientWidth, h = Math.max(420, V.host.clientHeight);
  const addLines = (pos, colors, color, lw, op = 1) => {
    const g = new LineSegmentsGeometry(); g.setPositions(pos); if (colors) g.setColors(colors);
    const mat = new LineMaterial({ linewidth: lw, color: color ?? 0xffffff, vertexColors: !!colors, transparent: true, opacity: op, depthWrite: false });
    mat.resolution.set(w, h); V.hlMats.push(mat);
    const o = new LineSegments2(g, mat); o.renderOrder = 6; V.hl.add(o);
  };
  if (S.pinned && sel != null && scope === 'all') {
    if (S.mode === 'kabsch') addLines(trackSegments(arr, [sel]), null, c, 4);
    else {
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pointCloud(arr, [sel]), 3));
      const o = new THREE.Points(g, new THREE.PointsMaterial({ size: 5.5, sizeAttenuation: false, color: c, transparent: true, opacity: 0.9, depthWrite: false }));
      o.renderOrder = 6; V.hl.add(o);
    }
    const [pos, colr] = probeGT(sel);
    addLines(pos, colr, null, 5.5);
  }
  if (scope === 'cmp' && sel != null) {
    const keys = S.data.methods.map(x => x.key);
    const loaded = await Promise.all(keys.map(loadMethod));
    if (tok !== viewToken) return;
    keys.forEach((k, n) => {
      const a = S.mode === 'kabsch' ? loaded[n].kabsch : loaded[n].pooled;
      const mm = M(k), main = k === S.method;
      if (S.mode === 'kabsch') addLines(trackSegments(a, [sel]), null, new THREE.Color(mm.color), main ? 5 : 2.4, main ? 1 : 0.8);
      else {
        const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pointCloud(a, [sel]), 3));
        const o = new THREE.Points(g, new THREE.PointsMaterial({ size: main ? 5 : 3, sizeAttenuation: false, color: new THREE.Color(mm.color), transparent: true, opacity: main ? 0.95 : 0.6, depthWrite: false }));
        V.hl.add(o);
      }
    });
  }
  updateOverlay(); V.dirty = true;
}
function updateOverlay() {
  const m = M(S.method), d = S.data;
  let html = `<div class="big">${m.label}</div>${fmt(m.kabsch_mean)} ± ${fmt(m.kabsch_sd)} µm · ${m.n_probes} probes`;
  if (S.sel != null && S.pinned) {
    const p = d.probes[S.sel];
    html += `<br><b>${p.k}</b>: ${fmt(rm(S.method, S.sel))} µm`;
  }
  if (S.scope === 'cmp' && S.sel != null) {
    const rows = d.methods.map(x => [x, rm(x.key, S.sel)]).filter(r => r[1] != null).sort((a, b) => a[1] - b[1]);
    html += '<div style="margin-top:.35rem;display:grid;grid-template-columns:repeat(2,auto);gap:0 .9rem;font-size:.76rem">' + rows.map(([x, v]) =>
      `<span><i style="display:inline-block;width:16px;height:4px;border-radius:2px;background:${x.color};vertical-align:middle;margin-right:.35rem"></i>${x.label}: ${fmt(v)} µm</span>`).join('') + '</div>';
  }
  $('#viewer-overlay').innerHTML = html;
}

// ───────────────────────── side panel ─────────────────────────
function probeOrder(key) {
  const d = S.data;
  return d.probes.map((_, i) => i).sort((a, b) => {
    const va = rm(key, a), vb = rm(key, b);
    if (va == null) return 1; if (vb == null) return -1; return va - vb;
  });
}
function updateList() {
  const d = S.data, q = $('#probe-search').value.trim().toLowerCase();
  const vmax = Math.max(...d.probes.map((_, i) => rm(S.method, i) ?? 0));
  const color = M(S.method).color;
  $('#probe-list').innerHTML = probeOrder(S.method).filter(i => {
    const p = d.probes[i]; return !q || `${p.k} ${p.id} ${p.subject} ${p.lab}`.toLowerCase().includes(q);
  }).map(i => {
    const p = d.probes[i], v = rm(S.method, i);
    return `<div class="probe-row${i === S.sel ? ' is-active' : ''}" data-i="${i}"><span class="mono">${p.k}</span>
      <span class="bar"><i style="width:${v == null ? 0 : (100 * v / vmax).toFixed(1)}%;background:${color}"></i></span><span class="v">${fmt(v)}</span></div>`;
  }).join('');
  const list = $('#probe-list'), a = $('#probe-list .is-active');
  if (a) { const top = a.offsetTop; if (top < list.scrollTop || top > list.scrollTop + list.clientHeight - 30) list.scrollTop = top - list.clientHeight / 2; }
}
function updateCard() {
  const d = S.data, i = S.sel; if (i == null) return;
  const p = d.probes[i], m = M(S.method);
  const majors = []; for (let j = p.off; j < p.off + p.n; j++) { const r = d.regions[S.ridx[j]]; const nm = r.major || r.acr; if (majors[majors.length - 1] !== nm) majors.push(nm); }
  const uniq = majors.filter((x, k) => majors.indexOf(x) === k);
  $('#probe-card').innerHTML = `<div class="side-kicker">Selected probe</div>
    <div class="pc-title mono">${p.k}</div>
    <div class="pc-grid"><span>Subject</span><b>${p.subject}</b><span>Lab</span><b>${p.lab}</b><span>Channels</span><b>${p.n}</b>
    <span>Error</span><b>${fmt(rm(m.key, i))} µm</b>
</div>
    <div class="small-note">${uniq.join(' → ')}</div>`;
}

// ───────────────────────── gallery ─────────────────────────
const G = { cells: [] };
const WIN = 6000; // um window per tile
function projXY(arr, j) { return S.proj === 'sag' ? [arr[3 * j], arr[3 * j + 1]] : [arr[3 * j + 2], arr[3 * j + 1]]; }
function darken(hex, f = 0.72) { const c = new THREE.Color(hex); c.multiplyScalar(f); return '#' + c.getHexString(); }
function drawTile(cv, i, arrs) {
  const d = S.data, p = d.probes[i], m = M(S.method);
  const dpr = Math.min(window.devicePixelRatio || 1, 2), W = cv.clientWidth || 100;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(W * dpr); }
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, W);
  let cx = 0, cy = 0; for (let j = p.off; j < p.off + p.n; j++) { const [x, y] = projXY(S.gt, j); cx += x; cy += y; } cx /= p.n; cy /= p.n;
  const sc = W / WIN, X = x => (x - cx) * sc + W / 2, Y = y => (y - cy) * sc + W / 2;
  // 1-mm grid ticks
  g.strokeStyle = '#eef1f7'; g.lineWidth = 1;
  for (let k = -3; k <= 3; k++) { g.beginPath(); g.moveTo(W / 2 + k * 1000 * sc, 0); g.lineTo(W / 2 + k * 1000 * sc, W); g.stroke(); g.beginPath(); g.moveTo(0, W / 2 + k * 1000 * sc); g.lineTo(W, W / 2 + k * 1000 * sc); g.stroke(); }
  const has = arrs && arrs.pooled[3 * p.off] !== MISSING;
  if (has) {
    g.fillStyle = m.color; g.globalAlpha = 0.16;
    for (let j = p.off; j < p.off + p.n; j++) { const [x, y] = projXY(arrs.pooled, j); g.fillRect(X(x) - 1, Y(y) - 1, 2, 2); }
    g.globalAlpha = 1;
    g.strokeStyle = darken(m.color, 0.85); g.lineWidth = 2; g.lineCap = 'round'; g.beginPath();
    for (let j = p.off; j < p.off + p.n; j += 4) { const [x, y] = projXY(arrs.kabsch, j); j === p.off ? g.moveTo(X(x), Y(y)) : g.lineTo(X(x), Y(y)); }
    g.stroke();
  }
  const gtPath = (w, col) => {
    g.lineWidth = w; g.lineCap = 'round';
    for (let j = p.off; j < p.off + p.n - 1; j += 2) {
      const [x0, y0] = projXY(S.gt, j), [x1, y1] = projXY(S.gt, Math.min(j + 2, p.off + p.n - 1));
      g.strokeStyle = col || regionColor(j); g.beginPath(); g.moveTo(X(x0), Y(y0)); g.lineTo(X(x1), Y(y1)); g.stroke();
    }
  };
  gtPath(4.2, '#ffffff'); gtPath(2.4);
  g.font = '600 10px "Google Sans",system-ui,sans-serif'; g.fillStyle = '#22314d';
  g.fillText(has ? `${fmt(rm(m.key, i))} µm` : 'n/a', 5, 12);
  g.font = '9px ui-monospace,Menlo,monospace'; g.fillStyle = '#7a879c'; g.fillText(p.k, 5, W - 5);
}
function galleryOrder() {
  const d = S.data;
  if (S.sort === 'err') return probeOrder(S.method);
  const key = S.sort;
  return d.probes.map((_, i) => i).sort((a, b) => (d.probes[a][key] + d.probes[a].k).localeCompare(d.probes[b][key] + d.probes[b].k));
}
async function updateGallery() {
  const arrs = await loadMethod(S.method);
  const el = $('#gallery');
  if (!G.cells.length) {
    el.innerHTML = S.data.probes.map((p, i) => `<div class="g-cell" data-i="${i}" title="${p.k} · ${p.subject} · ${p.lab}"><canvas></canvas></div>`).join('');
    G.cells = [...el.children];
    el.addEventListener('click', e => { const c = e.target.closest('.g-cell'); if (c) setSel(+c.dataset.i); });
  }
  for (const i of galleryOrder()) el.appendChild(G.cells[i]);
  G.cells.forEach((c, i) => { c.classList.toggle('is-active', i === S.sel); drawTile(c.firstChild, i, arrs); });
}

// ───────────────────────── detail (profiles + per-method errors) ─────────────────────────
function drawProfile(cv, axis, label, i, arrs) {
  const p = S.data.probes[i], m = M(S.method);
  const dpr = Math.min(window.devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const has = arrs && arrs.pooled[3 * p.off] !== MISSING;
  const vals = [];
  for (let j = p.off; j < p.off + p.n; j++) { vals.push(S.gt[3 * j + axis]); if (has) { vals.push(arrs.kabsch[3 * j + axis]); vals.push(arrs.pooled[3 * j + axis]); } }
  vals.sort((a, b) => a - b);
  let lo = vals[Math.floor(vals.length * 0.01)], hi = vals[Math.ceil(vals.length * 0.99) - 1];
  const pad = Math.max(300, (hi - lo) * 0.08); lo -= pad; hi += pad;
  const L = 34, R = 6, T = 20, B = 24, cmax = S.chn[p.off + p.n - 1] || 383;
  // probe depth runs top (high channel) to bottom (channel 0): y = channel, x = coordinate
  const X = v => L + (v - lo) / (hi - lo) * (W - L - R), Y = ch => T + (1 - ch / cmax) * (H - T - B);
  g.strokeStyle = '#e4e8f0'; g.fillStyle = '#7a879c'; g.font = '10px "Google Sans",system-ui,sans-serif'; g.lineWidth = 1;
  const step = (hi - lo) > 4000 ? 2000 : 1000;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) { g.beginPath(); g.moveTo(X(v), T); g.lineTo(X(v), H - B); g.stroke(); g.fillText((v / 1000).toFixed(0), X(v) - 3, H - B + 12); }
  for (const ch of [0, Math.round(cmax / 2), cmax]) g.fillText(String(ch), 2, Y(ch) + 3);
  g.fillStyle = '#22314d'; g.font = '700 11px "Google Sans",system-ui,sans-serif'; g.fillText(label + ' (mm)', L, 13);
  if (has) {
    g.fillStyle = m.color; g.globalAlpha = 0.22;
    for (let j = p.off; j < p.off + p.n; j++) g.fillRect(X(arrs.pooled[3 * j + axis]) - 1.1, Y(S.chn[j]) - 1.1, 2.2, 2.2);
    g.globalAlpha = 1;
    g.strokeStyle = darken(m.color, 0.85); g.lineWidth = 2; g.beginPath();
    for (let j = p.off; j < p.off + p.n; j += 4) { const x = X(arrs.kabsch[3 * j + axis]), y = Y(S.chn[j]); j === p.off ? g.moveTo(x, y) : g.lineTo(x, y); }
    g.stroke();
  }
  for (const [w, white] of [[4.4, true], [2.6, false]]) {
    g.lineWidth = w;
    for (let j = p.off; j < p.off + p.n - 2; j += 2) { g.strokeStyle = white ? '#ffffff' : regionColor(j); g.beginPath(); g.moveTo(X(S.gt[3 * j + axis]), Y(S.chn[j])); g.lineTo(X(S.gt[3 * j + 6 + axis]), Y(S.chn[j + 2])); g.stroke(); }
  }
}
async function updateDetail() {
  const d = S.data, i = S.sel; if (i == null) return;
  const p = d.probes[i], m = M(S.method), arrs = await loadMethod(S.method);
  const rows = d.methods.map(x => [x, rm(x.key, i)]).sort((a, b) => (a[1] ?? 1e9) - (b[1] ?? 1e9));
  const vmax = Math.max(...rows.map(r => r[1] ?? 0));
  $('#detail').innerHTML = `<div class="detail-kicker">${p.subject} · ${p.lab} · ${p.n} channels</div>
    <div class="detail-title mono">${p.k}</div>
    <div class="detail-meta">${p.id}<br><b style="color:${m.color}">${m.label}</b>: ${fmt(rm(m.key, i))} µm</div>
    <div class="profiles"><canvas></canvas><canvas></canvas><canvas></canvas></div>
    <div class="side-kicker" style="margin-top:1rem">All methods (<span style="text-transform:none">µm</span>)</div>
    <div class="cmp-bars">${rows.map(([x, v]) => `<div class="cmp-row${x.key === m.key ? ' is-active' : ''}" data-k="${x.key}">
      <span>${x.label}</span><span class="track"><i style="width:${v == null ? 0 : (100 * v / vmax).toFixed(1)}%;background:${x.color}"></i></span><span class="val">${fmt(v)}</span></div>`).join('')}</div>
    <div style="margin-top:1rem"><a class="btn" href="#explorer" id="to3d" style="font-size:.88rem;padding:.5rem .95rem">Show this probe in 3D</a></div>`;
  const cvs = $('#detail').querySelectorAll('.profiles canvas');
  drawProfile(cvs[0], 0, 'AP', i, arrs); drawProfile(cvs[1], 1, 'DV', i, arrs); drawProfile(cvs[2], 2, 'ML', i, arrs);
  $('#detail .cmp-bars').onclick = e => { const r = e.target.closest('.cmp-row'); if (r) setMethod(r.dataset.k); };
  $('#to3d').onclick = () => { S.scope = 'one'; emit(); focusProbe(i, true); };
}
function updateSummary() {
  $('#summary-strip').innerHTML = S.data.methods.map(m => `<span class="sum-pill"${m.key === S.method ? ' style="border-color:rgba(91,124,255,.55)"' : ''}>
    <i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${m.color};margin-right:.35rem"></i>${m.label}
    <b>${fmt(m.kabsch_mean)} ± ${fmt(m.kabsch_sd)}</b> µm · n=${m.n_probes}</span>`).join('');
}

// ───────────────────────── static sections ─────────────────────────
const FLOW = [
  { id: 'supp11', label: 'Readouts', kicker: 'Readouts', title: 'Frozen encoder, linear or transformer readout',
    desc: 'Chunk predictions are pooled per channel, then the probe geometry is fitted (Kabsch).' },
  { id: 'fig5', label: 'Pretraining', kicker: 'Pretraining', title: 'Coordinates emerge during pretraining',
    desc: 'Linear readouts at 0, 5k and 115k pretraining steps.' },
  { id: 'fig6', label: 'Scaling', kicker: 'Scaling', title: 'Data, decoder size and compute',
    desc: 'Base and large encoders, four objectives.' },
  { id: 'supp_datasets', label: 'Datasets', kicker: 'Data', title: 'Four recording datasets',
    desc: 'IBL, Allen, NeuroNexus and macaque.' },
];
function initFlow() {
  const el = $('#flow-thumbs');
  el.innerHTML = FLOW.map((f, k) => `<button class="flow-thumb${k ? '' : ' is-active'}" data-k="${k}"><div class="flow-thumb-label">${f.label}</div><img src="figures/${f.id}_sm.webp" alt="" loading="lazy"></button>`).join('');
  const show = k => {
    const f = FLOW[k];
    $('#flow-img').src = `figures/${f.id}.webp`; $('#flow-kicker').textContent = f.kicker; $('#flow-title').textContent = f.title; $('#flow-description').textContent = f.desc;
    el.querySelectorAll('.flow-thumb').forEach(b => b.classList.toggle('is-active', +b.dataset.k === k));
  };
  el.addEventListener('click', e => { const b = e.target.closest('.flow-thumb'); if (b) show(+b.dataset.k); });
  show(0);
}
function perfBars(id, rows, max, digits = 0) {
  $(id).innerHTML = rows.map(([lab, v, cls]) => `<div class="perf-row"><div class="perf-label">${lab}</div>
    <div class="perf-track"><div class="perf-fill ${cls || ''}" style="--w:${(100 * v / max).toFixed(1)}%"></div></div>
    <div class="perf-value">${digits ? v.toFixed(digits) : Math.round(v).toLocaleString('en-US')}${digits ? '' : ' µm'}</div></div>`).join('');
}
function initPerf() {
  perfBars('#perf-lfp', [['LFPformer (TF)', 1011.93], ['AnyNet', 1483.82, 'alt'], ['ViT', 1576.47, 'alt2'], ['kNN (PSD)', 1766.87, 'alt2']], 2100);
  perfBars('#perf-manip', [['LFPformer (lin.) + manip.', 709.55], ['Manip. alone', 790.36, 'alt'], ['LFPformer (TF) + manip.', 996.95, 'alt2']], 2100);
  perfBars('#perf-ibl', [['LFPformer', 0.0808], ['AnyNet', 0.0353, 'alt'], ['ViT', 0.0224, 'alt2']], 0.1, 3);
  perfBars('#perf-nn', [['LFPformer', 0.634], ['AnyNet', 0.550, 'alt'], ['ViT', 0.527, 'alt2']], 0.8, 2);
}
async function initFigures() {
  const figs = await fetch('data/figures.json').then(r => r.json());
  const el = $('#fig-chips');
  const short = { fig3: 'Fig. 3', fig4: 'Fig. 4', fig5: 'Fig. 5', fig6: 'Fig. 6', supp_datasets: 'Supp. datasets', supp7a: 'Supp. 7a', supp7b: 'Supp. 7b', supp11: 'Supp. 11' };
  el.innerHTML = `<div class="chip-group"><span class="chip-group-label">Main</span>${figs.filter(f => f.id.startsWith('fig')).map(f => `<button class="chip" data-id="${f.id}">${short[f.id]}</button>`).join('')}</div>
    <div class="chip-group"><span class="chip-group-label">Supplementary</span>${figs.filter(f => f.id.startsWith('supp')).map(f => `<button class="chip" data-id="${f.id}">${short[f.id]}</button>`).join('')}</div>`;
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const show = id => {
    const f = figs.find(x => x.id === id);
    $('#fig-stage').innerHTML = `<div class="fig-head"><h3>${f.title}</h3><div class="fig-actions"><a href="figures/${f.id}.pdf" target="_blank" rel="noopener">PDF</a><a href="figures/${f.id}.webp" target="_blank" rel="noopener">Full size</a></div></div>
      <img src="figures/${f.id}.webp" alt="${esc(f.title)}" width="${f.w}" height="${f.h}" style="height:auto" loading="lazy">
      <details class="caption"><summary>Caption</summary><p>${esc(f.caption)}</p></details>`;
    el.querySelectorAll('.chip').forEach(b => b.classList.toggle('is-active', b.dataset.id === id));
  };
  el.addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) show(b.dataset.id); });
  show('fig3');
}
function initBib() {
  $('#copy-bib').onclick = async () => { try { await navigator.clipboard.writeText($('#bib').textContent); $('#copy-bib').textContent = 'Copied'; setTimeout(() => $('#copy-bib').textContent = 'Copy', 1500); } catch (e) { } };
}

// ───────────────────────── boot ─────────────────────────
initFlow(); initPerf(); initFigures(); initBib();
import('./chronic.js').then(m => m.initChronic()).catch(e => console.error('chronic', e));
try {
  await loadData();
  buildChips($('#side-chips'), true); buildChips($('#rec-chips'), false);
  segControl($('#seg-mode'), 'mode', () => S.mode, v => { S.mode = v; emit(); });
  segControl($('#seg-scope'), 'scope', () => S.scope, v => { S.scope = v; emit(); if (v !== 'all' && S.sel != null) focusProbe(S.sel, true); if (v === 'all') resetView(); });
  segControl($('#seg-sort'), 'sort', () => S.sort, v => { S.sort = v; emit(); });
  segControl($('#seg-proj'), 'proj', () => S.proj, v => { S.proj = v; emit(); });
  $('#probe-search').addEventListener('input', updateList);
  $('#probe-list').addEventListener('click', e => { const r = e.target.closest('.probe-row'); if (r) { setSel(+r.dataset.i); if (S.scope !== 'all') focusProbe(+r.dataset.i, true); } });
  S.listeners.push(() => { updateViewer(); updateList(); updateCard(); updateGallery(); updateDetail(); updateSummary(); });
  updateList(); updateCard(); updateGallery(); updateDetail(); updateSummary();
  await initViewer();
  updateViewer();
} catch (err) {
  console.error(err);
  const l = $('#viewer-loading'); if (l) l.textContent = 'Could not load the 3D data (' + err.message + ').';
}
