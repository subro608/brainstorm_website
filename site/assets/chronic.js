// Chronic zero-shot recordings: two players on separate clocks.
//   full : 3D predicted probe + regions along the probe + predictions-vs-time over the whole session
//   maze : mouse x-y on the linear maze + regions along the probe + predictions-vs-time and track position, maze only
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const $ = (s, r = document) => r.querySelector(s);
const MESH_NV = 39828, MESH_NF = 79364;
const ROWS = 192, OUTSIDE = 255, TRAIL = 10, CHUNK = 3;
const BLOCK_COLORS = { pre: '#8DA0CB', maze: '#FC8D62', post: '#66C2A5', other: '#BBBBBB' };
const FONT = '"Google Sans",system-ui,sans-serif';
const toV = (ap, dv, ml, out, o) => { out[o] = (ml - 5700) / 1000; out[o + 1] = -(dv - 4000) / 1000; out[o + 2] = (ap - 6600) / 1000; };
const fmtT = s => { s = Math.max(0, Math.floor(s)); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, x = s % 60; return h ? `${h}:${String(m).padStart(2, '0')}:${String(x).padStart(2, '0')}` : `${m}:${String(x).padStart(2, '0')}`; };
const hex = h => { const x = parseInt(h.slice(1), 16); return [(x >> 16) & 255, (x >> 8) & 255, x & 255]; };

const G = { meta: null, si: 0, sess: {} };
const colorOf = k => (k === OUTSIDE ? '#E9ECF2' : G.meta.palette[k].color);
const blockAt = (s, t) => s.blocks.find(b => t >= b.t0 && t < b.t1) || s.blocks[s.blocks.length - 1];
const mazeRange = s => { const b = s.blocks.find(x => x.kind === 'maze'), ti = s.track_info || {};
  return [Math.max(b.t0, ti.t_first ?? b.t0), Math.min(b.t1, ti.t_last ?? b.t1)]; };

async function loadSession(i) {
  if (G.sess[i]) return G.sess[i];
  const s = G.meta.sessions[i];
  const [lab, probe, track] = await Promise.all(['labels', 'probe', 'track'].map(k =>
    fetch(`data/chronic/${i}_${k}.bin`).then(r => r.arrayBuffer())));
  const d = { s, labels: new Uint8Array(lab), probe: new Int16Array(probe), track: new Float32Array(track) };
  // whole-session region image: one pixel column per 3-s chunk, rows surface -> tip
  const n = s.n_chunks, cv = document.createElement('canvas'); cv.width = n; cv.height = ROWS;
  const g = cv.getContext('2d'), img = g.createImageData(n, ROWS), rgb = G.meta.palette.map(p => hex(p.color)), out = hex('#E9ECF2');
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < n; c++) {
    const k = d.labels[r * n + c], v = k === OUTSIDE ? out : rgb[k], o = 4 * (r * n + c);
    img.data[o] = v[0]; img.data[o + 1] = v[1]; img.data[o + 2] = v[2]; img.data[o + 3] = 255;
  }
  g.putImageData(img, 0, 0); d.tl = cv;
  const tr = d.track, m = tr.length / 3; let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (let i2 = 0; i2 < m; i2++) { x0 = Math.min(x0, tr[3 * i2 + 1]); x1 = Math.max(x1, tr[3 * i2 + 1]); y0 = Math.min(y0, tr[3 * i2 + 2]); y1 = Math.max(y1, tr[3 * i2 + 2]); }
  d.box = [x0, x1, y0, y1];
  return (G.sess[i] = d);
}
function trackIndex(tr, t) { const n = tr.length / 3; let lo = 0, hi = n - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (tr[3 * m] < t) lo = m + 1; else hi = m; } return lo; }

function canvas2d(cv) {
  const dpr = Math.min(devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H); return [g, W, H];
}

// regions along the probe (surface at top, tip at bottom)
function drawRegions(el, d, c) {
  const n = d.s.n_chunks, runs = [];
  for (let r = 0; r < ROWS; r++) { const k = d.labels[r * n + c]; if (runs.length && runs[runs.length - 1].k === k) runs[runs.length - 1].len++; else runs.push({ k, len: 1 }); }
  el.innerHTML = runs.map(u => `<div class="chr-run" style="flex:${u.len};background:${colorOf(u.k)}">${u.len >= 9 && u.k !== OUTSIDE ? `<span>${G.meta.palette[u.k].acr}</span>` : ''}</div>`).join('');
}

// ── 3D (full player only) ──
const V = {};
async function initViewer() {
  const host = $('#chr-viewer');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); host.prepend(renderer.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  Object.assign(V, { host, renderer, scene, camera, controls, dirty: true });
  controls.addEventListener('change', () => { V.dirty = true; });
  scene.add(new THREE.HemisphereLight(0xffffff, 0xdfe6f3, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, 0.9); dl.position.set(-6, 10, -8); scene.add(dl);
  const buf = await fetch('data/brain_mesh.bin').then(r => r.arrayBuffer());
  const vq = new Uint16Array(buf, 0, MESH_NV * 3), fq = new Uint16Array(buf, MESH_NV * 6, MESH_NF * 3);
  const pos = new Float32Array(MESH_NV * 3);
  for (let i = 0; i < MESH_NV; i++) toV(vq[3 * i], vq[3 * i + 1], vq[3 * i + 2], pos, 3 * i);
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(new Uint16Array(fq), 1)); geo.computeVertexNormals();
  scene.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color: 0xbfc9de, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide, shininess: 12 })));
  V.matProbe = new LineMaterial({ linewidth: 7, vertexColors: true, depthWrite: false, transparent: true });
  V.matTrail = new LineMaterial({ linewidth: 1.4, color: 0x6b7a94, transparent: true, opacity: 0.3, depthWrite: false });
  V.probe = new LineSegments2(new LineSegmentsGeometry(), V.matProbe); V.probe.renderOrder = 5; scene.add(V.probe);
  V.trail = new LineSegments2(new LineSegmentsGeometry(), V.matTrail); V.trail.renderOrder = 4; scene.add(V.trail);
  const resize = () => { const w = host.clientWidth, h = host.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); V.matProbe.resolution.set(w, h); V.matTrail.resolution.set(w, h); V.dirty = true; };
  new ResizeObserver(resize).observe(host); resize();
  renderer.setAnimationLoop(() => { controls.update(); if (V.dirty) { renderer.render(scene, camera); V.dirty = false; } });
  host.querySelector('.loading')?.remove();
}
function frameSession(d) {
  if (!V.scene) return;
  const n = d.s.n_chunks, p = d.probe, acc = [[], [], []];
  for (let c = 0; c < n; c += 7) for (let a = 0; a < 3; a++) acc[a].push((p[6 * c + a] + p[6 * c + 3 + a]) / 2);
  const med = acc.map(v => v.sort((x, y) => x - y)[v.length >> 1]), t = [0, 0, 0]; toV(med[0], med[1], med[2], t, 0);
  const target = new THREE.Vector3(...t), f = Math.max(1, 1.2 / (V.camera.aspect || 1.2));
  V.controls.target.copy(target); V.camera.position.copy(target).add(new THREE.Vector3(-6.2 * f, 3.6 * f, -4.6 * f)); V.dirty = true;
}
const probeAt = (d, c) => { const p = d.probe, tip = [p[6 * c], p[6 * c + 1], p[6 * c + 2]], base = [p[6 * c + 3], p[6 * c + 4], p[6 * c + 5]];
  return r => [0, 1, 2].map(a => base[a] + (tip[a] - base[a]) * r / (ROWS - 1)); };
function updateProbe(d, c) {
  if (!V.scene) return;
  const at = probeAt(d, c), n = d.s.n_chunks, pos = new Float32Array((ROWS - 1) * 6), col = new Float32Array((ROWS - 1) * 6), cc = new THREE.Color();
  for (let r = 0; r < ROWS - 1; r++) { const a = at(r), b = at(r + 1); toV(a[0], a[1], a[2], pos, 6 * r); toV(b[0], b[1], b[2], pos, 6 * r + 3);
    cc.set(colorOf(d.labels[r * n + c])); cc.toArray(col, 6 * r); cc.toArray(col, 6 * r + 3); }
  const g = new LineSegmentsGeometry(); g.setPositions(pos); g.setColors(col); V.probe.geometry.dispose(); V.probe.geometry = g;
  const tp = [];
  for (let k = Math.max(0, c - TRAIL); k < c; k++) { const f = probeAt(d, k), a = f(0), b = f(ROWS - 1), o = [0, 0, 0, 0, 0, 0]; toV(a[0], a[1], a[2], o, 0); toV(b[0], b[1], b[2], o, 3); tp.push(...o); }
  const tg = new LineSegmentsGeometry(); if (tp.length) tg.setPositions(tp); V.trail.geometry.dispose(); V.trail.geometry = tg; V.trail.visible = tp.length > 0;
  V.dirty = true;
}

// ── maze x-y ──
function drawMaze(cv, d, t) {
  const [g, W, H] = canvas2d(cv), tr = d.track, n = tr.length / 3, [x0, x1, y0, y1] = d.box, pad = 16;
  const sc = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad - 12) / (y1 - y0));
  const ox = (W - (x1 - x0) * sc) / 2, oy = (H - 12 - (y1 - y0) * sc) / 2, X = x => ox + (x - x0) * sc, Y = y => oy + (y1 - y) * sc;
  g.fillStyle = 'rgba(120,134,160,0.10)';
  for (let i = 0; i < n; i += 3) g.fillRect(X(tr[3 * i + 1]) - 1, Y(tr[3 * i + 2]) - 1, 2, 2);
  g.fillStyle = '#7a879c'; g.font = `10px ${FONT}`; g.fillText(`${Math.round(x1 - x0)} cm`, W - pad - 34, H - 2);
  const i = trackIndex(tr, t), tail = Math.max(0, i - 60);
  g.lineCap = 'round';
  for (let j = tail; j < i; j++) { g.strokeStyle = `rgba(208,28,139,${0.15 + 0.7 * (j - tail) / Math.max(1, i - tail)})`; g.lineWidth = 3;
    g.beginPath(); g.moveTo(X(tr[3 * j + 1]), Y(tr[3 * j + 2])); g.lineTo(X(tr[3 * j + 4]), Y(tr[3 * j + 5])); g.stroke(); }
  g.fillStyle = '#D01C8B'; g.beginPath(); g.arc(X(tr[3 * i + 1]), Y(tr[3 * i + 2]), 7, 0, 7); g.fill(); g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke();
}

// ── predictions-vs-time timelines ──
function drawTimeline(cv, d, a, b, t, withBlocks, withTrack) {
  const [g, W, H] = canvas2d(cv), X = u => (u - a) / (b - a) * W;
  let top = 0;
  if (withBlocks) {
    for (const bl of d.s.blocks) { const x0 = Math.max(0, X(bl.t0)), x1 = Math.min(W, X(bl.t1)); if (x1 <= x0) continue;
      g.fillStyle = BLOCK_COLORS[bl.kind]; g.fillRect(x0, 0, x1 - x0, 13);
      g.fillStyle = '#22314d'; g.font = `600 11px ${FONT}`; if (x1 - x0 > 70) g.fillText(bl.label, x0 + 6, 10); }
    top = 16;
  }
  const axis = 14, trackH = withTrack ? 52 : 0, mapH = H - top - axis - trackH - (withTrack ? 6 : 0);
  // region map, cropped to [a, b]
  const c0 = a / CHUNK, c1 = b / CHUNK;
  g.imageSmoothingEnabled = false; g.drawImage(d.tl, c0, 0, c1 - c0, ROWS, 0, top, W, mapH);
  g.fillStyle = '#7a879c'; g.font = `10px ${FONT}`;
  g.save(); g.translate(0, 0); g.fillText('surface', 3, top + 10); g.fillText('tip', 3, top + mapH - 3); g.restore();
  if (withTrack) {
    // mouse position along the track over time
    const y0 = top + mapH + 6, tr = d.track, n = tr.length / 3, [bx0, bx1] = d.box, Y = x => y0 + trackH - (x - bx0) / (bx1 - bx0) * trackH;
    g.fillStyle = '#f6f8fc'; g.fillRect(0, y0, W, trackH);
    g.strokeStyle = '#D01C8B'; g.lineWidth = 1.2; g.beginPath();
    const step = Math.max(1, Math.floor(n / (W * 2)));
    for (let i = 0; i < n; i += step) { const x = X(tr[3 * i]); i ? g.lineTo(x, Y(tr[3 * i + 1])) : g.moveTo(x, Y(tr[3 * i + 1])); }
    g.stroke(); g.fillStyle = 'rgba(255,255,255,.85)'; g.fillRect(2, y0 + 1, 96, 12); g.fillStyle = '#5d6a7b'; g.fillText('position on track', 5, y0 + 10);
  }
  // time axis
  const span = b - a, tick = span > 7200 ? 3600 : span > 1800 ? 600 : span > 600 ? 300 : 60;
  g.fillStyle = '#7a879c';
  const first = withBlocks ? Math.ceil(a / tick) * tick : a;   // maze axis counts from maze entry
  for (let u = first; u <= b; u += tick) { const x = X(u); g.fillRect(x, H - axis, 1, 3); g.fillText(fmtT(withBlocks ? u : u - a), Math.min(x + 2, W - 30), H - 2); }
  const x = X(t); g.fillStyle = '#111827'; g.fillRect(x - 1, 0, 2, H - axis);
}

// ── a player: own clock, controls and panels ──
class Player {
  constructor(o) { Object.assign(this, o, { t: 0, playing: false, chunk: -1 }); }
  range(d) { return this.kind === 'full' ? [0, d.s.duration_s] : mazeRange(d.s); }
  attach() {
    const r = this.root;
    r.querySelector('.chr-play').onclick = () => this.setPlaying(!this.playing);
    r.querySelector('.chr-speed').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; this.speed = +b.dataset.s;
      r.querySelectorAll('.chr-speed button').forEach(x => x.classList.toggle('is-active', x === b)); });
    const tl = r.querySelector('.chr-tl'); let drag = false;
    const seek = e => { const d = G.sess[G.si]; if (!d) return; const [a, b] = this.range(d), q = tl.getBoundingClientRect();
      this.t = a + Math.max(0, Math.min(1, (e.clientX - q.left) / q.width)) * (b - a); this.render(); };
    tl.addEventListener('pointerdown', e => { drag = true; tl.setPointerCapture(e.pointerId); seek(e); });
    tl.addEventListener('pointermove', e => { if (drag) seek(e); });
    tl.addEventListener('pointerup', () => { drag = false; });
    new ResizeObserver(() => this.render()).observe(tl);
  }
  setPlaying(p) { this.playing = p; this.root.querySelector('.chr-play').textContent = p ? 'Pause' : 'Play'; }
  reset(d) { const [a] = this.range(d); this.t = this.kind === 'full' ? 0 : a; this.chunk = -1; if (this.kind === 'full') frameSession(d); this.render(true); }
  advance(dt) { const d = G.sess[G.si]; if (!this.playing || !d) return; const [a, b] = this.range(d); this.t += dt * this.speed; if (this.t >= b) this.t = a; this.render(); }
  render(force) {
    const d = G.sess[G.si]; if (!d) return;
    const [a, b] = this.range(d), c = Math.min(d.s.n_chunks - 1, Math.floor(this.t / CHUNK));
    if (c !== this.chunk || force) { this.chunk = c; drawRegions(this.root.querySelector('.chr-regions'), d, c); if (this.kind === 'full') updateProbe(d, c); }
    const blk = blockAt(d.s, this.t);
    this.root.querySelector('.chr-clock').innerHTML = this.kind === 'full'
      ? `<b>${fmtT(this.t)}</b> / ${fmtT(d.s.duration_s)} · <span style="color:${BLOCK_COLORS[blk.kind]};font-weight:800">${blk.label}</span>`
      : `<b>${fmtT(this.t - a)}</b> / ${fmtT(b - a)} in the maze`;
    if (this.kind === 'maze') drawMaze(this.root.querySelector('.chr-maze'), d, this.t);
    drawTimeline(this.root.querySelector('.chr-tl'), d, a, b, this.t, this.kind === 'full', this.kind === 'maze');
  }
}

export async function initChronic() {
  G.meta = await fetch('data/chronic/sessions.json').then(r => r.json());
  const players = [new Player({ kind: 'full', root: $('#chr-full'), speed: 60 }), new Player({ kind: 'maze', root: $('#chr-mazep'), speed: 10 })];
  players.forEach(p => p.attach());
  $('#chr-chips').innerHTML = `<div class="chip-group"><span class="chip-group-label">Session</span>${G.meta.sessions.map(s =>
    `<button class="chip" data-i="${s.i}">${s.label}</button>`).join('')}</div>`;
  const select = async i => { G.si = i; const d = await loadSession(i); if (G.si !== i) return; players.forEach(p => p.reset(d));
    $('#chr-chips').querySelectorAll('.chip').forEach(b => b.classList.toggle('is-active', +b.dataset.i === i)); };
  $('#chr-chips').addEventListener('click', e => { const b = e.target.closest('.chip'); if (b) select(+b.dataset.i); });
  await select(0);
  const auto = !/[?&]rotate=0/.test(location.search);
  // each player starts when it first scrolls into view; the 3D view is created lazily
  players.forEach(p => {
    const io = new IntersectionObserver(async ents => {
      if (!ents.some(x => x.isIntersecting)) return; io.disconnect();
      if (p.kind === 'full') { await initViewer(); frameSession(G.sess[G.si]); p.render(true); }
      if (auto) p.setPlaying(true);
    }, { threshold: 0.3 });
    io.observe(p.root);
  });
  let last = performance.now();
  const tick = now => { const dt = Math.min(0.1, (now - last) / 1000); last = now; players.forEach(p => p.advance(dt)); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}
