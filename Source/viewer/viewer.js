'use strict';
/*
 * psynapse net viewer front-end.
 *
 * Everything is drawn onto ONE canvas so that the firing-rate histogram,
 * the spike raster, and the connection matrix share a single vertical
 * neuron-coordinate: row y === neuron ID across all three panels.
 *
 * Horizontal layout (left -> right):
 *     [ histogram ] [ raster (time ->, newest at right) ] [popbar] [ matrix ]
 *
 * Matrix orientation: horizontal axis = upstream/presynaptic i (columns),
 * vertical axis = downstream/postsynaptic j (rows). Cell (col i, row j) is
 * the weight of synapse i -> j. Above the matrix sit a stimulation strip
 * and an upstream population-colour bar; a downstream population bar sits
 * just left of the matrix, aligned with the raster rows.
 */

// ---- Layout configuration (CSS pixels) -------------------------------------
const CFG = {
  rowH: 8,          // px per neuron (shared vertical scale for all panels)
  matrixCell: 8,    // px per matrix column (== rowH keeps rows aligned)
  rasterColW: 2,    // px per time step
  rasterWidth: 240, // time steps shown
  histW: 80,        // firing-rate histogram width
  popBar: 10,       // downstream population bar thickness
  popBarTopH: 8,    // upstream population bar thickness
  stimH: 12,        // stimulation strip height
  gap: 12,          // gap between panels
  labelH: 16,       // label band top and bottom
  ringLen: 8,       // weight-history depth (for Δ views)
};

const REGION_COLORS = ['#4cc2ff', '#7ee787', '#ff7b72', '#d2a8ff', '#ffa657', '#79c0ff'];

// ---- State -----------------------------------------------------------------
let meta = null;                       // {numNeurons, regionIDs, regionNames}
let rasterCols = [];                   // Uint8Array(N) per step, newest last
let rateWindow = [];                   // recent firing columns for rolling rate
let latestInput = null;                // current external-input vector
const base = {};                       // {source: matrix} snapshot at load
const ring = { weights: [], modWeights: [] };  // recent matrices for Δ views
const latest = {};                     // {source: current matrix}
let colorScale = 1;                    // smoothed colour normalization

let viewSource = 'weights';
let deltaMode = 'current';
let playing = false;
let timer = null;
let hoverText = '';

const $ = id => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');

// ---- Geometry --------------------------------------------------------------
// Computed from CFG + neuron count; all panels reference these.
let L = null;
function layout() {
  const N = meta.numNeurons;
  const rowsY = CFG.labelH + CFG.stimH + 2 + CFG.popBarTopH + 4;
  const histX = 0;
  const rasterX = histX + CFG.histW + CFG.gap;
  const rasterW = CFG.rasterWidth * CFG.rasterColW;
  const popLeftX = rasterX + rasterW + CFG.gap;
  const matrixX = popLeftX + CFG.popBar + 2;
  const matrixW = N * CFG.matrixCell;
  const gridH = N * CFG.rowH;
  L = {
    N, rowsY, histX, rasterX, rasterW, popLeftX, matrixX, matrixW, gridH,
    width: matrixX + matrixW + 2,
    height: rowsY + gridH + CFG.labelH,
    stimY: CFG.labelH,
    popTopY: CFG.labelH + CFG.stimH + 2,
  };
}
const rowY = j => L.rowsY + j * CFG.rowH;
const colX = i => L.matrixX + i * CFG.matrixCell;
function regionColor(neuron) {
  return REGION_COLORS[meta.regionIDs[neuron] % REGION_COLORS.length];
}

// ---- Small helpers ---------------------------------------------------------
function hexRGB(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}
// Diverging map: negative -> blue, 0 -> dark, positive -> orange. v in [-1,1].
function diverging(v) {
  if (v > 0) { v = Math.min(v, 1); return [20 + 227 * v, 26 + 93 * v, 34 - 34 * v]; }
  if (v < 0) { v = Math.min(-v, 1); return [20 + 27 * v, 26 + 103 * v, 34 + 213 * v]; }
  return [20, 26, 34];
}
// Build an offscreen canvas (cols x rows) from a per-pixel colour function.
function pixelCanvas(cols, rows, colorFn) {
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  const octx = off.getContext('2d');
  const img = octx.createImageData(cols, rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const c = colorFn(x, y);
      const idx = (y * cols + x) * 4;
      img.data[idx] = c[0]; img.data[idx + 1] = c[1]; img.data[idx + 2] = c[2]; img.data[idx + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  return off;
}

// ---- Which matrix to display (source + delta) ------------------------------
function subtract(a, b) {
  const N = a.length, out = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(N);
    for (let j = 0; j < N; j++) row[j] = a[i][j] - b[i][j];
    out[i] = row;
  }
  return out;
}
function shownMatrix() {
  const cur = latest[viewSource];
  if (!cur) return null;
  if (deltaMode === 'current') return cur;
  const hist = ring[viewSource] || [];
  let ref;
  if (deltaMode === 'dall') ref = base[viewSource];
  else if (deltaMode === 'd1') ref = hist[hist.length - 2];
  else if (deltaMode === 'd5') ref = hist[Math.max(0, hist.length - 6)];
  if (!ref) return null;
  return subtract(cur, ref);
}

// ---- Drawing ---------------------------------------------------------------
function draw() {
  if (!meta) return;
  layout();
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = L.width + 'px';
  canvas.style.height = L.height + 'px';
  canvas.width = Math.round(L.width * dpr);
  canvas.height = Math.round(L.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, L.width, L.height);
  ctx.textBaseline = 'alphabetic';
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';

  drawLabels();
  drawHistogram();
  drawRaster();
  drawPopBars();
  drawStimStrip();
  drawMatrix();
}

function drawLabels() {
  ctx.fillStyle = '#8b949e';
  ctx.fillText('rate', L.histX, CFG.labelH - 4);
  ctx.fillText('spike raster  (time →)', L.rasterX, CFG.labelH - 4);
  ctx.fillText('synapse matrix  (upstream i →, downstream j ↓)', L.matrixX, CFG.labelH - 4);
}

function drawHistogram() {
  const N = meta.numNeurons;
  if (rateWindow.length === 0) return;
  const rates = new Float32Array(N);
  for (const col of rateWindow) for (let j = 0; j < N; j++) rates[j] += col[j];
  for (let j = 0; j < N; j++) rates[j] /= rateWindow.length;
  for (let j = 0; j < N; j++) {
    ctx.fillStyle = regionColor(j);
    const w = Math.max(0.5, rates[j] * CFG.histW);
    ctx.fillRect(L.histX + CFG.histW - w, rowY(j), w, Math.max(1, CFG.rowH - 1));
  }
}

function drawRaster() {
  const N = meta.numNeurons;
  const cols = CFG.rasterWidth;
  const startCol = cols - rasterCols.length;  // left-pad when history < width
  const off = pixelCanvas(cols, N, (x, y) => {
    const ci = x - startCol;
    if (ci < 0 || ci >= rasterCols.length) return [12, 15, 20];
    if (rasterCols[ci][y]) return hexRGB(regionColor(y));
    return [12, 15, 20];
  });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, L.rasterX, L.rowsY, L.rasterW, L.gridH);
}

function drawPopBars() {
  const N = meta.numNeurons;
  // Downstream population bar (vertical), left of matrix, aligned to rows.
  for (let j = 0; j < N; j++) {
    ctx.fillStyle = regionColor(j);
    ctx.fillRect(L.popLeftX, rowY(j), CFG.popBar, CFG.rowH);
  }
  // Upstream population bar (horizontal), above matrix, aligned to columns.
  for (let i = 0; i < N; i++) {
    ctx.fillStyle = regionColor(i);
    ctx.fillRect(colX(i), L.popTopY, CFG.matrixCell, CFG.popBarTopH);
  }
}

function drawStimStrip() {
  const N = meta.numNeurons;
  ctx.fillStyle = '#141a22';
  ctx.fillRect(L.matrixX, L.stimY, L.matrixW, CFG.stimH);
  if (!latestInput) return;
  let maxIn = 0;
  for (let i = 0; i < N; i++) maxIn = Math.max(maxIn, Math.abs(latestInput[i]));
  if (maxIn === 0) return;
  for (let i = 0; i < N; i++) {
    const v = Math.abs(latestInput[i]) / maxIn;
    if (v <= 0) continue;
    ctx.fillStyle = 'rgba(255,166,87,' + (0.25 + 0.75 * v).toFixed(3) + ')';
    ctx.fillRect(colX(i), L.stimY, CFG.matrixCell, CFG.stimH);
  }
}

function drawMatrix() {
  const M = shownMatrix();
  const N = meta.numNeurons;
  if (!M) {
    ctx.fillStyle = '#8b949e';
    ctx.fillText('(no data for this view yet)', L.matrixX, L.rowsY + 14);
    return;
  }
  // Normalize by a smoothed max-abs so the colours don't flicker.
  let maxAbs = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) maxAbs = Math.max(maxAbs, Math.abs(M[i][j]));
  colorScale = Math.max(colorScale * 0.9, maxAbs, 1e-6);
  // Pixel (px=i upstream=column, py=j downstream=row) shows synapse i -> j.
  const off = pixelCanvas(N, N, (px, py) => diverging(M[px][py] / colorScale));
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, L.matrixX, L.rowsY, L.matrixW, L.gridH);
}

// ---- Data flow -------------------------------------------------------------
async function loadInit() {
  meta = await (await fetch('/api/init')).json();
  rasterCols = []; rateWindow = []; latestInput = null;
  ring.weights = []; ring.modWeights = [];
  base.weights = meta.weights; base.modWeights = meta.modWeights;
  latest.weights = meta.weights; latest.modWeights = meta.modWeights;
  latest.plasticityWeights = null; base.plasticityWeights = null;
  colorScale = 1;
  $('stepCount').textContent = meta.step;
  draw();
}

function pushMatrix(source, M) {
  latest[source] = M;
  const r = ring[source];
  r.push(M);
  if (r.length > CFG.ringLen) r.shift();
}

async function doStep(n) {
  const res = await (await fetch('/api/step?n=' + n, { method: 'POST' })).json();
  for (const col of res.firingColumns) {
    const arr = Uint8Array.from(col);
    rasterCols.push(arr);
    if (rasterCols.length > CFG.rasterWidth) rasterCols.shift();
    rateWindow.push(arr);
    if (rateWindow.length > 60) rateWindow.shift();
  }
  latestInput = res.inputColumns[res.inputColumns.length - 1];
  pushMatrix('weights', res.weights);
  pushMatrix('modWeights', res.modWeights);
  $('stepCount').textContent = res.step;
  const last = rasterCols[rasterCols.length - 1];
  const frac = last.reduce((a, b) => a + b, 0) / last.length;
  $('rate').textContent = (frac * 100).toFixed(1) + '%';
  draw();
}

// ---- Transport + view controls ---------------------------------------------
function tick() { doStep(parseInt($('burst').value, 10)); }

function setPlaying(on) {
  playing = on;
  $('playBtn').textContent = on ? '❚❚ Pause' : '▶ Play';
  $('playBtn').classList.toggle('primary', !on);
  if (timer) { clearInterval(timer); timer = null; }
  if (on) timer = setInterval(tick, 1000 / parseInt($('speed').value, 10));
}

$('playBtn').onclick = () => setPlaying(!playing);
$('stepBtn').onclick = () => { setPlaying(false); tick(); };
$('resetBtn').onclick = async () => { setPlaying(false); await loadInit(); };
$('speed').oninput = () => { if (playing) setPlaying(true); };
$('burst').oninput = () => { $('burstVal').textContent = $('burst').value; };

function wireSegGroup(groupId, attr, apply) {
  const group = $(groupId);
  group.querySelectorAll('.seg').forEach(btn => {
    btn.onclick = () => {
      if (btn.disabled) return;
      group.querySelectorAll('.seg').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      apply(btn.dataset[attr]);
      colorScale = 1;
      draw();
    };
  });
}
wireSegGroup('sourceGroup', 'source', v => { viewSource = v; });
wireSegGroup('deltaGroup', 'delta', v => { deltaMode = v; });

// ---- Tab switching ---------------------------------------------------------
document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const panel = document.getElementById('tab-' + tab.dataset.tab);
    if (panel) panel.classList.add('active');
    if (tab.dataset.tab === 'net' && meta) draw();
  };
});

// ---- Hover read-out --------------------------------------------------------
canvas.onmousemove = ev => {
  if (!meta) return;
  const r = canvas.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  const j = Math.floor((y - L.rowsY) / CFG.rowH);
  if (j < 0 || j >= meta.numNeurons) { setHover(''); return; }
  if (x >= L.matrixX && x < L.matrixX + L.matrixW) {
    const i = Math.floor((x - L.matrixX) / CFG.matrixCell);
    const M = shownMatrix();
    const val = M ? M[i][j] : NaN;
    setHover('syn ' + i + '→' + j + ' = ' + (isNaN(val) ? '—' : val.toFixed(3)));
  } else {
    setHover('neuron ' + j + ' (' + meta.regionNames[meta.regionIDs[j]] + ')');
  }
};
canvas.onmouseleave = () => setHover('');
function setHover(t) { if (t !== hoverText) { hoverText = t; $('hover').textContent = t || ' '; } }

window.addEventListener('resize', () => { if (meta) draw(); });
loadInit();
