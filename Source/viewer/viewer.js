'use strict';
/*
 * psynapse net viewer front-end.
 *
 * The main view draws onto ONE canvas so the firing-rate histogram, spike
 * raster, and connection matrix share a single vertical neuron-coordinate
 * (row y === neuron ID). Layout, left -> right:
 *     [ histogram ] [ raster (time ->) ] [popbar] [ matrix ] [ colour scale ]
 *
 * Matrix orientation: horizontal axis = upstream/presynaptic i (columns),
 * vertical axis = downstream/postsynaptic j (rows); cell (i, j) = synapse
 * i -> j. Clicking a row selects that neuron and opens the detail panel.
 */

// ---- Layout configuration (CSS pixels) -------------------------------------
const CFG = {
  rowH: 8, matrixCell: 8, rasterColW: 2, rasterWidth: 240,
  histW: 80, popBar: 10, popBarTopH: 8, stimH: 12,
  gap: 12, labelH: 16, ringLen: 8,
  scaleBarW: 14, scaleGap: 14,
};
const DONUT = 132;  // logical size of each detail donut canvas (px)
const REGION_COLORS = ['#4cc2ff', '#7ee787', '#ff7b72', '#d2a8ff', '#ffa657', '#79c0ff'];

// ---- State -----------------------------------------------------------------
let meta = null;
let rasterCols = [];
let rateWindow = [];
let latestInput = null;
let latestThresholds = null;
let latestFiring = null;               // most recent firing vector (0/1)
let selected = null;                   // selected neuron ID, or null
let lastTableSel = null;
const base = {};
const ring = { weights: [], modWeights: [] };
const latest = {};
let colorScale = 1;

let viewSource = 'weights';
let deltaMode = 'current';
let playing = false;
let timer = null;
let hoverText = '';

const $ = id => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');

// ---- Geometry --------------------------------------------------------------
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
  const scaleBarX = matrixX + matrixW + CFG.scaleGap;
  L = {
    N, rowsY, histX, rasterX, rasterW, popLeftX, matrixX, matrixW, gridH, scaleBarX,
    width: scaleBarX + CFG.scaleBarW + 48,
    height: rowsY + gridH + CFG.labelH,
    stimY: CFG.labelH, popTopY: CFG.labelH + CFG.stimH + 2,
  };
}
const rowY = j => L.rowsY + j * CFG.rowH;
const colX = i => L.matrixX + i * CFG.matrixCell;
const regionColor = neuron => REGION_COLORS[meta.regionIDs[neuron] % REGION_COLORS.length];
const regionName = neuron => meta.regionNames[meta.regionIDs[neuron]];

// ---- Small helpers ---------------------------------------------------------
function hexRGB(hex) {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}
function diverging(v) {
  if (v > 0) { v = Math.min(v, 1); return [20 + 227 * v, 26 + 93 * v, 34 - 34 * v]; }
  if (v < 0) { v = Math.min(-v, 1); return [20 + 27 * v, 26 + 103 * v, 34 + 213 * v]; }
  return [20, 26, 34];
}
const rgb = c => 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
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
  return ref ? subtract(cur, ref) : null;
}

// ---- Main-view drawing -----------------------------------------------------
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
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';

  drawLabels();
  drawHistogram();
  drawRaster();
  drawPopBars();
  drawStimStrip();
  drawMatrix();
  drawScaleBar();
  drawSelection();
  updateDetail();
}

function drawLabels() {
  ctx.fillStyle = '#8b949e'; ctx.textAlign = 'left';
  ctx.fillText('rate', L.histX, CFG.labelH - 4);
  ctx.fillText('spike raster  (time →)', L.rasterX, CFG.labelH - 4);
  ctx.fillText('synapse matrix  (upstream i →, downstream j ↓)', L.matrixX, CFG.labelH - 4);
}

function drawHistogram() {
  const N = meta.numNeurons;
  if (rateWindow.length === 0) return;
  const rates = new Float32Array(N);
  for (const col of rateWindow) for (let j = 0; j < N; j++) rates[j] += col[j];
  for (let j = 0; j < N; j++) {
    ctx.fillStyle = regionColor(j);
    const w = Math.max(0.5, (rates[j] / rateWindow.length) * CFG.histW);
    ctx.fillRect(L.histX + CFG.histW - w, rowY(j), w, Math.max(1, CFG.rowH - 1));
  }
}

function drawRaster() {
  const N = meta.numNeurons, cols = CFG.rasterWidth;
  const startCol = cols - rasterCols.length;
  const off = pixelCanvas(cols, N, (x, y) => {
    const ci = x - startCol;
    if (ci < 0 || ci >= rasterCols.length) return [12, 15, 20];
    return rasterCols[ci][y] ? hexRGB(regionColor(y)) : [12, 15, 20];
  });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, L.rasterX, L.rowsY, L.rasterW, L.gridH);
}

function drawPopBars() {
  const N = meta.numNeurons;
  for (let j = 0; j < N; j++) { ctx.fillStyle = regionColor(j); ctx.fillRect(L.popLeftX, rowY(j), CFG.popBar, CFG.rowH); }
  for (let i = 0; i < N; i++) { ctx.fillStyle = regionColor(i); ctx.fillRect(colX(i), L.popTopY, CFG.matrixCell, CFG.popBarTopH); }
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
  const M = shownMatrix(), N = meta.numNeurons;
  if (!M) {
    ctx.fillStyle = '#8b949e'; ctx.textAlign = 'left';
    ctx.fillText('(no data for this view yet)', L.matrixX, L.rowsY + 14);
    return;
  }
  let maxAbs = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) maxAbs = Math.max(maxAbs, Math.abs(M[i][j]));
  colorScale = Math.max(colorScale * 0.9, maxAbs, 1e-6);
  const off = pixelCanvas(N, N, (px, py) => diverging(M[px][py] / colorScale));  // px=i upstream, py=j downstream
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, L.matrixX, L.rowsY, L.matrixW, L.gridH);
}

function drawScaleBar() {
  const x = L.scaleBarX, w = CFG.scaleBarW, y = L.rowsY, h = L.gridH, steps = 100;
  for (let k = 0; k < steps; k++) {
    ctx.fillStyle = rgb(diverging(1 - 2 * k / steps));
    ctx.fillRect(x, y + h * k / steps, w, h / steps + 1);
  }
  ctx.strokeStyle = '#2a313c'; ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  ctx.fillStyle = '#8b949e'; ctx.textAlign = 'left';
  const lab = colorScale >= 100 ? colorScale.toFixed(0) : colorScale.toPrecision(2);
  ctx.fillText('+' + lab, x + w + 4, y + 8);
  ctx.fillText('0', x + w + 4, y + h / 2 + 3);
  ctx.fillText('−' + lab, x + w + 4, y + h);
  const src = viewSource === 'weights' ? 'weight' : viewSource === 'modWeights' ? 'mod wt' : 'plast';
  ctx.fillText(src + (deltaMode === 'current' ? '' : ' Δ'), x - 2, y - 4);
}

function drawSelection() {
  if (selected === null) return;
  const j = selected;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 1;
  ctx.strokeRect(L.histX - 0.5, rowY(j) - 0.5, (L.matrixX + L.matrixW) - L.histX + 1, CFG.rowH + 1);
  ctx.strokeRect(colX(j) - 0.5, L.rowsY - 0.5, CFG.matrixCell + 1, L.gridH + 1);  // column j = its outputs
  ctx.restore();
}

// ---- Neuron detail panel ---------------------------------------------------
function getInputs(j) {
  const out = [], W = latest.weights;
  for (let i = 0; i < meta.numNeurons; i++) {
    const w = W[i][j];
    if (w !== 0) out.push({ n: i, w, region: meta.regionIDs[i], firing: latestFiring && latestFiring[i] });
  }
  return out;
}
function getOutputs(j) {
  const out = [], W = latest.weights;
  for (let k = 0; k < meta.numNeurons; k++) {
    const w = W[j][k];
    if (w !== 0) out.push({ n: k, w, region: meta.regionIDs[k], firing: latestFiring && latestFiring[k] });
  }
  return out;
}

function prepDonut(cv) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = DONUT * dpr; cv.height = DONUT * dpr;
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, DONUT, DONUT);
  return c;
}
function wedge(c, cx, cy, rIn, rOut, a0, a1) {
  c.beginPath();
  c.arc(cx, cy, rOut, a0, a1);
  c.arc(cx, cy, rIn, a1, a0, true);
  c.closePath();
}
// Draw a donut of synapses. Slice size ∝ |weight|; sign shown by colour.
function drawDonut(cv, syns, opts) {
  const c = prepDonut(cv), cx = DONUT / 2, cy = DONUT / 2;
  const rOut = DONUT * 0.46, rIn = DONUT * 0.30;
  const gOut = DONUT * 0.275, gIn = DONUT * 0.215, centerR = DONUT * 0.185;
  let total = 0;
  for (const s of syns) total += Math.abs(s.w);

  let a = -Math.PI / 2;
  for (const s of syns) {
    if (total === 0) break;
    const sweep = 2 * Math.PI * Math.abs(s.w) / total;
    wedge(c, cx, cy, rIn, rOut, a, a + sweep);
    c.fillStyle = REGION_COLORS[s.region % REGION_COLORS.length];
    c.fill();
    if (s.w < 0) { c.fillStyle = 'rgba(20,40,120,0.5)'; c.fill(); }  // inhibitory: cool wash
    if (s.firing) { c.strokeStyle = 'rgba(255,255,255,0.9)'; c.lineWidth = 2; c.stroke(); }
    a += sweep;
  }
  if (total === 0) {
    c.strokeStyle = '#2a313c'; c.lineWidth = 1;
    wedge(c, cx, cy, rIn, rOut, 0, 2 * Math.PI); c.stroke();
  }

  // Inner gauge annulus (inputs only): drive swept 0->360, threshold tick.
  if (opts.gauge) {
    let maxDrive = 0, activation = 0;
    for (const s of syns) { if (s.w > 0) maxDrive += s.w; if (s.firing) activation += s.w; }
    c.strokeStyle = '#20262e'; c.lineWidth = gOut - gIn;
    c.beginPath(); c.arc(cx, cy, (gOut + gIn) / 2, 0, 2 * Math.PI); c.stroke();
    if (maxDrive > 0) {
      const frac = Math.max(0, Math.min(1, activation / maxDrive));
      c.strokeStyle = '#4cc2ff';
      c.beginPath(); c.arc(cx, cy, (gOut + gIn) / 2, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * frac); c.stroke();
      const tFrac = Math.max(0, Math.min(1, opts.thr / maxDrive));
      const ta = -Math.PI / 2 + 2 * Math.PI * tFrac;
      c.strokeStyle = '#ff7b72'; c.lineWidth = 2;
      c.beginPath();
      c.moveTo(cx + gIn * Math.cos(ta), cy + gIn * Math.sin(ta));
      c.lineTo(cx + gOut * Math.cos(ta), cy + gOut * Math.sin(ta));
      c.stroke();
    }
  }

  // Center: lights up when the neuron itself fires.
  c.beginPath(); c.arc(cx, cy, centerR, 0, 2 * Math.PI);
  c.fillStyle = opts.fireSelf ? '#ffd666' : '#141a22'; c.fill();
  c.fillStyle = opts.fireSelf ? '#3d2f00' : '#8b949e';
  c.textAlign = 'center'; c.textBaseline = 'middle';
  c.font = '11px ui-sans-serif, system-ui, sans-serif';
  c.fillText('' + (selected === null ? '' : selected), cx, cy);
  c.textBaseline = 'alphabetic';
}

function buildTable(rows, kind) {
  rows = rows.slice().sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  const head = kind === 'in'
    ? '<tr><th>pre</th><th>pop</th><th>weight</th><th>fire</th></tr>'
    : '<tr><th>post</th><th>pop</th><th>weight</th></tr>';
  let body = '';
  for (const s of rows) {
    const dot = '<span class="dot" style="background:' + REGION_COLORS[s.region % REGION_COLORS.length] + '"></span> ';
    const pop = dot + meta.regionNames[s.region];
    if (kind === 'in')
      body += '<tr><td>' + s.n + '</td><td>' + pop + '</td><td>' + s.w.toFixed(2) + '</td><td>' + (s.firing ? '●' : '') + '</td></tr>';
    else
      body += '<tr><td>' + s.n + '</td><td>' + pop + '</td><td>' + s.w.toFixed(2) + '</td></tr>';
  }
  const cap = kind === 'in' ? 'inputs (' + rows.length + ')' : 'outputs (' + rows.length + ')';
  return '<table><caption>' + cap + '</caption>' + head + body + '</table>';
}

function updateDetail() {
  const panel = $('detail');
  if (selected === null) { panel.classList.add('hidden'); lastTableSel = null; return; }
  panel.classList.remove('hidden');
  const swatch = '<span class="dot" style="background:' + regionColor(selected) + '"></span> ';
  $('detailTitle').innerHTML = 'Neuron ' + selected + ' · ' + swatch + regionName(selected);
  const inputs = getInputs(selected), outputs = getOutputs(selected);
  drawDonut($('donutIn'), inputs, { gauge: true, thr: latestThresholds ? latestThresholds[selected] : 0, fireSelf: latestFiring && latestFiring[selected] });
  drawDonut($('donutOut'), outputs, { gauge: false, fireSelf: latestFiring && latestFiring[selected] });
  if (selected !== lastTableSel) {   // weights drift slowly; rebuild tables on selection change
    $('inputTable').innerHTML = buildTable(inputs, 'in');
    $('outputTable').innerHTML = buildTable(outputs, 'out');
    lastTableSel = selected;
  }
}

// ---- Data flow -------------------------------------------------------------
async function loadInit() {
  meta = await (await fetch('/api/init')).json();
  rasterCols = []; rateWindow = []; latestInput = null;
  latestThresholds = meta.thresholds;
  latestFiring = new Uint8Array(meta.numNeurons);
  ring.weights = []; ring.modWeights = [];
  base.weights = meta.weights; base.modWeights = meta.modWeights;
  latest.weights = meta.weights; latest.modWeights = meta.modWeights;
  latest.plasticityWeights = null; base.plasticityWeights = null;
  colorScale = 1; lastTableSel = null;
  if (selected !== null && selected >= meta.numNeurons) selected = null;
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
    latestFiring = arr;
  }
  latestInput = res.inputColumns[res.inputColumns.length - 1];
  latestThresholds = res.thresholds;
  pushMatrix('weights', res.weights);
  pushMatrix('modWeights', res.modWeights);
  $('stepCount').textContent = res.step;
  const last = rasterCols[rasterCols.length - 1];
  $('rate').textContent = (last.reduce((a, b) => a + b, 0) / last.length * 100).toFixed(1) + '%';
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
      apply(btn.dataset[attr]); colorScale = 1; draw();
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

// ---- Selection + hover -----------------------------------------------------
function neuronAt(ev) {
  const r = canvas.getBoundingClientRect();
  const y = ev.clientY - r.top;
  const j = Math.floor((y - L.rowsY) / CFG.rowH);
  return (j >= 0 && j < meta.numNeurons) ? j : null;
}
canvas.onclick = ev => {
  if (!meta) return;
  const j = neuronAt(ev);
  if (j !== null) { selected = j; draw(); }
};
canvas.onmousemove = ev => {
  if (!meta) return;
  const r = canvas.getBoundingClientRect();
  const x = ev.clientX - r.left, j = neuronAt(ev);
  if (j === null) { setHover(''); return; }
  if (x >= L.matrixX && x < L.matrixX + L.matrixW) {
    const i = Math.floor((x - L.matrixX) / CFG.matrixCell);
    const M = shownMatrix(); const val = M ? M[i][j] : NaN;
    setHover('syn ' + i + '→' + j + ' = ' + (isNaN(val) ? '—' : val.toFixed(3)));
  } else {
    setHover('neuron ' + j + ' (' + regionName(j) + ')');
  }
};
canvas.onmouseleave = () => setHover('');
function setHover(t) { if (t !== hoverText) { hoverText = t; $('hover').textContent = t || ' '; } }
$('detailClose').onclick = () => { selected = null; draw(); };

window.addEventListener('resize', () => { if (meta) draw(); });
loadInit();
