"""Browser-based live viewer for psynapse Net simulations.

Runs a tiny standard-library HTTP server (no external dependencies) that
steps a Net and streams its state to a browser page, which renders:

    - a spike raster (neurons x time),
    - per-neuron firing-rate bars, grouped by region,
    - a connection-weight heatmap that updates as plasticity changes it.

The point is observability: watching activation patterns and weights evolve
during a simulation, rather than only summary statistics. The demo network
built by build_demo_net() is meant to be replaced later (e.g. with a net
generated from a Connectome), but it gives the viewer something lively to
show out of the box.

Usage:
    python netViewer.py [port]        # serve at http://localhost:<port> (default 8080)
    python netViewer.py --selftest    # step the demo net headless and print stats
"""

# Standard library
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Third-party
import numpy as np

# Local
import net


# ---- Demo network -----------------------------------------------------------

# Constant drive applied to the input region each step. Kept above the mean
# threshold so input neurons fire regularly (subject to their refractory
# period) and seed activity that propagates downstream.
INPUT_DRIVE = 1.5

REGION_NAMES = ['input', 'hidden', 'output']
NEURONS_PER_REGION = 20


def build_demo_net():
    """Build a small three-region net that sustains visible activity.

    Returns:
        (theNet, regionIDs, inputIndices) where regionIDs is a list giving
        each neuron's region index, and inputIndices selects the neurons that
        receive external drive each step.
    """
    numRegions = len(REGION_NAMES)
    numNeurons = NEURONS_PER_REGION * numRegions

    theNet = net.Net(
        numNeurons=numNeurons,
        thresholdMean=1.0,
        thresholdSigma=0.2,
        refractoryPeriodMean=4,
        refractoryPeriodSigma=1,
        hebbianPlasticityRate=0.05,
        homeostaticPlasticityFactor=0.05,
        historyLength=300,
    )

    regionIDs = []
    for regionIndex in range(numRegions):
        regionIDs.extend([regionIndex] * NEURONS_PER_REGION)
    regionMap = dict(enumerate(REGION_NAMES))
    theNet.setAttributes('region', values=regionIDs, attributeMap=regionMap)

    # Feedforward + recurrent connectivity. Strengths are well above threshold
    # so a single upstream spike can drive a downstream neuron.
    mu, sigma = 4.0, 1.5
    theNet.randomizeConnections(  # input -> hidden
        40, mu, sigma,
        attributeName='region', attributeValueA=0, attributeValueB=1,
    )
    theNet.randomizeConnections(  # hidden -> hidden (recurrent)
        60, mu, sigma,
        attributeName='region', attributeValueA=1, attributeValueB=1,
    )
    theNet.randomizeConnections(  # hidden -> output
        40, mu, sigma,
        attributeName='region', attributeValueA=1, attributeValueB=2,
    )

    inputIndices = np.arange(NEURONS_PER_REGION)
    return theNet, regionIDs, inputIndices


# ---- Simulation state -------------------------------------------------------

def _to_list(array):
    """Convert a cupy-or-numpy array to a plain nested Python list."""
    return net.cp2np(array).tolist()


class Simulation:
    """Holds the live Net and steps it under a lock for thread safety."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.net, self.regionIDs, self.inputIndices = build_demo_net()
            self.stepCount = 0
        return self.init_snapshot()

    def init_snapshot(self):
        """Metadata + full state the browser needs to (re)initialize."""
        with self.lock:
            return dict(
                numNeurons=self.net.numNeurons,
                regionIDs=[int(r) for r in self.regionIDs],
                regionNames=REGION_NAMES,
                step=self.stepCount,
                weights=_to_list(self.net.connections),
            )

    def step(self, numSteps):
        """Advance the simulation numSteps steps, returning the new state.

        firingColumns is a list (one entry per step) of 0/1 firing vectors,
        so the browser can append them to its raster in order.
        """
        drive = np.full(self.inputIndices.size, INPUT_DRIVE)
        firingColumns = []
        with self.lock:
            for _ in range(numSteps):
                self.net.addInput(drive, indices=self.inputIndices)
                self.net.activate()
                self.stepCount += 1
                firing = net.cp2np(self.net.history[:, 0])
                firingColumns.append([int(f) for f in firing])
            return dict(
                step=self.stepCount,
                firingColumns=firingColumns,
                weights=_to_list(self.net.connections),
                thresholds=_to_list(self.net.thresholds),
            )


SIM = Simulation()


# ---- HTTP server ------------------------------------------------------------

class ViewerHandler(BaseHTTPRequestHandler):
    def _send(self, body, contentType):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', contentType)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj):
        self._send(json.dumps(obj), 'application/json')

    def _not_found(self):
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._send(PAGE, 'text/html; charset=utf-8')
        elif path == '/api/init':
            self._send_json(SIM.init_snapshot())
        else:
            self._not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/step':
            query = parse_qs(parsed.query)
            numSteps = int(query.get('n', ['1'])[0])
            numSteps = max(1, min(numSteps, 50))
            self._send_json(SIM.step(numSteps))
        elif parsed.path == '/api/reset':
            self._send_json(SIM.reset())
        else:
            self._not_found()

    def log_message(self, *args):
        # Silence the default per-request stderr logging.
        pass


# ---- Browser page (self-contained: HTML + CSS + JS) -------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>psynapse net viewer</title>
<style>
  :root {
    --bg: #0e1116; --panel: #161b22; --edge: #2a313c;
    --text: #d7dde5; --muted: #8b949e; --accent: #4cc2ff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 10px 16px; border-bottom: 1px solid var(--edge); background: var(--panel);
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .2px; }
  header .stat { color: var(--muted); }
  header .stat b { color: var(--text); font-variant-numeric: tabular-nums; }
  .spacer { flex: 1; }
  button, .ctrl {
    background: #21262d; color: var(--text); border: 1px solid var(--edge);
    border-radius: 6px; padding: 5px 11px; font: inherit; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); color: #04121c; border-color: var(--accent); font-weight: 600; }
  label.ctrl { display: inline-flex; align-items: center; gap: 7px; cursor: default; }
  input[type=range] { accent-color: var(--accent); }
  main { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; padding: 14px; }
  .panel { background: var(--panel); border: 1px solid var(--edge); border-radius: 8px; padding: 12px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); margin: 0 0 8px; }
  .full { grid-column: 1 / -1; }
  canvas { width: 100%; display: block; background: #0a0d12; border-radius: 4px; image-rendering: pixelated; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; color: var(--muted); font-size: 12px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
</style>
</head>
<body>
<header>
  <h1>psynapse&nbsp;·&nbsp;net viewer</h1>
  <button id="playBtn" class="primary">▶ Play</button>
  <button id="stepBtn">Step</button>
  <button id="resetBtn">Reset</button>
  <label class="ctrl">Speed
    <input id="speed" type="range" min="1" max="60" value="20">
  </label>
  <label class="ctrl">Steps/tick
    <input id="burst" type="range" min="1" max="10" value="1">
    <span id="burstVal" style="color:var(--text)">1</span>
  </label>
  <span class="spacer"></span>
  <span class="stat">step <b id="stepCount">0</b></span>
  <span class="stat">firing <b id="rate">0.0%</b></span>
</header>

<main>
  <section class="panel full">
    <h2>Spike raster &nbsp;<span style="color:var(--muted);text-transform:none;letter-spacing:0">(neurons &times; time &rarr;)</span></h2>
    <canvas id="raster"></canvas>
    <div class="legend" id="regionLegend"></div>
  </section>

  <section class="panel">
    <h2>Firing rate by neuron</h2>
    <canvas id="rates"></canvas>
  </section>

  <section class="panel">
    <h2>Connection weights <span style="color:var(--muted);text-transform:none;letter-spacing:0">(pre &rarr; post)</span></h2>
    <canvas id="weights"></canvas>
    <div class="legend">
      <span><i class="swatch" style="background:#2f81f7"></i> inhibitory</span>
      <span><i class="swatch" style="background:#141a22"></i> none</span>
      <span><i class="swatch" style="background:#f7772f"></i> excitatory</span>
    </div>
  </section>
</main>

<script>
const REGION_COLORS = ['#4cc2ff', '#7ee787', '#ff7b72', '#d2a8ff', '#ffa657', '#79c0ff'];
const RASTER_WIDTH = 240;   // time steps shown

let meta = null;            // {numNeurons, regionIDs, regionNames}
let rasterCols = [];        // array of Uint8Array(N), newest last
let rateWindow = [];        // recent firing columns for rolling rate
let latestWeights = null;
let weightScale = 1;
let playing = false;
let timer = null;

const $ = id => document.getElementById(id);

// Render an N-wide, M-tall matrix of values into an offscreen canvas via
// ImageData, then blit it scaled (nearest-neighbour) onto a visible canvas.
function blitMatrix(canvas, cols, rows, colorFn, cssHeight) {
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  const octx = off.getContext('2d');
  const img = octx.createImageData(cols, rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const [r, g, b] = colorFn(x, y);
      const i = (y * cols + x) * 4;
      img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  const cssW = canvas.clientWidth || 600;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.height = cssHeight + 'px';
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

function drawRaster() {
  if (!meta) return;
  const N = meta.numNeurons;
  const cols = RASTER_WIDTH;
  const startX = cols - rasterCols.length;
  blitMatrix($('raster'), cols, N, (x, y) => {
    const ci = x - startX;
    if (ci < 0 || ci >= rasterCols.length) return [12, 15, 20];
    if (rasterCols[ci][y]) {
      const c = REGION_COLORS[meta.regionIDs[y] % REGION_COLORS.length];
      return [parseInt(c.slice(1,3),16), parseInt(c.slice(3,5),16), parseInt(c.slice(5,7),16)];
    }
    return [12, 15, 20];
  }, Math.max(180, N * 4));
}

function drawWeights() {
  if (!latestWeights) return;
  const N = latestWeights.length;
  let maxAbs = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) maxAbs = Math.max(maxAbs, Math.abs(latestWeights[i][j]));
  // Smooth the scale so the heatmap doesn't flicker as weights drift.
  weightScale = Math.max(weightScale * 0.9, maxAbs, 0.5);
  blitMatrix($('weights'), N, N, (x, y) => {
    // Row = presynaptic (i), col = postsynaptic (j).
    const w = latestWeights[y][x] / weightScale;
    if (w > 0) return [20 + 235*w, 26 + 93*w, 34 - 34*Math.min(w,1)];
    if (w < 0) return [20 + 27*w*-1, 26 + 103*-w, 34 + 189*-w];
    return [20, 26, 34];
  }, ($('weights').clientWidth || 300));
}

function drawRates() {
  const canvas = $('rates');
  if (!meta || rateWindow.length === 0) { return; }
  const N = meta.numNeurons;
  const rates = new Float32Array(N);
  for (const col of rateWindow)
    for (let y = 0; y < N; y++) rates[y] += col[y];
  for (let y = 0; y < N; y++) rates[y] /= rateWindow.length;

  const cssW = canvas.clientWidth || 300;
  const cssH = Math.max(180, N * 4);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
  canvas.style.height = cssH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);
  const barH = cssH / N;
  for (let y = 0; y < N; y++) {
    ctx.fillStyle = REGION_COLORS[meta.regionIDs[y] % REGION_COLORS.length];
    ctx.fillRect(0, y * barH, Math.max(1, rates[y] * (cssW - 2)), Math.max(1, barH - 1));
  }
}

function updateLegend() {
  const el = $('regionLegend');
  el.innerHTML = '';
  meta.regionNames.forEach((name, i) => {
    const s = document.createElement('span');
    s.innerHTML = '<i class="swatch" style="background:' +
      REGION_COLORS[i % REGION_COLORS.length] + '"></i>' + name;
    el.appendChild(s);
  });
}

function redraw() { drawRaster(); drawWeights(); drawRates(); }

async function loadInit() {
  meta = await (await fetch('/api/init')).json();
  rasterCols = []; rateWindow = [];
  latestWeights = meta.weights; weightScale = 1;
  $('stepCount').textContent = meta.step;
  updateLegend();
  redraw();
}

async function doStep(n) {
  const res = await (await fetch('/api/step?n=' + n, { method: 'POST' })).json();
  for (const col of res.firingColumns) {
    const arr = Uint8Array.from(col);
    rasterCols.push(arr);
    if (rasterCols.length > RASTER_WIDTH) rasterCols.shift();
    rateWindow.push(arr);
    if (rateWindow.length > 60) rateWindow.shift();
  }
  latestWeights = res.weights;
  $('stepCount').textContent = res.step;
  const last = rasterCols[rasterCols.length - 1];
  const frac = last.reduce((a, b) => a + b, 0) / last.length;
  $('rate').textContent = (frac * 100).toFixed(1) + '%';
  redraw();
}

function tick() {
  const burst = parseInt($('burst').value, 10);
  doStep(burst);
}

function setPlaying(on) {
  playing = on;
  $('playBtn').textContent = on ? '❚❚ Pause' : '▶ Play';
  if (timer) { clearInterval(timer); timer = null; }
  if (on) {
    const hz = parseInt($('speed').value, 10);
    timer = setInterval(tick, 1000 / hz);
  }
}

$('playBtn').onclick = () => setPlaying(!playing);
$('stepBtn').onclick = () => { setPlaying(false); tick(); };
$('resetBtn').onclick = async () => { setPlaying(false); await loadInit(); };
$('speed').oninput = () => { if (playing) setPlaying(true); };
$('burst').oninput = () => { $('burstVal').textContent = $('burst').value; };
window.addEventListener('resize', redraw);

loadInit();
</script>
</body>
</html>
"""


# ---- Entry point ------------------------------------------------------------

def selftest(numSteps=200):
    """Step the demo net headless and report basic activity statistics."""
    snap = SIM.init_snapshot()
    print('Demo net: {n} neurons, regions {r}'.format(
        n=snap['numNeurons'], r=snap['regionNames']))
    fired = 0
    for _ in range(numSteps):
        result = SIM.step(1)
        fired += sum(result['firingColumns'][0])
    meanFrac = fired / (numSteps * snap['numNeurons'])
    print('Ran {s} steps. Mean firing fraction/step: {f:.3f}'.format(
        s=numSteps, f=meanFrac))
    weights = np.array(SIM.init_snapshot()['weights'])
    print('Nonzero connections: {nz}, weight range [{lo:.2f}, {hi:.2f}]'.format(
        nz=int(np.count_nonzero(weights)), lo=weights.min(), hi=weights.max()))


def main():
    if '--selftest' in sys.argv:
        selftest()
        return
    port = 8080
    for arg in sys.argv[1:]:
        if arg.isdigit():
            port = int(arg)
    server = ThreadingHTTPServer(('127.0.0.1', port), ViewerHandler)
    print('psynapse net viewer running at http://localhost:{p}/'.format(p=port))
    print('Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping.')
        server.shutdown()


if __name__ == '__main__':
    main()
