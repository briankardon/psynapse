"""Browser-based live viewer for psynapse Net simulations.

Runs a tiny standard-library HTTP server (no external dependencies) that
steps a Net and streams its state to a browser page (served from the
adjacent viewer/ directory). The page renders, in one vertically-aligned
row (neuron ID = row across all three):

    - a per-neuron firing-rate histogram,
    - a spike raster (neurons x time),
    - a connection-weight matrix (upstream i horizontal, downstream j
      vertical; cell i,j = synapse i->j), with a stimulation strip and
      population-colour bars along its edges.

The point is observability: watching activation patterns, stimulation, and
weights evolve during a simulation. The demo network built by
build_demo_net() is meant to be replaced later (e.g. with a net generated
from a Connectome).

Usage:
    python netViewer.py [port]        # serve at http://localhost:<port> (default 8137)
    python netViewer.py --selftest    # step the demo net headless and print stats
"""

# Standard library
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    # A few modulatory (threshold) connections so the "mod weight" view has
    # something to show even before the dynamics work begins.
    theNet.randomizeConnections(
        20, 0.0, 1.0,
        attributeName='region', attributeValueA=1, attributeValueB=2,
        modulatory=True,
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
                modWeights=_to_list(self.net.modConnections),
            )

    def step(self, numSteps):
        """Advance the simulation numSteps steps, returning the new state.

        firingColumns and inputColumns are lists (one entry per step) of
        per-neuron vectors, so the browser can append them to its raster and
        stimulation strip in order. Weight matrices are sent once (final).
        """
        drive = np.full(self.inputIndices.size, INPUT_DRIVE)
        firingColumns = []
        inputColumns = []
        with self.lock:
            for _ in range(numSteps):
                externalInput = np.zeros(self.net.numNeurons)
                externalInput[self.inputIndices] = INPUT_DRIVE
                self.net.addInput(drive, indices=self.inputIndices)
                self.net.activate()
                self.stepCount += 1
                firing = net.cp2np(self.net.history[:, 0])
                firingColumns.append([int(f) for f in firing])
                inputColumns.append([float(v) for v in externalInput])
            return dict(
                step=self.stepCount,
                firingColumns=firingColumns,
                inputColumns=inputColumns,
                weights=_to_list(self.net.connections),
                modWeights=_to_list(self.net.modConnections),
                thresholds=_to_list(self.net.thresholds),
            )


SIM = Simulation()


# ---- HTTP server ------------------------------------------------------------

STATIC_DIR = (Path(__file__).parent / 'viewer').resolve()
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
}


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

    def _serve_static(self, name):
        name = name.lstrip('/') or 'index.html'
        path = (STATIC_DIR / name).resolve()
        # Guard against path traversal outside the viewer directory.
        if STATIC_DIR not in path.parents or not path.is_file():
            self._not_found()
            return
        self._send(path.read_bytes(), CONTENT_TYPES.get(path.suffix, 'application/octet-stream'))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/init':
            self._send_json(SIM.init_snapshot())
        elif path == '/':
            self._serve_static('index.html')
        elif path.startswith('/api/'):
            self._not_found()
        else:
            self._serve_static(path)

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


# ---- Entry point ------------------------------------------------------------

DEFAULT_PORT = 8137


def make_server(preferredPort):
    """Bind the viewer server, falling back if the preferred port is taken.

    A port already owned by another process surfaces on Windows as
    PermissionError (WinError 10013) rather than the usual "address in use",
    so we try the preferred port, then a couple of neighbours, then finally
    an OS-assigned ephemeral port (0), which effectively always succeeds.
    """
    for port in (preferredPort, preferredPort + 1, preferredPort + 2, 0):
        try:
            return ThreadingHTTPServer(('127.0.0.1', port), ViewerHandler)
        except OSError as err:
            print('Port {p} unavailable ({e}); trying the next one...'.format(
                p=port, e=err))
    # Unreachable: binding to port 0 does not raise, but be explicit.
    raise OSError('Could not bind the viewer to any port.')


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
    port = DEFAULT_PORT
    for arg in sys.argv[1:]:
        if arg.isdigit():
            port = int(arg)
    server = make_server(port)
    actualPort = server.server_address[1]
    print('psynapse net viewer running at http://localhost:{p}/'.format(p=actualPort))
    if actualPort != port:
        print('(requested port {p} was unavailable)'.format(p=port))
    print('Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping.')
        server.shutdown()


if __name__ == '__main__':
    main()
