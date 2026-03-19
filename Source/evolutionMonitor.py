import threading
from dataclasses import dataclass, field
import numpy as np

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go


@dataclass
class EvolutionState:
    '''Thread-safe shared state between evolution loop and monitoring GUI.

    The evolution loop writes to this; the Dash app reads from it.
    '''
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Per-generation history (appended each generation)
    generation: int = 0
    total_generations: int = 0
    gen_scores: list = field(default_factory=list)
    survivor_scores: list = field(default_factory=list)
    best_scores: list = field(default_factory=list)
    mean_scores: list = field(default_factory=list)
    gen_times: list = field(default_factory=list)

    # Current generation progress
    current_connectome: int = 0
    total_connectomes: int = 0

    # Timing
    elapsed: float = 0.0
    estimated_remaining: float = 0.0

    # Phase 2: Control signals (evolution loop reads these)
    paused: bool = False
    stop_requested: bool = False


def _format_time(seconds):
    '''Format seconds into a human-readable string.'''
    if seconds <= 0:
        return '0s'
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days > 0:
        parts.append(f'{days}d')
    if hours > 0:
        parts.append(f'{hours}h')
    if minutes > 0:
        parts.append(f'{minutes}m')
    if secs > 0 or len(parts) == 0:
        parts.append(f'{secs}s')
    return ' '.join(parts)


def create_app(state):
    '''Create and return a Dash app that monitors the given EvolutionState.'''

    app = Dash(__name__)

    app.layout = html.Div(style={'fontFamily': 'monospace', 'padding': '20px'}, children=[
        dcc.Interval(id='refresh', interval=2000),

        # Header
        html.H2('Evolution Monitor'),
        html.Div(id='header-stats', style={'marginBottom': '20px', 'fontSize': '16px'}),

        # Charts row 1
        html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
            dcc.Graph(id='fitness-plot', style={'flex': '1'}),
            dcc.Graph(id='distribution-plot', style={'flex': '1'}),
        ]),

        # Charts row 2
        dcc.Graph(id='timing-plot'),
    ])

    @app.callback(
        [Output('header-stats', 'children'),
         Output('fitness-plot', 'figure'),
         Output('distribution-plot', 'figure'),
         Output('timing-plot', 'figure')],
        [Input('refresh', 'n_intervals')]
    )
    def update_dashboard(_n):
        # Read state (no lock needed for reads)
        gen = state.generation
        total_gen = state.total_generations
        cur_co = state.current_connectome
        total_co = state.total_connectomes
        elapsed = state.elapsed
        remaining = state.estimated_remaining
        best_scores = list(state.best_scores)
        mean_scores = list(state.mean_scores)
        gen_scores = list(state.gen_scores)
        gen_times = list(state.gen_times)

        # Header stats
        if total_gen > 0:
            header = html.Div([
                html.Span(f'Generation {gen + 1} / {total_gen}'),
                html.Span(' | ', style={'margin': '0 10px'}),
                html.Span(f'Connectome {cur_co} / {total_co}'),
                html.Span(' | ', style={'margin': '0 10px'}),
                html.Span(f'Elapsed: {_format_time(elapsed)}'),
                html.Span(' | ', style={'margin': '0 10px'}),
                html.Span(f'Remaining: {_format_time(remaining)}'),
            ])
        else:
            header = html.Span('Waiting for evolution to start...')

        # Fitness over time
        fitness_fig = go.Figure()
        if len(best_scores) > 0:
            gens = list(range(1, len(best_scores) + 1))
            fitness_fig.add_trace(go.Scatter(
                x=gens, y=best_scores, mode='lines+markers', name='Best'))
            fitness_fig.add_trace(go.Scatter(
                x=gens, y=mean_scores, mode='lines+markers', name='Mean'))
        fitness_fig.update_layout(
            title='Fitness Over Generations',
            xaxis_title='Generation',
            yaxis_title='Score (lower is better)',
            margin=dict(l=50, r=20, t=40, b=40))

        # Score distribution for latest generation
        dist_fig = go.Figure()
        if len(gen_scores) > 0:
            latest_scores = gen_scores[-1]
            dist_fig.add_trace(go.Box(y=latest_scores, name=f'Gen {len(gen_scores)}'))
        dist_fig.update_layout(
            title='Score Distribution (Latest Generation)',
            yaxis_title='Score',
            margin=dict(l=50, r=20, t=40, b=40))

        # Generation timing
        timing_fig = go.Figure()
        if len(gen_times) > 0:
            timing_fig.add_trace(go.Bar(
                x=list(range(1, len(gen_times) + 1)),
                y=gen_times))
        timing_fig.update_layout(
            title='Time Per Generation',
            xaxis_title='Generation',
            yaxis_title='Seconds',
            margin=dict(l=50, r=20, t=40, b=40))

        return header, fitness_fig, dist_fig, timing_fig

    return app


def launch(state, port=8050):
    '''Launch the monitoring dashboard.'''
    app = create_app(state)
    app.run(debug=False, port=port)
