# -*- coding: utf-8 -*-
from bokeh import events, layouts, models
from bokeh.plotting import curdoc
from scipy.integrate import solve_ivp
import sympy as sp

from common import VectorFieldVisualization

TRAJECTORY_LINE_WIDTH = 2


class VanDerPolVisualization(VectorFieldVisualization):
    param_symbols = sp.symbols('k,')
    param_ranges = [(-2, 2)]
    param_steps = [0.1]
    param_defaults = [0]

    def __init__(self):
        super().__init__()

        self.trajectories_source = models.ColumnDataSource(data=dict(
            xs=[], ys=[],
        ))
        self.trajectory_starts_source = models.ColumnDataSource(data=dict(
            x=[], y=[],
        ))

        self.plot.on_event(events.Tap, self.plot_clicked)
        self.plot.multi_line(
            xs='xs', ys='ys',
            source=self.trajectories_source,
            line_width=TRAJECTORY_LINE_WIDTH,
            line_color='black',
        )
        self.plot.scatter(
            x='x', y='y',
            source=self.trajectory_starts_source,
            color='black',
        )

        self.t_slider = models.Slider(
            start=0, end=10, step=0.1, value=2,
            title='Trajectory duration',
        )

    @property
    def symbolic_eqns(self):
        y, dy = self.state_symbols
        k, = self.param_symbols
        return [
            dy,
            k * (1 - y**2) * dy - y,
        ]

    def as_layout(self):
        return layouts.column(
            self.plot,
            self.param_slider_box,
            self.t_slider,
        )

    def update_vector_field(self):
        super().update_vector_field()

        # Clear simulations.
        self.trajectories_source.data = dict(xs=[], ys=[])
        self.trajectory_starts_source.data = dict(x=[], y=[])

    def plot_clicked(self, event):
        self.trajectory_starts_source.stream(dict(
            x=[event.x], y=[event.y],
        ))
        solution = self.simulate(event.x, event.y)
        self.trajectories_source.stream(dict(
            xs=[solution.y[0].tolist()],
            ys=[solution.y[1].tolist()],
        ))

    def simulate(self, y0, dy0):
        return solve_ivp(self.ode, (0, self.t_slider.value), [y0, dy0], max_step=.01)

    def ode(self, t, y):
        return self.numeric_eqn(*y, *self.param_values.values()).squeeze()


def main():
    vis = VanDerPolVisualization()
    curdoc().add_root(vis.as_layout())


if __name__.startswith('bk_script'):
    main()
