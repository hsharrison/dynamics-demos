# -*- coding: utf-8 -*-
from bokeh import events, layouts, models
from bokeh.plotting import curdoc, figure
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

AXIS_TITLE_FONT_SIZE = '22pt'
AXIS_TICK_FONT_SIZE = '14pt'
FIGURE_WIDTH = 800
FIGURE_HEIGHT = 800
VECTOR_LINE_WIDTH = 1
TRAJECTORY_LINE_WIDTH = 2


class VanDerPolVisualization:
    y_range = -2, 2
    dy_range = -2, 2
    k_range = -2, 2
    vector_density = 16, 16
    arrow_scale = 0.08

    def __init__(self):
        k, y, dy = sp.symbols('k, y, ý')
        self.state_symbols = y, dy
        self.param_symbols = k,

        self.symbolic_eqns = [
            dy,
            k * (1 - y**2) * dy - y,
        ]
        self.numeric_eqn = sp.lambdify((y, dy, k), sp.Matrix(self.symbolic_eqns))

        ys = np.linspace(*self.y_range, self.vector_density[0])
        dys = np.linspace(*self.dy_range, self.vector_density[1])
        y_grid, dy_grid = np.meshgrid(ys, dys)
        self._ys = y_grid.flatten()
        self._dys = dy_grid.flatten()

        self.vector_source = models.ColumnDataSource(data=dict(
            xs=[], ys=[],
        ))
        self.trajectories_source = models.ColumnDataSource(data=dict(
            xs=[], ys=[],
        ))
        self.trajectory_starts_source = models.ColumnDataSource(data=dict(
            x=[], y=[],
        ))

        self.plot = figure(
            width=FIGURE_WIDTH, height=FIGURE_HEIGHT,
            x_range=self.y_range, y_range=self.dy_range,
        )
        self.plot.on_event(events.Tap, self.plot_clicked)
        self.plot.scatter(x=self._ys, y=self._dys)
        self.plot.multi_line(
            xs='xs', ys='ys',
            source=self.vector_source,
            line_width=VECTOR_LINE_WIDTH,
        )
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

        self.k_slider = models.Slider(
            start=self.k_range[0], end=self.k_range[1], step=0.1, value=0,
            title='k',
        )
        self.k_slider.on_change('value', self.on_slider_change)
        self.t_slider = models.Slider(
            start=0, end=10, step=0.1, value=2,
            title='Trajectory duration',
        )

    def as_layout(self):
        return layouts.column(
            self.plot,
            self.k_slider,
            self.t_slider,
        )

    def on_slider_change(self, attr, old, new):
        return self.update_vector_field(self.k_slider.value)

    def update_vector_field(self, k):
        points = np.stack([self._ys, self._dys], axis=-1)
        vecs = self.numeric_eqn(self._ys, self._dys, k).squeeze().T
        endpoints = points + vecs * self.arrow_scale
        xs, ys = np.stack([points, endpoints], axis=-1).transpose([1, 0, 2])
        self.vector_source.data = dict(xs=xs.tolist(), ys=ys.tolist())

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
        return self.numeric_eqn(*y, self.k_slider.value).squeeze()


def main():
    vis = VanDerPolVisualization()
    curdoc().add_root(vis.as_layout())


if __name__.startswith('bk_script'):
    main()
