from bokeh import events, models, layouts
from bokeh.plotting import figure
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp


AXIS_TITLE_FONT_SIZE = '22pt'
AXIS_TICK_FONT_SIZE = '14pt'
FIGURE_WIDTH = 800
FIGURE_HEIGHT = 800
VECTOR_LINE_WIDTH = 1
TRAJECTORY_LINE_WIDTH = 2
TIME_SLIDER_PARAMS = dict(start=2, end=20, step=0.1, value=10)


class VectorFieldVisualization:
    y_range = -2, 2
    dy_range = -2, 2
    vector_density = 16, 16
    arrow_scale = 0.08

    state_symbols = sp.symbols('y, ý')

    param_symbols = []
    param_ranges = []
    param_steps = []
    param_defaults = []

    def __init__(self):
        self.numeric_eqn = sp.lambdify(self.all_symbols, sp.Matrix(self.symbolic_eqns))

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
            title='Phase portrait',
        )
        self.plot.xaxis.axis_label = 'y'
        self.plot.yaxis.axis_label = 'ý'
        self.plot.axis.axis_label_text_font_size = AXIS_TITLE_FONT_SIZE
        self.plot.axis.major_label_text_font_size = AXIS_TICK_FONT_SIZE

        self.plot.on_event(events.Tap, self.plot_clicked)

        self.plot.scatter(x=self._ys, y=self._dys)
        self.plot.multi_line(
            xs='xs', ys='ys',
            source=self.vector_source,
            line_width=VECTOR_LINE_WIDTH,
        )
        self.setup_trajectory_glyphs(self.trajectory_starts_source, self.trajectories_source)

        self.t_slider = models.Slider(
            title='Trajectory duration',
            **TIME_SLIDER_PARAMS,
        )

        self.param_sliders = {}
        for param, range_, step, value in zip(
                self.param_symbols, self.param_ranges, self.param_steps, self.param_defaults,
        ):
            slider = models.Slider(
                start=range_[0], end=range_[1], step=step, value=value,
                title=str(param),
            )
            slider.on_change('value', self.on_slider_change)
            self.param_sliders[param] = slider

    def setup_trajectory_glyphs(self, start_source, trajectory_source):
        self.plot.multi_line(
            xs='xs', ys='ys',
            source=trajectory_source,
            line_width=TRAJECTORY_LINE_WIDTH,
            line_color='black',
        )
        self.plot.scatter(
            x='x', y='y',
            source=start_source,
            color='black',
        )

    @property
    def symbolic_eqns(self):
        raise NotImplementedError

    @property
    def all_symbols(self):
        return [*self.state_symbols, *self.param_symbols]

    @property
    def param_values(self):
        return {param: slider.value for param, slider in self.param_sliders.items()}

    @property
    def param_slider_box(self):
        return layouts.widgetbox(*self.param_sliders.values())

    def as_layout(self):
        return layouts.column(
            self.plot,
            self.param_slider_box,
            self.t_slider,
        )

    def on_slider_change(self, attr, old, new):
        self.update_vector_field()
        self.clear_simulations()

    def update_vector_field(self):
        points = np.stack([self._ys, self._dys], axis=-1)
        vecs = self.numeric_eqn(self._ys, self._dys, *self.param_values.values()).squeeze().T
        endpoints = points + vecs * self.arrow_scale
        xs, ys = np.stack([points, endpoints], axis=-1).transpose([1, 0, 2])
        self.vector_source.data = dict(xs=xs.tolist(), ys=ys.tolist())

    def clear_simulations(self):
        self.trajectories_source.data = dict(xs=[], ys=[])
        self.trajectory_starts_source.data = dict(x=[], y=[])

    def plot_clicked(self, event):
        self.trajectory_starts_source.stream(dict(
            x=[event.x], y=[event.y],
        ))
        trajectory = self.simulate(event.x, event.y)
        self.trajectories_source.stream(dict(
            xs=[trajectory[0].tolist()],
            ys=[trajectory[1].tolist()],
        ))

    def simulate(self, y0, dy0):
        solution = solve_ivp(self.ode, (0, self.t_slider.value), [y0, dy0], max_step=.01)
        return solution.y

    def ode(self, t, y):
        return self.numeric_eqn(*y, *self.param_values.values()).squeeze()
