from bokeh import models, layouts
from bokeh.plotting import figure
import numpy as np
import sympy as sp


AXIS_TITLE_FONT_SIZE = '22pt'
AXIS_TICK_FONT_SIZE = '14pt'
FIGURE_WIDTH = 800
FIGURE_HEIGHT = 800
VECTOR_LINE_WIDTH = 1


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

        self.plot = figure(
            width=FIGURE_WIDTH, height=FIGURE_HEIGHT,
            x_range=self.y_range, y_range=self.dy_range,
        )
        self.plot.xaxis.axis_label = 'y'
        self.plot.yaxis.axis_label = 'ý'
        self.plot.axis.axis_label_text_font_size = AXIS_TITLE_FONT_SIZE
        self.plot.axis.major_label_text_font_size = AXIS_TICK_FONT_SIZE

        self.plot.scatter(x=self._ys, y=self._dys)
        self.plot.multi_line(
            xs='xs', ys='ys',
            source=self.vector_source,
            line_width=VECTOR_LINE_WIDTH,
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
        return layouts.column(self.plot, self.param_slider_box)

    def on_slider_change(self, attr, old, new):
        return self.update_vector_field()

    def update_vector_field(self):
        points = np.stack([self._ys, self._dys], axis=-1)
        vecs = self.numeric_eqn(self._ys, self._dys, *self.param_values.values()).squeeze().T
        endpoints = points + vecs * self.arrow_scale
        xs, ys = np.stack([points, endpoints], axis=-1).transpose([1, 0, 2])
        self.vector_source.data = dict(xs=xs.tolist(), ys=ys.tolist())
