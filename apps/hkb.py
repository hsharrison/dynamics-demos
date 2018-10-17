# -*- coding: utf-8 -*-
from itertools import product
from pathlib import Path
import pickle

from bokeh import layouts, models
from bokeh.plotting import curdoc, figure
import numpy as np
import sympy as sp
from tqdm import tqdm


AXIS_TITLE_FONT_SIZE = '22pt'
AXIS_TICK_FONT_SIZE = '14pt'
FIGURE_WIDTH = 800
FIGURE_HEIGHT = 400
LINE_WIDTH = 3
CIRCLE_LINE_WIDTH = 2
CIRCLE_SIZE = 20
CIRCLE_HEIGHT = 0.085
ARROW_SIZE = 15


class HKBVisualization:
    phis_in_pi_units = np.arange(-0.1, 2.101, .001)
    phis = phis_in_pi_units * np.pi
    b_values = np.arange(0, 0.51, 0.01)
    d_omega_values = np.arange(-1, 1.1, 0.1)

    def __init__(self, phase_y_range=(-2, 2), potential_y_range=(-2, 2)):
        phi, b, d_omega = sp.symbols('phi, b, Delta_omega')
        self.state_symbol = phi
        self.param_symbols = b, d_omega
        self.symbolic_eqn = d_omega - sp.sin(phi) - 2 * b * sp.sin(2 * phi)
        self.numeric_eqn = sp.lambdify((phi, b, d_omega), self.symbolic_eqn)
        self.symbolic_slope = sp.diff(self.symbolic_eqn, self.state_symbol)
        self.numeric_slope = sp.lambdify((phi, b, d_omega), self.symbolic_slope)
        self.symbolic_pot = -sp.integrate(self.symbolic_eqn, self.state_symbol)
        self.numeric_pot = sp.lambdify((phi, b, d_omega), self.symbolic_pot)

        self.roots = self.get_roots()

        self.phase_plot, self.pot_plot = self.init_plots(phase_y_range, potential_y_range)

        # Set up empty DataSources. As much as possible, these are shared between the plots.
        self.line_source = models.ColumnDataSource(data=dict(
            x=[], phase=[], v=[],
        ))
        self.attractor_source = models.ColumnDataSource(data=dict(
            x=[], phase=[], v=[],
        ))
        self.repeller_source = models.ColumnDataSource(data=dict(
            x=[], phase=[], v=[],
        ))
        self.half_source = models.ColumnDataSource(data=dict(
            x=[], phase=[], v=[],
        ))
        self.arrow_source = models.ColumnDataSource(data=dict(
            x=[], angle=[],
        ))

        # Draw all the glyphs.
        for plot, y in [(self.phase_plot, 'phase'), (self.pot_plot, 'v')]:
            plot.line(
                x='x', y=y,
                source=self.line_source,
                line_width=LINE_WIDTH,
            )
            plot.circle(
                x='x', y=y,
                source=self.attractor_source,
                size=CIRCLE_SIZE,
            )
            plot.circle(
                x='x', y=y,
                source=self.repeller_source,
                size=CIRCLE_SIZE,
                fill_color=None,
                line_width=CIRCLE_LINE_WIDTH,
            )
            plot.circle(
                x='x', y=y,
                source=self.half_source,
                size=CIRCLE_SIZE,
                fill_color=None,
                line_width=CIRCLE_LINE_WIDTH,
            )
            plot.wedge(
                x='x', y=y,
                source=self.half_source,
                radius=CIRCLE_SIZE / 2,
                start_angle=-np.pi / 2,
                end_angle=np.pi / 2,
                radius_units='screen',
                line_color=None,
            )
        self.phase_plot.triangle(
            x='x', y=0, angle='angle',
            source=self.arrow_source,
            size=ARROW_SIZE,
            color='black',
        )

        self.b_slider = models.Slider(
            start=self.b_values[0], end=self.b_values[-1], step=self.b_values[1] - self.b_values[0],
            value=0,
            title='b',
        )
        self.d_omega_slider = models.Slider(
            start=self.d_omega_values[0], end=self.d_omega_values[-1],
            step=self.d_omega_values[1] - self.d_omega_values[0],
            value=0,
            title='Δω',
        )
        self.b_slider.on_change('value', self.on_slider_change)
        self.d_omega_slider.on_change('value', self.on_slider_change)

    def as_layout(self):
        return layouts.column(
            self.phase_plot,
            self.pot_plot,
            layouts.widgetbox([self.b_slider, self.d_omega_slider]),
            sizing_mode='stretch_both',
        )

    def on_slider_change(self, attr, old, new):
        return self.update_sources()

    def update_sources(self):
        b = self.b_slider.value
        d_omega = self.d_omega_slider.value
        attractors, repellers, halfs = self.roots[int(np.round(b * 100)), int(np.round(d_omega * 100))]

        self.line_source.data = self.data_from_phis(self.phis, b, d_omega)
        self.attractor_source.data = self.data_from_phis(attractors, b, d_omega)
        self.repeller_source.data = self.data_from_phis(repellers, b, d_omega)
        self.half_source.data = self.data_from_phis(halfs, b, d_omega)

        section_edges = np.concatenate(
            [attractors, repellers, halfs, [self.phase_plot.x_range.start, self.phase_plot.x_range.end]]
        )
        section_edges.sort()
        section_midpoints = np.array([(i + j) / 2 for i, j in zip(section_edges, section_edges[1:])])
        self.arrow_source.data = dict(
            x=section_midpoints,
            angle=[
                -np.pi / 2 if phase > 0 else np.pi / 2
                for phase in self.numeric_eqn(section_midpoints, b, d_omega)
            ],
        )

    def data_from_phis(self, phis, b, d_omega):
        return dict(
            x=phis,
            phase=self.numeric_eqn(phis, b, d_omega),
            v=self.numeric_pot(phis, b, d_omega),
        )

    def get_roots(self):
        """Precalculate the roots to save on CPU and make the animation faster."""
        roots_path = Path('hkb-roots.pickle')
        if roots_path.exists():
            with roots_path.open('rb') as file:
                roots = pickle.load(file)

        else:
            roots = {
                # Use integer rather than float in dict keys to avoid floating-point weirdness.
                # Keep in mind that this limits the slider resolution.
                (int(np.round(100 * b)), int(np.round(100 * d_omega))): self.calculate_roots(b, d_omega)
                for b, d_omega in tqdm(
                    product(self.b_values, self.d_omega_values),
                    total=self.b_values.shape[0] * self.d_omega_values.shape[0],
                    desc='Calculating roots',
                )
            }
            with roots_path.open('wb') as file:
                pickle.dump(roots, file)

        return roots

    def calculate_roots(self, b, d_omega):
        attractors = []
        repellers = []
        halfs = []
        for root in sp.solveset(
                self.symbolic_eqn.subs(self.param_symbols[0], b).subs(self.param_symbols[1], d_omega),
                self.state_symbol,
                domain=sp.Interval(*self.phis[[0, -1]]),
        ):
            root = float(root)
            slope = self.numeric_slope(root, b, d_omega)
            # We can get away with a lower atol than default, because we don't expect to get values too close to zero
            # otherwise, and rtol doesn't do anything for us as we are comparing to 0.
            if np.isclose(slope, 0, atol=1e-5):
                halfs.append(root)
            elif slope > 0:
                repellers.append(root)
            else:
                attractors.append(root)

            return np.array(attractors), np.array(repellers), np.array(halfs)

    @classmethod
    def init_plots(cls, phase_y_range, potential_y_range):
        x_range = models.Range1d(*cls.phis[[0, -1]])
        first_x = -1
        x_ticker = models.FixedTicker(ticks=np.arange(first_x, cls.phis_in_pi_units[-1], 0.5) * np.pi)
        x_tick_formatter = models.FuncTickFormatter.from_py_func(pi_format)

        phase_plot = cls.init_phase_plot(
            width=FIGURE_WIDTH, height=FIGURE_HEIGHT,
            x_range=x_range, y_range=phase_y_range,
            title='Phase portrait',
        )
        phase_plot.xaxis.ticker = x_ticker
        phase_plot.xgrid.ticker = x_ticker
        phase_plot.xaxis[0].formatter = x_tick_formatter

        pot_plot = cls.init_potential_plot(
            width=FIGURE_WIDTH, height=FIGURE_HEIGHT,
            x_range=x_range, y_range=potential_y_range,
            x_axis_location='above',
            title='System potential',
        )
        pot_plot.xaxis.ticker = x_ticker
        pot_plot.xgrid.ticker = x_ticker
        pot_plot.xaxis[0].formatter = x_tick_formatter

        return phase_plot, pot_plot

    @staticmethod
    def init_phase_plot(**figure_kwargs):
        p = figure(**figure_kwargs)

        p.xaxis.axis_label = 'ϕ'
        p.yaxis.axis_label = 'dϕ / dt'
        p.xaxis.axis_label_text_font = 'helvetica'
        p.yaxis.axis_label_text_font = 'helvetica'

        # Scale is arbitrary, so no need for y axis ticks.
        y_ticker = models.FixedTicker(ticks=[])
        p.yaxis.ticker = y_ticker
        p.ygrid.ticker = y_ticker

        p.axis.axis_label_text_font_size = AXIS_TITLE_FONT_SIZE
        p.axis.major_label_text_font_size = AXIS_TICK_FONT_SIZE

        # Draw line at y=0.
        p.add_layout(models.Span(
            location=0,
            dimension='width',
            line_color='black',
        ))

        return p

    @staticmethod
    def init_potential_plot(**figure_kwargs):
        p = figure(**figure_kwargs)

        p.yaxis.axis_label = 'V'

        # Scale is arbitrary, so no need for y axis ticks.
        y_ticker = models.FixedTicker(ticks=[])
        p.yaxis.ticker = y_ticker
        p.ygrid.ticker = y_ticker

        p.axis.axis_label_text_font_size = AXIS_TITLE_FONT_SIZE
        p.axis.major_label_text_font_size = AXIS_TICK_FONT_SIZE

        return p


def pi_format():
    return (
        '{:.1f}π'.format(tick / 3.14159)
        .replace('1.0', '')
        .replace('0.0π', '0')
        .replace('.0π', 'π')
        .replace('0.5π', 'π/2')
    )


def main():
    vis = HKBVisualization()
    curdoc().add_root(vis.as_layout())


if __name__.startswith('bk_script'):
    main()
