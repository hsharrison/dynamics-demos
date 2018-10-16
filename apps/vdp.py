# -*- coding: utf-8 -*-
from bokeh.plotting import curdoc
import sympy as sp

from common import VectorFieldVisualization


class VanDerPolVisualization(VectorFieldVisualization):
    param_symbols = sp.symbols('k,')
    param_ranges = [(-2, 2)]
    param_steps = [0.1]
    param_defaults = [0]
    y_range = -3, 3
    dy_range = -3, 3

    @property
    def symbolic_eqns(self):
        y, dy = self.state_symbols
        k, = self.param_symbols
        return [
            dy,
            k * (1 - y**2) * dy - y,
        ]


def main():
    vis = VanDerPolVisualization()
    curdoc().add_root(vis.as_layout())


if __name__.startswith('bk_script'):
    main()
