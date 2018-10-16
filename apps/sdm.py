# -*- coding: utf-8 -*-
from itertools import chain

from bokeh import events, layouts, models
from bokeh.plotting import curdoc, figure
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from sympy.utilities.autowrap import autowrap

from common import VectorFieldVisualization

FIGURE_WIDTH = 600
PADDING = 0.5
CIRCLE_SIZE = 20
AGENT_COLOR = 'navy'
GOAL_COLOR = 'green'
OBSTACLE_COLOR = 'firebrick'
NULLCLINE_WIDTH = 3
INDICATOR_SIZE = 30
INDICATOR_LINE_WIDTH = 5
TRAJECTORY_LINE_WIDTH = 3
GOAL_COLLISION_RADIUS = 0.2


class SteeringModelVisualization(VectorFieldVisualization):
    arrow_scale = .025
    nullcline_density = .01
    state_symbols = sp.symbols('phi, dphi, x, y')
    y_range = -3.14, 3.14
    default_params = dict(
        b=3.25,
        k_g=7.5,
        c_1=0.4,
        c_2=0.4,
        k_o=198,
        c_3=6.5,
        c_4=0.8,
    )
    goal_symbols = sp.symbols('gx, gy')

    def __init__(self, start_position, goal_position, obstacles, speed=1, param_values=None):
        self.position = start_position
        self.goal_position = goal_position
        self.obstacles = obstacles
        self.obstacle_symbols = [sp.symbols(f'o{i}x, o{i}y') for i, _ in enumerate(obstacles)]
        self.speed = speed
        self.params = self.default_params
        self.params.update(param_values or {})

        xs = [start_position[0], goal_position[0], *(o[0] for o in obstacles)]
        ys = [start_position[1], goal_position[1], *(o[1] for o in obstacles)]
        self.birdseye_plot = figure(
            width=FIGURE_WIDTH,
            match_aspect=True,
            x_range=models.Range1d(min(xs) - PADDING, max(xs) + PADDING),
            y_range=models.Range1d(min(ys) - PADDING, max(ys) + PADDING),
            title='Top-down view',
        )
        super().__init__()

        self.numeric_eqn = None
        self.numeric_nullcline_eqn =None
        self.compiled_eqn = None
        self.compile()

        self.objects_source = models.ColumnDataSource(data=dict(x=[], y=[], color=[], name=[]))
        self.objects_source.on_change('data', self.positions_changed)
        self.trajectory_source = models.ColumnDataSource(data=dict(x=[], y=[]))

        self.angle_indicator_source = models.ColumnDataSource(data=dict(x=[], y=[], color=[]))
        self.nullcline_source = models.ColumnDataSource(data=dict(x=[], y=[]))

        self.plot.width = FIGURE_WIDTH
        self.plot.width = FIGURE_WIDTH
        self.plot.xaxis.axis_label = 'φ'
        self.plot.yaxis.axis_label = 'dφ/dt'

        self.plot.line(
            x='x', y='y',
            source=self.nullcline_source,
            color='black',
            line_width=NULLCLINE_WIDTH,
        )
        self.plot.cross(
            x='x', y='y', color='color',
            source=self.angle_indicator_source,
            size=INDICATOR_SIZE,
            line_width=INDICATOR_LINE_WIDTH,
            alpha=0.7,
        )

        for dim in ['width', 'height']:
            self.plot.add_layout(models.Span(
                location=0, dimension=dim,
                line_color='black',
                line_width=1,
                line_dash='dashed',
            ))

        self.birdseye_plot.axis.visible = False
        self.birdseye_plot.grid.visible = False
        self.birdseye_plot.on_event(events.DoubleTap, self.on_doubleclick)

        objects = self.birdseye_plot.circle(
            x='x', y='y', color='color',
            source=self.objects_source,
            size=CIRCLE_SIZE,
        )
        self.birdseye_plot.line(
            x='x', y='y',
            source=self.trajectory_source,
            color=AGENT_COLOR,
        )

        drag_tool = models.PointDrawTool(renderers=[objects])
        hover_tool = models.HoverTool(
            renderers=[objects],
            tooltips=[('Object', '@name')],
        )
        self.birdseye_plot.tools = [drag_tool, hover_tool]
        self.birdseye_plot.toolbar.active_drag = drag_tool

        self.sim_button = models.Button(label='Sim')
        self.sim_button.on_click(self.sim_button_clicked)
        self.clear_button = models.Button(label='Clear')
        self.clear_button.on_click(self.clear_button_clicked)

        self.update_birdseye_plot()

    def compile(self):
        self.numeric_eqn = sp.lambdify(self.all_symbols, sp.Matrix(self.symbolic_eqns))
        self.numeric_nullcline_eqn = sp.lambdify([self.state_symbols[0], *self.all_symbols[2:]], self.nullcline_eqn)
        self.compiled_eqn = autowrap(sp.Matrix(self.symbolic_eqns), args=self.all_symbols, backend='f2py')

    def setup_trajectory_glyphs(self, start_source, trajectory_source):
        self.birdseye_plot.multi_line(
            xs='xs', ys='ys',
            source=trajectory_source,
            line_width=TRAJECTORY_LINE_WIDTH,
            line_color=AGENT_COLOR,
        )
        self.birdseye_plot.circle(
            x='x', y='y',
            source=start_source,
            fill_color=None,
            line_color=AGENT_COLOR,
            size=CIRCLE_SIZE,
        )

    @property
    def all_symbols(self):
        return [*self.state_symbols, *self.goal_symbols, *chain.from_iterable(self.obstacle_symbols)]

    @property
    def position_values(self):
        return [*self.position, *self.goal_position, *chain.from_iterable(self.obstacles)]

    @property
    def symbolic_eqns(self):
        phi, dphi, x, y = self.state_symbols

        goal_distance, goal_angle = polar_vec((x, y), self.goal_symbols)

        damping_term = -self.params['b'] * dphi
        goal_distance_mod = sp.exp(-self.params['c_1'] * goal_distance) + self.params['c_2']
        goal_term = -self.params['k_g'] * (phi - goal_angle) * goal_distance_mod
        obstacle_terms = 0
        for obs_pos in self.obstacle_symbols:
            obs_distance, obs_angle = polar_vec((x, y), obs_pos)
            obs_beta = phi - obs_angle
            obs_beta_mod = sp.exp(-self.params['c_3'] * abs(obs_beta))
            obs_distance_mod = sp.exp(-self.params['c_4'] * obs_distance)
            obstacle_terms += self.params['k_o'] * obs_beta * obs_beta_mod * obs_distance_mod
        return [
            dphi,
            damping_term + goal_term + obstacle_terms,
            self.speed * sp.cos(phi),
            self.speed * sp.sin(phi),
        ]

    @property
    def nullcline_eqn(self):
        return (self.symbolic_eqns[1] + self.params['b'] * self.state_symbols[1]).simplify() / self.params['b']

    def as_layout(self):
        return layouts.row(
            self.plot,
            layouts.column(
                self.birdseye_plot,
                layouts.row(self.sim_button, self.clear_button),
            ),
        )

    def positions_changed(self, attr, old, new):
        self.position = new['x'][0], new['y'][0]
        self.goal_position = new['x'][1], new['y'][1]
        self.obstacles = [
            (x, y) for x, y in zip(new['x'][2:], new['y'][2:])
        ]
        self.update_vector_field()

    def update_vector_field(self):
        points = np.stack([self._ys, self._dys], axis=-1)
        vecs = self.numeric_eqn(self._ys, self._dys, *self.position_values).squeeze().T[:, :2]
        endpoints = points + vecs * self.arrow_scale
        xs, ys = np.stack([points, endpoints], axis=-1).transpose([1, 0, 2])
        self.vector_source.data = dict(xs=xs.tolist(), ys=ys.tolist())
        phis = np.arange(*self.y_range, self.nullcline_density)
        self.nullcline_source.data = dict(
            x=phis, y=self.numeric_nullcline_eqn(phis, *self.position_values),
        )
        goal_angle = polar_vec(self.position, self.goal_position, numeric=True)[1]
        obs_angles = [polar_vec(self.position, obs, numeric=True)[1] for obs in self.obstacles]
        self.angle_indicator_source.data = dict(
            x=[goal_angle, *obs_angles], y=[0 for _ in range(len(self.obstacles) + 1)],
            color=[GOAL_COLOR, *(OBSTACLE_COLOR for _ in self.obstacles)],
        )

    def update_birdseye_plot(self):
        xs, ys = zip(self.position, self.goal_position, *self.obstacles)
        self.objects_source.data = dict(
            x=list(xs), y=list(ys),
            color=[AGENT_COLOR, GOAL_COLOR, *(OBSTACLE_COLOR for _ in self.obstacles)],
            name=['Agent', 'Goal', *('Obstacle' for _ in self.obstacles)],
        )

    def sim_button_clicked(self):
        self.trajectory_starts_source.stream(dict(
            x=[self.position[0]], y=[self.position[1]],
        ))
        for phi0 in np.linspace(-np.pi, np.pi, 8):
            trajectory = self.simulate(phi0)
            self.trajectories_source.stream(dict(
                xs=[trajectory[2].tolist()],
                ys=[trajectory[3].tolist()],
            ))

    def clear_button_clicked(self):
        self.clear_simulations()

    def plot_clicked(self, event):
        # Disable the usual sim-to-click behavior of the parent class.
        pass

    def simulate(self, phi0):
        y0 = [phi0, 0, *self.position]
        solution = solve_ivp(self.get_ode(), (0, self.t_slider.value), y0, max_step=.01)
        trajectory = solution.y
        # Crop trajectory
        goal_distance, goal_angle = polar_vec(trajectory[2:], self.goal_position, numeric=True)
        in_radius = goal_distance < GOAL_COLLISION_RADIUS
        if in_radius.any():
            trajectory = trajectory[:, :np.argmax(in_radius)]
        return trajectory

    def get_ode(self):
        return lambda t, y: self.compiled_eqn(*y, *self.position_values[2:]).squeeze()

    def on_doubleclick(self, event):
        print('Adding obstacle...', end='')
        i = len(self.obstacles)
        self.obstacles.append((event.x, event.y))
        self.obstacle_symbols.append(sp.symbols(f'o{i}x, o{i}y'))
        self.compile()
        self.update_birdseye_plot()
        print('done.')


def polar_vec(position_a, position_b, numeric=False):
    if numeric:
        atan2 = np.arctan2
        sqrt = np.sqrt
    else:
        atan2 = sp.atan2
        sqrt = sp.sqrt
    vec_x = position_b[0] - position_a[0]
    vec_y = position_b[1] - position_a[1]
    distance = sqrt(vec_x**2 + vec_y**2)
    angle = atan2(vec_y, vec_x)
    return distance, angle


def main():
    vis = SteeringModelVisualization(
        start_position=(0, 0),
        goal_position=(2, 7),
        obstacles=[(1, 3)],
    )
    curdoc().add_root(vis.as_layout())


if __name__.startswith('bk_script'):
    main()
