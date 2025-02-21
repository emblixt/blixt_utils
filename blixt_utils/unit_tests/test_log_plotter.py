import unittest
import numpy as np
import os
import inspect
from bokeh.plotting import column, figure, show
from bokeh.io import output_file
from bokeh.models import Slider, ColorPicker, Range1d

from blixt_utils.plotting.log_plotter import LogPlotter, LogColumn, Line

output_file('C:\\Users\emb\Documents\plot.html')

class SetUp(unittest.TestCase):
    def test_log_plotter1(self):
        lp = LogPlotter()
        print(lp.width)
        lp.width = 600
        print(lp.width)
        self.assertTrue(lp.width == 600)

    def test_line(self):
        line1 = Line(x=None, y=None, line_args={'legend_label':"Temp.", 'color':"blue", 'line_width':2})
        # line1 = Line()

        p = figure(title="Multiple line example", x_axis_label="x", y_axis_label="y")
        p.line(line1.x, line1.y, **line1.line_args)
        self.assertIsInstance(line1, Line)

        show(p)

    def test_column_with_lines(self):
        lp = LogPlotter()
        line1 = Line(x=None, y=None, line_args={'legend_label':"D1", 'color':"blue", 'line_width':2})
        line2 = Line(x=np.random.normal(10., 1., 500), y=np.linspace(1000., 2000., 500), line_args={'legend_label':"D2", 'color':"red", 'line_width':1})
        line3 = Line(x=np.random.normal(100., 1., 500), y=np.linspace(1000., 2000., 500), line_args={'legend_label':"D3", 'color':"yellow", 'line_width':2})
        c1 = LogColumn('c1',  lines=[line1, line2, line3])
        lp.columns = [c1]
        show(lp.figure())
        # print(inspect.getmembers(p.children[0].y_range))
        self.assertTrue(True)

    def test_two_column_plot(self):
        lp = LogPlotter(width=800, height=1000)
        line1 = Line(x=None, y=None, line_args={'legend_label': "D1", 'color': "blue", 'line_width': 2})
        line2 = Line(x=np.random.normal(10., 1., 500), y=np.linspace(1000., 2000., 500),
                     line_args={'legend_label': "D2", 'color': "red", 'line_width': 1})
        line3 = Line(x=np.random.normal(100., 1., 500), y=np.linspace(1000., 2000., 500),
                     line_args={'legend_label': "D3", 'color': "yellow", 'line_width': 2})
        c2 = LogColumn('c2',  lines=[line1, line2])
        c1 = LogColumn('c1',  lines=[line1, line2, line3], rel_width=2)
        lp.columns = [c2, c1]
        show(lp.figure())
        self.assertTrue(True)
