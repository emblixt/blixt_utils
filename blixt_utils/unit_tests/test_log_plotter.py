import unittest
import numpy as np
import os, sys
import inspect
from bokeh.plotting import column, figure, show, curdoc
from bokeh.io import output_file
from bokeh.models import Slider, ColorPicker, Range1d, Span, CustomJS, CheckboxEditor, Text
from bokeh.models import (ColumnDataSource, DataTable, NumberEditor, SelectEditor, StringEditor, StringFormatter,
    IntEditor, TableColumn)
# from html5lib.constants import hexDigits


# To test blixt_rp and blixt_utils libraries directly, without installation:
project_dir = str(os.path.dirname(__file__).replace('blixt_utils\\blixt_utils\\unit_tests', ''))
sys.path.append(os.path.join(project_dir, 'blixt_rp'))
sys.path.append(os.path.join(project_dir, 'blixt_utils'))

from blixt_utils.plotting.log_plotter import LogPlotter, LogColumn, Line
import blixt_utils.plotting.log_plotter as bupl
from blixt_rp.core.core import Template
from blixt_rp.core.seismic import SeismicTraces

# output_file('C:\\Users\marte\Documents\plot.html')
output_file('C:\\Users\emb\Documents\plot.html')

line1 = Line(x=None, y=None, style=Template(**{'name': "Temp.", 'line_color': "blue", 'line_width': 2}))
line2 = Line(x=np.random.normal(10., 1., 500), y=np.linspace(1000., 2000., 500),
             style=Template(name = "D2", line_color = "red", line_width = 1))
line3 = Line(x=np.random.normal(100., 1., 500), y=np.linspace(1000., 2000., 500),
             style=Template(name = "D3", line_color = "yellow", line_width = 2))
seismic1 = SeismicTraces()


class SetUp(unittest.TestCase):
    def test_log_plotter1(self):
        lp = LogPlotter()
        print(lp.width)
        lp.width = 600
        print(lp.width)
        self.assertTrue(lp.width == 600)

    def test_line(self):
        # line1 = Line()

        p = figure(title="Multiple line example", x_axis_label="x", y_axis_label="y")
        p.line(line1.x, line1.y, **line1.line_args)
        self.assertIsInstance(line1, Line)

        show(p)

    def test_column_with_lines(self):
        c1 = LogColumn('c1',  lines=[line1, line2, line3])

        p = bupl.create_column_figure(c1, None, None, 300, 600, True, True, True, None)
        show(p)
        # print(inspect.getmembers(p.children[0].y_range))
        self.assertTrue(True)

    def test_two_column_plot(self):
        lp = LogPlotter(width=800, height=1000)
        c2 = LogColumn('c2',  lines=[line1, line2])
        c1 = LogColumn('c1',  lines=[line1, line2, line3], rel_width=2)
        lp.columns = [c2, c1]
        show(lp.figure())
        self.assertTrue(True)

    def test_two_column_plot_with_seismic(self):
        lp = LogPlotter(width=800, height=1000)
        c2 = LogColumn('c2',  lines=[line1, line2])
        c1 = LogColumn('c1',  seismic_traces=seismic1, rel_width=2)
        lp.columns = [c2, c1]
        show(lp.figure())
        self.assertTrue(True)

    def test_link_table(self):
        lp = LogPlotter(width=800, height=1000)
        c3 = LogColumn('c3', rel_width=0.5)
        c2 = LogColumn('c2',  lines=[line1, line2])
        c1 = LogColumn('c1',  lines=[line2, line3, line1], rel_width=2)
        lp.columns = [c2, c3, c1]
        p = lp.figure()

        print(p.children)

        data_table = bupl.add_strat_table(
            p,
            {'name': ['Viking GP', 'Draupne FM', 'Another FM'],
            'top': [1000., 1400., 1630.], 'base': [2600., 1630., 1900.], 'color': ['red', 'blue', 'green'],
             'level': [0, 1, 1]},
            width = 800,
            column_index=1
        )
        show(column(p, data_table))
        self.assertTrue(True)

    def test_python_interaction(self):
        # This script is called  by the main.py script under blixt_projects/bokeh_testing
        # And can be invoked by calling:
        # C:\Users\emb\Documents\PycharmProjects\blixt_projects>C:\Users\emb\Documents\PycharmProjects\venv\Scripts\bokeh serve --show bokeh_testing
        from scipy.signal import savgol_filter
        p = figure()
        source = ColumnDataSource(data=dict(
            x=line1.y,
            y1=line1.x,
            y2=savgol_filter(line1.x, 20, 1)
        ))
        p.line('x', 'y1', source=source)
        line = p.line('x', 'y2', source=source, legend_label="smooth.", color="red", line_width=2)

        slider1 = Slider(start=20, end=520, step=50, value=20)
        slider2 = Slider(start=1, end=5, step=1, value=1)

        def callback1(attr, old, new):
            source.data['y2'] = savgol_filter(line1.x, new, 1)

        def callback2(attr, old, new):
            print(attr, old, new)
            source.data['y2'] = savgol_filter(line1.x, slider1.value, new)

        slider1.on_change('value_throttled', callback1)
        slider2.on_change('value_throttled', callback2)

        return p, slider1, slider2
        # curdoc().add_root(column(p, slider))

if __name__ == '__main__':
    test = SetUp()
    _p, _slider = test.test_python_interaction()
