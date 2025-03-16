from bokeh.plotting import figure, show
from bokeh.models import Line, NumberEditor
from bokeh.layouts import row, column
from bokeh.models import Slider, ColorPicker
from bokeh.io import output_file
from bokeh.layouts import gridplot
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, Span, Legend, ColumnDataSource, Text,
                          Rect, CustomJS)
import numpy as np

from AkerBP_PL1099.NordenskioldAVO import plot_CGG22M05_traces_above_Nordenskiold

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    CrosshairTool(),
    ResetTool(),
    SaveTool()
]

output_file('C:\\Users\emb\Documents\plot.html')

# prepare some data
x = [1, 2, 3, 4, 5]
y1 = [6, 7, 2, 4, 5]
y2 = [2, 3, 4, 5, 6]
y3 = [4, 5, 5, 7, 2]

def first_step_one():
    # create a new plot with a title and axis labels
    p = figure(title="Multiple line example", x_axis_label="x", y_axis_label="y")

    # add multiple renderers
    p.line(x, y1, legend_label="Temp.", color="blue", line_width=2)
    p.line(x, y2, legend_label="Rate", color="red", line_width=2)
    p.line(x, y3, legend_label="Objects", color="green", line_width=2)

    # show the results
    show(p)

def test_rectangles():
    # Create figure
    p = figure(width=600, height=400, x_axis_location='above')
    N = 3
    y = np.linspace(-2, 2, N)
    x = 0
    w = 3
    h = 2

    source = ColumnDataSource(dict(y=y))

    glyph = Rect(x=x, y="y", width=w, height=h, angle=0, fill_color="#cab2d6")
    p.add_glyph(source, glyph)
    print(p.width, p.height)
    show(p)


def test_spacing():
    # Create figure
    p = figure(width=600, height=400, x_axis_location='above')

    # Primary axis
    p.xaxis.axis_label = "Primary Axis"
    p.xaxis.axis_label_standoff = 20  # Increase space between label and axis

    # Secondary axis
    p.extra_x_ranges = {"x2": Range1d(start=0, end=100)}
    secondary_axis = LinearAxis(x_range_name="x2", axis_label="Secondary Axis", axis_label_standoff=40)
    p.add_layout(secondary_axis, 'above')

    # Plot data
    p.line([10, 20, 30, 40], [1, 2, 3, 4],  color="blue")
    p.line( [15, 30, 45, 60], [1, 2, 3, 4], color="red", x_range_name="x2")

    show(p)

def test_callback():
    from bokeh.models import ColumnDataSource, DataTable, Span, CustomJS, TableColumn

    source = ColumnDataSource(data={'x': [1, 2, 3], 'y': [4, 5, 6]})
    table_columns = [
        TableColumn(field='x', title='X',
                    editor=NumberEditor()),
        TableColumn(field='y', title='Y',
                    editor=NumberEditor())
        ]

    p = figure()
    p.line([0, 10], [0, 10])
    span = Span(location=0, dimension='width', line_color='red', line_dash='dashed')

    # source.data.update({'x': [4, 5, 6], 'y': [7, 8, 9]})

    callback = CustomJS(args=dict(source=source, span=span), code="""
        const data = source.data;
        console.log('Data changed:', data);
        span.location = data['y'][0];  
        source.change.emit(); // Update span location based on first row of 'y' column
    """)
    source.js_on_change('patching', callback)

    data_table = DataTable(source=source, columns=table_columns, editable=True)  # Define your columns

    p.add_layout(span)
    show(column(p, data_table))




class BokehPlotter:
    # Usage example:
    # plotter = BokehPlotter(width=500, height=300)
    # x = [1, 2, 3, 4, 5]
    # y = [6, 7, 2, 4, 5]
    # line_plot = plotter.create_line_plot(x, y, title="My Line Plot")
    # plotter.show_plot(line_plot)

    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height

    def create_line_plot(self, _x, y, title="Line Plot", x_label="X", y_label="Y"):
        plot = figure(width=self.width, height=self.height, title=title, x_axis_label=x_label, y_axis_label=y_label)
        plot.line(_x, y, line_width=2, color='navy')
        return plot

    def create_scatter_plot(self, _x, y, title="Scatter Plot", x_label="X", y_label="Y"):
        plot = figure(width=self.width, height=self.height, title=title, x_axis_label=x_label, y_axis_label=y_label)
        plot.circle(_x, y, size=8, color='firebrick', alpha=0.5)
        return plot

    def create_two_column_plot(self, data_list, titles, x_labels, y_labels):
        plots = []
        for i, data in enumerate(data_list):
            p = figure(width=self.width, height=self.height, title=titles[i],
                       x_axis_label=x_labels[i], y_axis_label=y_labels[i])
            line = p.line(data['x'], data['y'], line_width=2, color='navy')
            picker = ColorPicker(title='Line color')
            picker.js_link('color', line.glyph, 'line_color')
            slider = Slider(start=1, end=10, step=0.2, value=2)
            slider.js_link('value', line.glyph, 'line_width')
            plots.append(column(p, row(picker, slider)))

        grid = gridplot(plots, ncols=2)
        return grid

    def linked_plots(self):
        _x = list(range(11))
        _y0 = _x
        _y1 = [10 - xx for xx in _x]
        _y2 = [abs(xx - 5) for xx in _x]

        # create a new plot
        s1 = figure(width=250, height=250, title=None, tools=tools)
        s1.scatter(_x, _y0, size=10, color="navy", alpha=0.5)

        # create a new plot and share both ranges
        # s2 = figure(width=250, height=250, x_range=s1.x_range, y_range=s1.y_range, title=None, tools=tools)
        s2 = figure(width=250, height=250, x_range=s1.x_range, title=None, tools=tools)
        s2.scatter(_x, _y1, size=10, color="firebrick", alpha=0.5, marker='triangle')
        s2.y_range = s1.y_range

        # create a new plot and share only one range
        s3 = figure(width=250, height=250, x_range=s1.x_range, title=None)
        s3.scatter(_x, _y2, size=10, color="olive", alpha=0.5, marker='square')

        p = gridplot([[s1, s2, s3]], toolbar_location='right', merge_tools=True)
        return p

    def show_plot(self, plot, output_filename="bokeh_plot.html"):
        output_file(output_filename)
        show(plot)


if __name__ == '__main__':
    # first_step_one()
    # plotter = BokehPlotter()
    # lps = plotter.linked_plots()
    # plotter.show_plot(lps)
    test_rectangles()
