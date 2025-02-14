from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import Slider, ColorPicker
from bokeh.io import output_file
from bokeh.layouts import gridplot

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

    def create_line_plot(self, x, y, title="Line Plot", x_label="X", y_label="Y"):
        plot = figure(width=self.width, height=self.height, title=title, x_axis_label=x_label, y_axis_label=y_label)
        plot.line(x, y, line_width=2, color='navy')
        return plot

    def create_scatter_plot(self, x, y, title="Scatter Plot", x_label="X", y_label="Y"):
        plot = figure(width=self.width, height=self.height, title=title, x_axis_label=x_label, y_axis_label=y_label)
        plot.circle(x, y, size=8, color='firebrick', alpha=0.5)
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

    def show_plot(self, plot, output_filename="bokeh_plot.html"):
        output_file(output_filename)
        show(plot)

