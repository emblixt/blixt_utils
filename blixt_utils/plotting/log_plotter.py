import numpy as np
from IPython.core.magics.code import extract_code_ranges
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import Slider, ColorPicker, Range1d, LinearAxis, Span, Legend
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool
from bokeh.io import output_file
from bokeh.layouts import gridplot

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    # CrosshairTool(),
    ResetTool(),
    SaveTool()
]

def get_legend_names(_lines: list):
    """

    :param _lines:
        list
        List of Line objects
    :return:
    """
    _legend_names = []
    for i, _line in enumerate(_lines):
        _name = None
        if 'legend_label' in list(_line.line_args.keys()):
            _name = _line.line_args.pop('legend_label')
        if _name is None:
            _name = 'D{}'.format(i)
        _legend_names.append(_name)
    return _legend_names


class LogPlotter:
    """
    Class for plotting multiple well logs, from one well, in different "columns", all with a common y-axis (time
    or depth)
    """

    def __init__(self,
                 width: int = 300,
                 height: int = 600,
                 columns: list |None = None
                 ):
        """

        :param width:
        :param height:
        :param columns:
            list
            list of LogColumn objects
        """
        if columns is None:
            columns = []
        self._width = width
        self._height = height
        self._columns = columns


    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, w):
        self._width = w

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, h):
        self._height = h

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, l: list):
        self._columns = l

    def column(self, _column):
        self._columns.append(_column)

    @property
    def column_width(self):
        n_cols = sum([_column.rel_width for _column in self.columns])
        return int(self.width / n_cols)

    def figure(self):
        children = []
        _w = Span(dimension="width", line_dash="dashed", line_width=1)
        _h = Span(dimension="height", line_dash="dashed", line_width=0)
        for i, _column in enumerate(self.columns):
            extra_axes = []
            _p = figure(width=int(_column.rel_width * self.column_width),
                        height=int(self.height * (1. - _column.rel_header_height)),
                        tools=tools)  #, active_inspect=None)
            _p.toolbar.logo = None
            _p.add_tools(CrosshairTool(overlay=[_w, _h]))
            _p.x_range = Range1d(*_column.lines[0].x_range)
            legend_names = get_legend_names(_column.lines)
            for j, _line in enumerate(_column.lines):
                _legend_label = '{:.1f} : {} : {:.1f}'.format(_line.x_range[0], legend_names[j], _line.x_range[1])
                if j == 0:
                    _p.line(_line.x, _line.y, legend_label=_legend_label, **_line.line_args)
                else:
                    _p.extra_x_ranges[legend_names[j]] = Range1d(*_line.x_range)
                    _p.line(_line.x, _line.y, **_line.line_args, legend_label=_legend_label,
                           x_range_name=legend_names[j])
                    extra_axes.append(LinearAxis(axis_label=legend_names[j], x_range_name=legend_names[j]))
            if i == 0:
                _p.y_range.flipped = True
                _p.xaxis.visible = False
            else:
                # _p.y_range = children[0].y_range
                _p.xaxis.visible = False
                _p.yaxis.visible = False

            _p.legend.click_policy = 'hide'
            _p.add_layout(_p.legend[0], 'above')
            _p.legend.label_text_font_size = '8pt'

            children.append(_p)

        # TODO
        # That this works means that I can separate out the creation of each column figure to a function / method
        for i, child in enumerate(children):
            if i > 0:
                child.y_range = children[0].y_range

        return gridplot([[_child for _child in children]],
                        toolbar_location='right', merge_tools=True)



class LogColumn:
    """
    A class that can contain multiple well logs (lines) in the same "column"
    """

    def __init__(self,
                 name: str,
                 rel_width: float = 1.,
                 rel_header_height: float = 0.1,
                 lines: list | None = None
                 ):
        """
        :param name:
            str
            Unique string that separates the different columns from each other
        :param rel_width:
            float
            Width relative to the width of one of N columns of equal width
            E.G. for a 400 pixel wide LogPlotter with 4 LogColumns with rel_width=1, each LogColumn will be 100 wide
        :param rel_header_height:
            float
            Height of header relative to the total height
        :param lines:
            list
            A list of dictionaries, where each dictionary represent a line (a log) that should be added to this column
            Each dictionary should contain the following keys; 'x', 'y', 'legend_label', 'color', 'line_width'
        :param y_range:
            Range1d
            Determines the y range of the column
        """
        if lines is None:
            lines = []
        self._lines = lines
        self._rel_width = rel_width
        self._rel_header_height = rel_header_height
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def rel_width(self):
        return self._rel_width

    @rel_width.setter
    def rel_width(self, rw):
        self._rel_width = rw

    @property
    def rel_header_height(self):
        return self._rel_header_height

    @rel_header_height.setter
    def rel_header_height(self, rh):
        self._rel_header_height = rh

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, l: list):
        for _line in l:
            if not isinstance(_line, Line):
                raise IOError('{} is not a Line object'.format(_line))
        self._lines = l

    def line(self, _line):
        if not isinstance(_line, Line):
            raise IOError('{} is not a Line object'.format(_line))
        self._lines.append(_line)


class Line:
    """
    Simple class for defining a line object
    """
    def __init__(self,
                 x=None,
                 y=None,
                 line_args=None,
                 x_range: tuple = (None, None)
    ):
        """

        :param x:
        :param y:
        :param line_args:
            dict
            Dictionary with arguments that goes to the bokeh.figure.line() method
        """
        if x is None:
            x = np. random.normal(0., 1., 1000)
        self._x = x
        if y is None:
            y = np.linspace(500, 3500, len(self._x))
        self._y = y
        if line_args is None:
            line_args = {}
        self._line_args = line_args
        self._x_range = x_range

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def line_args(self):
        return self._line_args

    @property
    def min(self):
        return np.nanmin(self._x)

    @property
    def max(self):
        return np.nanmax(self._x)

    @property
    def x_range(self):
        _min = self._x_range[0]
        _max = self._x_range[1]
        if _min is None:
            _min = self.min
        if _max is None:
            _max = self.max
        return _min, _max



