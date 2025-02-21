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
    CrosshairTool(),
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
    def column_data_source(self):
        _dict = None
        for i, _column in enumerate(self.columns):
            if i == 0:
                _dict = _column.column_data_source
            else:
                _dict.update(_column.column_data_source)
        return _dict


    def figure(self):
        return row([_column.figure() for _column in self.columns])



class LogColumn:
    """
    A class that can contain multiple well logs (lines) in the same "column"
    """

    def __init__(self,
                 tag: str,
                 parent: LogPlotter | None = None,
                 rel_width: float = 1.,
                 rel_header_height: float = 0.1,
                 lines: list | None = None,
                 y_range: Range1d | None =  None,
                 show_tools: bool = False
                 ):
        """
        :param tag:
            str
            Unique string that separates the different columns from each other
        :param parent:
            LogPlotter
            Parent log plotter that should contain this column plot
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
        self._y_range = y_range
        if parent is None:
            parent = LogPlotter()
        self._parent = parent
        self._tag = tag


        extra_axes = []
        if show_tools:
            _tools = tools
        else:
            _tools = ''
        p = figure(width=int(self._parent.width * self._rel_width),
                   height=int(self._parent.height * (1. - self._rel_header_height)),
                   tools=_tools, active_inspect=None)
        p.toolbar.logo = None
        p.x_range = Range1d(*self.lines[0].x_range)

        legend_names = get_legend_names(self.lines)

        for i, _line in enumerate(self.lines):
            _legend_label = '{:.1f} : {} : {:.1f}'.format(_line.x_range[0], legend_names[i], _line.x_range[1])
            if i == 0:
                p.line(_line.x, _line.y, legend_label=_legend_label, **_line.line_args)
            else:
                p.extra_x_ranges[legend_names[i]] = Range1d(*_line.x_range)
                p.line(_line.x, _line.y, **_line.line_args, legend_label=_legend_label,
                       x_range_name=legend_names[i])
                extra_axes.append(LinearAxis(axis_label=legend_names[i], x_range_name=legend_names[i]))

        if self.y_range is None:
            p.y_range.flipped = True
        elif isinstance(self.y_range, Range1d):
            p.y_range = self.y_range

        p.legend.click_policy = 'hide'
        p.add_layout(p.legend[0], 'above')
        p.legend.label_text_font_size='8pt'
        self._p = p

    def figure(self):
        return self._p

    @property
    def tag(self):
        return self._tag

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

    @property
    def y_range(self):
        return self._y_range

    @property
    def column_data_source(self):
        _dict = {}
        for i, _line in enumerate(self.lines):
            _dict['{}x{}'.format(self.tag, i)] = _line.x
            _dict['{}y{}'.format(self.tag, i)] = _line.y
        return _dict



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



