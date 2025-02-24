import bokeh.plotting
import numpy as np
from pandas import DataFrame
from IPython.core.magics.code import extract_code_ranges
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, Span, Legend, ColumnDataSource, Text,
                          CustomJS)
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool
from bokeh.models import (DataTable, NumberEditor, SelectEditor, StringEditor, StringFormatter,
                          IntEditor, TableColumn, CheckboxEditor)
from bokeh.io import output_file
from bokeh.layouts import gridplot

# from blixt_utils.misc.templates import necessary_keys

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    # CrosshairTool(),
    ResetTool(),
    SaveTool()
]

def get_legend_names(_lines: list) -> list:
    """

    :param _lines:
        list
        List of Line objects
    :return:
        list
        List of legend names
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

    def figure(self) -> gridplot:
        children = []
        _w = Span(dimension="width", line_dash="dashed", line_width=1)
        _h = Span(dimension="height", line_dash="dashed", line_width=0)
        for i, _column in enumerate(self.columns):
            _p = create_column_figure(_column, _w, _h, self.column_width, self.height,
                                      _y_range_flipped=i==0,
                                      _x_axis_visible=True,
                                      _y_axis_visible=i==0,
                                      _tools=tools)
            children.append(_p)

        # Let the y-axis of all columns be controlled by the y-axis of the first column
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


def create_column_figure(_column: LogColumn,
                         _w: Span | None,
                         _h: Span | None,
                         _width: int,
                         _height: int,
                         _y_range_flipped: bool,
                         _x_axis_visible: bool,
                         _y_axis_visible: bool,
                         _tools: list | None) -> bokeh.plotting.figure:
    """

    :param _column:
        LogColumn
    :param _w:
        Span
        Width setting for the shared crosshair
    :param _h:
        Span
        Height setting for the shared crosshair
    :param _width:
        int
        Width of a default column in pixels
    :param _height:
        int
        height of figure in pixels
    :param _y_range_flipped:
        bool
        Set to True to flip the y-axis range
    :param _x_axis_visible:
        bool
        Set to True to show the x-axis
    :param _y_axis_visible:
        bool
        Set to True to show the y-axis
    :param _tools:
        list
        List of tools included in the toolbar, Except for CrossHairTool, which is added separately
    :return:
    """
    if _w is None:
        _w = Span(dimension="width", line_dash="dashed", line_width=1)
    if _h is None:
        _h = Span(dimension="height", line_dash="dashed", line_width=1)
    if _tools is None:
        _tools = "pan,wheel_zoom,box_zoom,reset"

    _p = figure(width=int(_column.rel_width * _width),
                height=int(_height * (1. - _column.rel_header_height)),
                x_axis_location = 'above',
                tools=_tools)  # , active_inspect=None)
    _p.toolbar.logo = None
    _p.add_tools(CrosshairTool(overlay=[_w, _h]))
    _p.x_range = Range1d(*_column.lines[0].x_range)

    add_lines(_p, _column)

    _p.y_range.flipped = _y_range_flipped
    _p.xaxis.visible = _x_axis_visible
    _p.yaxis.visible = _y_axis_visible

    _p.legend.click_policy = 'hide'
    # _p.add_layout(_p.legend[0], 'above')
    _p.legend.location = 'top_right'
    _p.legend.label_text_font_size = '8pt'

    return _p


def add_lines(_p: bokeh.plotting.figure,
              _column: LogColumn,
              ):
    """
    Add lines to the column figure
    :param _p:
    :param _column:
    :return:
    """
    legend_names = get_legend_names(_column.lines)
    extra_axes = []
    for j, _line in enumerate(_column.lines):
        # Create new legend that includes the range of the data
        _legend_label = '{:.1f} : {} : {:.1f}'.format(_line.x_range[0], legend_names[j], _line.x_range[1])
        if j == 0:
            _p.line(_line.x, _line.y, legend_label=_legend_label, **_line.line_args)
        else:
            _p.extra_x_ranges[legend_names[j]] = Range1d(*_line.x_range)
            _p.line(_line.x, _line.y, **_line.line_args, legend_label=_legend_label,
                    x_range_name=legend_names[j])
            this_ax = LinearAxis(axis_label=legend_names[j], x_range_name=legend_names[j], axis_label_text_font_size='10px')
            extra_axes.append(this_ax)
    for _ax in extra_axes:
        _p.add_layout(_ax, 'above')


def add_strat_table(_p: bokeh.plotting.figure,
                    stratigraphy: dict,
                    width: int | None = None) -> bokeh.models.DataTable:
    """
    Adds the stratigraphic levels in the stratigraphy dictionary to the plot, and adds a table that
    control them.
    :param _p:
        Figure to add the stratigraphy too
    :param stratigraphy:
        Dictionary with at least these key : value pairs:
            'name' : list with the name of each level
            'top' : list of depths [m MD] to the top of each level
        Optional key : value pairs
            'base' : List of depths [m MD] to the base of each level
            'visible' : List of bool
            'level' : List of levels (e.g. 'Group', 'Formation', 'Member')
            'color' :
            'line_style' :
            'line_width' : List of integers
            'font_size' : List of strings, e.g. ['10px', ...]
    :return:
    """
    if width is None:
        width = 600
    code = """
        //Try to loop over all spans 
        const data = source.data;
        for (var i = 0; i < spans.length; i++) {
            const boolValue = data['visible'][i] === 1 ? true : false;
            spans[i].location = data['top'][i];
            spans[i].line_color = data['color'][i];
            spans[i].visible = boolValue;
            spans[i].line_dash = data['line_style'][i];
            spans[i].line_width = data['line_width'][i];
            //console.log('Data changed:', boolValue);
        };
        source.change.emit();
        """
    _necessary_keys = ['name', 'top']
    _optional_keys = ['base', 'visible', 'level', 'color', 'line_style', 'line_width', 'font_size']
    _current_keys = list(stratigraphy.keys())
    for _key in _necessary_keys:
        if _key not in _current_keys:
            raise IOError('Necessary key {} is lacking in stratigraphy dictionary'.format(_key))

    for _key in _optional_keys:
        if _key not in _current_keys:
            if _key == 'base':
                stratigraphy[_key] = stratigraphy['top']
            elif _key == 'visible':
                stratigraphy[_key] = [True] * len(stratigraphy['top'])
            elif _key == 'level':
                stratigraphy[_key] = ['Formation'] * len(stratigraphy['top'])
            elif _key == 'color':
                stratigraphy[_key] = ['blue'] * len(stratigraphy['top'])
            elif _key == 'line_style':
                stratigraphy[_key] = ['solid'] * len(stratigraphy['top'])
            elif _key == 'line_width':
                stratigraphy[_key] = [2] * len(stratigraphy['top'])
            elif _key == 'font_size':
                stratigraphy[_key] =['15px'] * len(stratigraphy['top'])

    stratigraphy_frame = DataFrame(stratigraphy)

    names = sorted(stratigraphy_frame['name'].unique())
    levels = sorted(stratigraphy_frame['level'].unique())
    # line_widths = [1, 2, 3, 4, 5]
    line_widths = ['1', '2', '3', '4', '5']
    line_styles = ['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']

    table_columns = [
        TableColumn(field='visible', title='Show',
                    editor=CheckboxEditor(),
                    width=30),
        TableColumn(field='name', title='Stratigraphy',
                    editor=SelectEditor(options=names),
                    formatter=StringFormatter(font_style='bold')),
        TableColumn(field='level', title='Level',
                    editor=SelectEditor(options=levels)),
        TableColumn(field='top', title='Top [m MD]',
                    editor=NumberEditor()),
        TableColumn(field='base', title='Base [m MD]',
                    editor=NumberEditor()),
        TableColumn(field='color', title='Color',
                    editor=StringEditor()),
        TableColumn(field='line_style', title='Line style',
                    editor=SelectEditor(options=line_styles)),
        TableColumn(field='line_width', title='Line width',
                    # editor=IntEditor()),
                    editor = SelectEditor(options=line_widths)),
    TableColumn(field='font_size', title='Font size',
                    editor=StringEditor())

    ]

    strati_source = ColumnDataSource(stratigraphy_frame)

    def spans():
        _tmp = []
        for _i, _name in enumerate(strati_source.data['name']):
            _tmp.append(Span(location=strati_source.data['top'][_i], dimension='width',
                             line_width=strati_source.data['line_width'][_i],
                             line_dash=strati_source.data['line_style'][_i],
                             line_color=strati_source.data['color'][_i]))
        return _tmp

    text_glyph = None
    _spans = spans()
    for _span in _spans:
        for i, _child in enumerate(_p.children):
            _child[0].add_layout(_span)
            if i == 0:
                text_glyph = Text(x=_child[0].x_range.start, y='top',
                                  text='name', text_font_size='font_size', text_alpha='visible')
                _child[0].add_glyph(strati_source, text_glyph)

    span_callback = CustomJS(args=dict(source=strati_source, spans=_spans), code=code)

    strati_source.js_on_change('patching', span_callback)  # 'patching' is important!

    return DataTable(
        source=strati_source,
        columns=table_columns,
        editable=True,
        width=width,
        index_position=-1,
        index_header='row index',
        index_width=60)

