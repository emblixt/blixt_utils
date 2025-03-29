import matplotlib as mpl
from copy import deepcopy

import bokeh.plotting
import numpy as np
from openpyxl.styles.builtins import title
from pandas import DataFrame
from typing import Literal
# from IPython.core.magics.code import extract_code_ranges
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, Spacer
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, LogAxis,  Span, Legend, ColumnDataSource, Text,
                          CustomJS, CustomJSTransform, LinearColorMapper)
from bokeh.models import PanTool,WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, ColorBar, LogColorMapper
from bokeh.models import (DataTable, NumberEditor, SelectEditor, StringEditor, StringFormatter,
                          IntEditor, TableColumn, CheckboxEditor, Rect)
from bokeh.io import output_file
from bokeh.layouts import gridplot
from bokeh.transform import transform

import bruges

# from blixt_utils.misc.templates import necessary_keys
from blixt_rp.core.core import Template

tools = [
    PanTool(),
    WheelZoomTool(),
    HoverTool(),
    # CrosshairTool(),
    ResetTool(),
    SaveTool()
]

test_data_length = 1000


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

    def add_column(self, _column, keep_column_width=True):
        if keep_column_width:
            old_width = self.width
            old_column_width = self.column_width
            self.width = old_width + _column.rel_width * old_column_width
        self._columns.append(_column)

    def insert_column(self, _i, _column, keep_column_width=True):
        if keep_column_width:
            old_width = self.width
            old_column_width = self.column_width
            self.width = old_width + _column.rel_width * old_column_width
        self._columns.insert(_i, _column)

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
                                      _x_axis_visible=len(_column) > 0,
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
                 rel_header_height: float = 0.,
                 lines: list | None = None,
                 seismic_traces: list | None = None,
                 scale: str = 'linear'
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
            A list of Line objects
        :param seismic_traces:
            list
            A list of SeismicTraces objects
        :param scale:
            str
            'linear' or 'log'
        """
        if lines is None:
            lines = []
        self._lines = lines
        if seismic_traces is None:
            seismic_traces = []
        self._seismic_traces = seismic_traces
        self._rel_width = rel_width
        self._rel_header_height = rel_header_height
        self._name = name
        if scale is None:
            scale = 'linear'
        self.scale = scale

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

    def add_line(self, _line):
        if not isinstance(_line, Line):
            raise IOError('{} is not a Line object'.format(_line))
        self._lines.append(_line)

    @property
    def seismic_traces(self):
        return self._seismic_traces

    @seismic_traces.setter
    def seismic_traces(self, l: list):
        for _st in l:
            if not isinstance(_st, SeismicTraces):
                raise IOError('{} is not a SeismicTraces object'.format(_st))
        self._seismic_traces = l

    def add_seismic_traces(self, _st):
        if not isinstance(_st, SeismicTraces):
            raise IOError('{} is not a SeismicTraces object'.format(_st))
        self._lines.append(_st)

    def __len__(self):
        return max([len(self.lines), len(self.seismic_traces)])

class Line:
    """
    Simple class for defining a line object
    """
    def __init__(self,
                 x=None,
                 y=None,
                 style: Template | None = None,
                 source: ColumnDataSource | None = None
    ):
        """

        :param x:
        :param y:
        :param style:
            Template
            Template object which we use to create the line arguments that goes to the bokeh.figure.line() method.
            Most important arguments are:
            line_color='black', line_style='solid,  line_width=1, name='name'
        :param source:
            ColumnDataSource that can be updated interactively in bokeh.
            If given, the x and y must be strings that exists within this source. E.G.
            > source = ColumnDataSource(dict(alpha=<some data>, beta=<some other data>))
            > Line(x='alpha', y='beta', source=source)
        """
        if x is None:
            x = np. random.normal(0., 1., test_data_length)
        self._x = x
        if y is None:
            y = np.linspace(500, 3500, len(self._x))
        self._y = y
        if style is None:
            style = Template(**dict(line_color='black', line_style = 'solid', line_width = 2, name = 'TEST',
                                    min=1000., max=13000.))
        self.style = style
        self.source = source

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def line_args(self):
        # handle linestyle
        if self.style.line_style == '-':
            line_dash = 'solid'
        elif self.style.line_style == '--':
            line_dash = 'dashed'
        elif self.style.line_style == ':':
            line_dash = 'dotted'
        elif self.style.line_style == '-.':
            line_dash = 'dotdash'
        else:
            line_dash = 'solid'

        _d = {'line_color': 'blue' if self.style.line_color is None else self.style.line_color,
              'line_width': 1.0 if self.style.line_width is None else self.style.line_width,
              'line_dash': line_dash,
              'legend_label': self.style.name}
        return _d

    @property
    def min(self):
        if self.source is None:
            return np.nanmin(self.x)
        else:
            return np.nanmin(self.source.data[self.x])

    @property
    def max(self):
        if self.source is None:
            return np.nanmax(self._x)
        else:
            return np.nanmax(self.source.data[self.x])

    def x_range(self, from_style=False):
        _min = None
        _max = None
        if from_style:
            if 'min' in list(self.style.keys()):
                _min = self.style.min
            if 'max' in list(self.style.keys()):
                _max = self.style.max

        if _min is None:
            _min = self.min
        if _max is None:
            _max = self.max
        return _min, _max


class SeismicTraces:
    """
    Simple class that contains N number of seismic traces,
    Each trace must have the same depth/time dimension

    """
    def __init__(self,
                 x: np.ndarray | None = None,
                 y: np.ndarray | None = None,
                 traces: np.ndarray | None = None,
                 trace_type: Literal['avo', 'eei', 'index'] | None = None
                 ):
        """

        :param x:
            np.ndarray of length M with values in the x direction
            Contains incidence angles when trace_type is 'avo',
            Chi angles when trace_type is 'eei', and index when trace_type is 'index'
        :param y:
            Numpy array of depth/time values, length N
        :param traces:
            np.ndarray of size MxN, contains M number of seismic traces, each of length N
        :param trace_type:
            string that describes the type of seismic variation in the x direction
            'avo': Incidence angle (AVO)
            'eei': Chi angle (Extended Elastic Impedance)
            'index': index along an Inline, Xline or arbitrary seismic line
        """
        if x is None:
            x = np.linspace(0,35, 128)
        self._x = x
        if y is None:
            y = np.linspace(500, 3500, test_data_length)
        self._y = y
        if trace_type is None:
            trace_type = 'avo'
        self._trace_type = trace_type

        if traces is None:
            _synts = SyntheticTraces(trace_length=test_data_length, n_traces=len(x))
            traces = _synts.get_traces(simulate_avo=self._trace_type=='avo')
        self._traces = traces

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def trace_type(self):
        return self._trace_type

    @property
    def traces(self):
        return self._traces

class SyntheticTraces:
    def __init__(self,
                 trace_length:int | None = None,
                 dt: float | None = None,
                 n_traces: int | None = None,
                 n_reflectors: int | None = None,
                 seed: int | None = None):

        if trace_length is None:
            trace_length = 3000
        if dt is None:
            dt = 0.001  # seconds
        if n_traces is None:
            n_traces = 128
        if n_reflectors is None:
            n_reflectors = 3

        x = np.linspace(0, n_traces - 1, n_traces)
        y = np.linspace(0, trace_length - 1, trace_length)
        traces = np.zeros((n_traces, trace_length))

        self.trace_length = trace_length
        self.dt = dt
        self.n_traces = n_traces
        self.seed = seed
        self.n_reflectors = n_reflectors
        self.rng = np.random.default_rng(self.seed)

        self.w_length = 0.082  # Ricker wavelength in seconds
        self.f0 = 25.  # Hz

        self.traces = traces
        self.wavelet = None

    def get_traces(self, simulate_avo=False) -> np.ndarray:
        idx_refl = self.rng.integers(100, self.trace_length, self.n_reflectors)
        refl = np.zeros(self.trace_length)
        refl[idx_refl] = 2 * self.rng.random(self.n_reflectors) - 1

        # Convolve reflectivity model with a Ricker wavelet '''
        this_wavelet = bruges.filters.wavelets.ricker(self.w_length, self.dt, self.f0)
        self.wavelet = this_wavelet.amplitude
        this_trace = np.convolve(refl, self.wavelet, mode='same')
        for i in range(self.traces.shape[0]):
            if simulate_avo:
                _refl = deepcopy(refl)
                _refl[idx_refl[1]] = _refl[idx_refl[1]] * i*2/(self.traces.shape[0] - 1)
                this_trace=  np.convolve(_refl, self.wavelet, mode='same')
            self.traces[i,:] = this_trace

        return self.traces

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
    Creates a figure for one column

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
                x_axis_type = _column.scale,
                tools=_tools)  # , active_inspect=None)
    # style the plot
    _p.toolbar.logo = None
    _p.add_tools(CrosshairTool(overlay=[_w, _h]))
    hover = _p.select(dict(type=HoverTool))
    hover.tooltips = [("(x,y)", "($x, $y)"), ("Value", "@value")]

    if len(_column.lines) > 0:
        _p.x_range = Range1d(*_column.lines[0].x_range(from_style=True))

    _p.xaxis.visible = _x_axis_visible
    _p.xaxis.axis_label_text_font_size='10px'
    _p.xaxis.major_label_text_font_size='10px'
    _p.xaxis.axis_label_standoff=0

    # Add lines with log curves, if they exist in this column
    add_lines(_p, _column)

    # Add seismic, if it exists in this column
    add_seismic_traces(_p, _column, 0)

    _p.y_range.flipped = _y_range_flipped
    _p.yaxis.visible = _y_axis_visible

    # add legends if they exists
    if len(_p.legend) > 0:
        _p.legend.click_policy = 'hide'
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
    extra_axes = []
    for j, _line in enumerate(_column.lines):
        _legend_label = _line.line_args['legend_label']
        if j == 0:
            if _line.source is None:
                _p.line(x=_line.x, y=_line.y,  **_line.line_args)
            else:
                _p.line(x=_line.x, y=_line.y, source=_line.source, **_line.line_args)
            _p.xaxis.axis_label = _legend_label
            _p.x_range = Range1d(*_line.x_range(from_style=True))
            # print('XXX', _p.x_range.start, _p.x_range.end)
        else:
            _p.extra_x_ranges[_legend_label] = Range1d(*_line.x_range(from_style=True))
            if _line.source is None:
                _p.line(x=_line.x, y=_line.y, **_line.line_args, x_range_name=_legend_label)
            else:
                _p.line(x=_line.x, y=_line.y, source=_line.source, **_line.line_args, x_range_name=_legend_label)
            if _column.scale == 'log':
                this_ax = LogAxis(axis_label=_legend_label, x_range_name=_legend_label,
                                     axis_label_text_font_size='10px',
                                     major_label_text_font_size = '10px',  # )  # ,
                                     axis_label_standoff=0)  # this only adds space between new axes and its label
            else:
                this_ax = LinearAxis(axis_label=_legend_label, x_range_name=_legend_label,
                                     axis_label_text_font_size='10px',
                                     major_label_text_font_size = '10px',  # )  # ,
                                     axis_label_standoff=0)  # this only adds space between new axes and its label
            extra_axes.append(this_ax)
    for _ax in extra_axes:
        _p.add_layout(_ax, 'above')


def add_seismic_traces(_p: bokeh.plotting.figure,
                _column: LogColumn,
                _index: int = 0):
    """
    Similar to add_line(), but adds the seismic traces of index _index to the given column _column
    :param _p:
    :param _column:
    :param _index:
    :return:
    """
    trace_source = None
    _seismic_color_map = None
    for j, _seismic in enumerate(_column.seismic_traces):
        if j != _index:
            continue
        trace_source = ColumnDataSource({'value': [_seismic.traces.T]})
        _seismic_color_map = seismic_color_map(min_val=np.min(_seismic.traces), max_val=np.max(_seismic.traces))
    if trace_source is not None:
        # _p.image('value', source=trace_source, color_mapper=_seismic_color_map, dh=test_data_length, dw=128,
        #          x=0, y=0)
        _p.image('value', source=trace_source, color_mapper=_seismic_color_map,
                 dh=np.max(_column.seismic_traces[_index].y) - np.min(_column.seismic_traces[_index].y),
                 dw=np.max(_column.seismic_traces[_index].x) - np.min(_column.seismic_traces[_index].x),
                 x=np.min(_column.seismic_traces[_index].x),
                 y=np.min(_column.seismic_traces[_index].y)
                 )

        # add color bar
        color_bar = ColorBar(color_mapper=_seismic_color_map,
                             title='Seismic',
                             title_text_align = 'right',
                             label_standoff=3, major_label_text_font_size='10px')
        _p.add_layout(color_bar, 'above')


def add_strat_table(_p: bokeh.plotting.figure,
                    stratigraphy: dict,
                    width: int | None = None,
                    column_index: int | None = None) -> bokeh.models.DataTable:
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
    :param width:
        width in pixels
    :param column_index:
        int or None
        The figure _p has several columns, or children. If column_index is not None, it plots the intervals in this
        column
    :return:
    """
    if width is None:
        width = 600
    code = """
        //Try to loop over all spans 
        const data = source.data;
        const mid_points = new Float64Array(data['top'].length);
        const heights = new Float64Array(data['top'].length);
        for (var i = 0; i < spans.length; i++) {
            const boolValue = data['visible'][i] === 1 ? true : false;
            spans[i].location = data['top'][i];
            spans[i].line_color = data['color'][i];
            spans[i].visible = boolValue;
            spans[i].line_dash = data['line_style'][i];
            spans[i].line_width = data['line_width'][i]; 
            mid_points[i] = 0.5 * (data['top'][i] + data['base'][i])
            heights[i] = data['base'][i] - data['top'][i]
        };
        source.data['mid'] = mid_points
        source.data['height'] = heights
        console.log('Data changed:', data['mid'][i]);
        source.change.emit();
        """
    mid_point_func = """
        // Find midpoint of each interval
        const top = top_data;
        const base = base_data;
        const mid_point = new Float64Array(top.length);
    for (var i = 0; i < top.length; i++) {
        mid_point[i] = 0.5 * (top[i] + base[i])
    };
    return mid_point
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
                stratigraphy[_key] = [1 if np.mod(i, 2) == 0 else 2 for i in range(len(stratigraphy['top']))]
            elif _key == 'color':
                stratigraphy[_key] = ['blue'] * len(stratigraphy['top'])
            elif _key == 'line_style':
                stratigraphy[_key] = ['solid'] * len(stratigraphy['top'])
            elif _key == 'line_width':
                stratigraphy[_key] = [2] * len(stratigraphy['top'])
            elif _key == 'font_size':
                stratigraphy[_key] =['15px'] * len(stratigraphy['top'])

    # create extra source items
    stratigraphy['mid'] = list(0.5 * (np.array(stratigraphy['top']) + np.array(stratigraphy['base'])))
    stratigraphy['height'] = list(np.array(stratigraphy['base']) - np.array(stratigraphy['top']))

    stratigraphy_frame = DataFrame(stratigraphy)

    names = sorted(stratigraphy_frame['name'].unique())
    levels = sorted(stratigraphy_frame['level'].unique())
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
                    editor=IntEditor(step=1)),
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

    # mid_point = CustomJSTransform(func=mid_point_func)  # the bokeh transform method seems to only handle one input variable!

    def spans():
        _tmp = []
        for _i, _name in enumerate(strati_source.data['name']):
            _tmp.append(Span(location=strati_source.data['top'][_i], dimension='width',
                             line_width=strati_source.data['line_width'][_i],
                             line_dash=strati_source.data['line_style'][_i],
                             line_color=strati_source.data['color'][_i]))
        return _tmp

    def strati_units():
        if column_index is not None:
            for _i, _name in enumerate(strati_source.data['name']):
                glyph = Rect(x="level", y="mid", width=1, height="height", angle=0, fill_color="color")
                # glyph = Rect(x=0, y=transform('top', 'base'), width=10, height=100, angle=0, fill_color="color")
                _p.children[column_index][0].add_glyph(strati_source, glyph)

    strati_units()

    text_glyph = None
    _spans = spans()
    for _span in _spans:
        for i, _child in enumerate(_p.children):
            _child[0].add_layout(_span)
            if i == 0:
                text_glyph = Text(x=_child[0].x_range.start, y='top',
                                  text='name', text_font_size='font_size', text_alpha='visible')
                _child[0].add_glyph(strati_source, text_glyph)
        if column_index is not None:
            for i, _child in enumerate(_p.children):
                if i == column_index:
                    text_glyph = Text(x='level', y='mid',
                                      text='name', text_font_size='font_size', text_font_style='bold',
                                      text_alpha=1, text_align='center', text_baseline='middle',
                                      angle=90., angle_units='deg')
                    # text_glyph.level = 'overlay'
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


def seismic_color_map(min_val=-1, max_val=1, n=256, symmetric=True) -> bokeh.models.mappers.LinearColorMapper:
    # Construct cmap dictionary
    c_dict = {'red': ((0, 0.6314, 0.6314),
                     (0.33, 0, 0),
                     (0.4, 0.302, 0.302),
                     (0.5, 0.8, 0.8),
                     (0.6, 0.3804, 0.3804),
                     (0.667, 0.749, 0.749),
                     (1, 1, 1)),
             'green': ((0, 1, 1),
                       (0.33, 0, 0),
                       (0.4, 0.302, 0.302),
                       (0.5, 0.8, 0.8),
                       (0.6, 0.2706, 0.2706),
                       (0.667, 0, 0),
                       (1, 1, 1)),
             'blue': ((0, 1, 1),
                      (0.33, 0.749, 0.749),
                      (0.4, 0.302, 0.302),
                      (0.5, 0.8, 0.8),
                      (0.6, 0, 0),
                      (0.667, 0, 0),
                      (1, 0, 0))}

    if symmetric:
        if np.sign(min_val) != np.sign(max_val):
            max_magnitude = np.max([np.abs(min_val), np.abs(max_val)])
            min_val = np.sign(min_val) * max_magnitude
            max_val = np.sign(max_val) * max_magnitude

    cmap = mpl.colors.LinearSegmentedColormap('Seismic', c_dict).reversed()
    _colors = cmap(np.linspace(0, 1, n))
    return LinearColorMapper(palette=[mpl.colors.to_hex(_c) for _c in _colors], low=min_val, high=max_val)
