# ----------------------------------------------------------------------------------------------------------
# Filename: cross_plotter.py
#  Purpose: Cross plot data using the bokeh interactive plotting library
#           Similar to crossplot.py, but tried to modernize it
#
# ---------------------------------------------------------------------------------------------------
from bokeh.models import (Slider, ColorPicker, Range1d, LinearAxis, LogAxis,  Span, Legend, ColumnDataSource, Text,
                          CustomJS, CustomJSTransform, LinearColorMapper)
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, CrosshairTool, HoverTool, TextInput
from bokeh.plotting import figure, show

import unittest
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

default_tools = [
    PanTool(),
    WheelZoomTool(),
    BoxZoomTool(),
    HoverTool(),
    CrosshairTool(),
    ResetTool(),
    SaveTool()
]

cnames = [tmp['color'] for tmp, j in zip(plt.rcParams['axes.prop_cycle'], range(20))]
markers = ['circle', 'diamond', 'hex', 'inverted_triangle', 'plus', 'square', 'star', 'triangle']


class DataSource:
    """
    Class containing a dictionary of data, E.G. for one well, which can easily generate a ColumnDataSource
    But can hold some extra attributes and methods that the ColumnDataSource itself cant have
    """
    def __init__(self,
                 name: None | str = None,
                 data: None | dict = None,
                 templates: None | dict = None,
                 ):
        """
        Container for data related to one well
        :param name:
            str
            Name of  well
        :param data:
            dict
            Dictionary of data, with variable names as keys.
            Remember that a ColumnDataSource requires that all variable has the same length
        :param templates:
            dict
            Dictionary of Template objects for each variable, and preferably also one Template for this
            DataSource itself.
        """
        from blixt_rp.core.core import Template
        self._name = name
        self._data = data
        if templates is None:
            templates = {}
        if data is not None:
            for _key in list(data.keys()):
                if _key not in list(templates.keys()):
                    templates[_key] = None
        self._templates = templates

    def keys(self):
        return self._data.keys()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        if not isinstance(new_data, dict):
            raise IOError('Data must be input as a dictionary, not {}'.format(type(new_data)))
        self._data = new_data

    @property
    def templates(self):
        return self._templates

    @templates.setter
    def templates(self, new_templates):
        for _key in list(self.keys()):
            if _key not in list(new_templates.keys()):
                new_templates[_key] = None
        self._templates = new_templates

    @property
    def source(self):
        return ColumnDataSource(data=self.data, name=self.name)

    @source.setter
    def source(self, new_source):
        if isinstance(new_source, ColumnDataSource):
            self.data = dict(new_source.data)
        elif isinstance(new_source, dict):
            self.data = new_source
        else:
            raise IOError('New source must be either ColumnDataSource or Dict, not {}'.format(
                type(new_source)
            ))


class ParamSelector:

    def __init__(self,
                 selector: str,
                 options: list,
                 ):
        self.selector = selector
        self.options = options
        self.code = """
        """

    def drop_menu(self):
        from bokeh.models import Select
        return Select(title=self.selector, value=self.options[0], options=self.options)

class WorkingIntervalsTable:
    def __init__(self,
                 intervals,
                 width: None | int = None):
        """

        :param intervals:
            blixt_rp.core.core.Intervals object

        :param width:
        """
        if width is None:
            width = 600
        self.width = width
        self._intervals = intervals

    @property
    def intervals(self):
        return self._intervals

    @property
    def source(self):
        """
        Returns a ColumnDataSource with the following columns:
            'use', 'name', 'level', 'wells', 'color'
        :return:
            ColumnDataSource
        """
        _well_names = self.intervals.well_names()
        _interval_names = self.intervals.interval_names()

        _dict = self.intervals.get_strat_units()
        _ = _dict.pop('desc')
        _ = _dict.pop('source')
        _dict['use'] = [True] * len(_dict['name'])

        # add a column with the well names that contains the interval of each row
        _wells = []
        for _name in _dict['name']:
            _wl = []
            for _wn in _well_names:
                if self.intervals.get_interval(_name, _wn) is not None:
                    _wl.append(_wn)
            _wells.append(_wl)

        _dict['wells'] = _wells

        return ColumnDataSource(_dict)

    def __dict__(self):
        return self.intervals.get_intervals_dict()


class CrossPlotter:
    """
    Class for holding the cross plot
    """
    def __init__(self,
                 data_sources: None | dict = None,
                 width: None | int = None,
                 height: None | int = None,
                 tools: None | list = None):
        """

        :param data_sources:
            dictionary of DataSource's
            All variables within one DataSource have the same length
            The same variable name can be reused across DataSource's
        :param width:
            int
        :param height:
            int
        :param tools:
        """

        if width is None:
            width = 600
        if height is None:
            height = 600
        self.width = width
        self.height = height
        self._data_sources = data_sources
        if data_sources is not None:
            self._sources = {_key: _val.source for _key, _val in data_sources.items()}
        else:
            self._sources = None
        if tools is None:
            tools = default_tools
        self._tools = tools

    @property
    def sources(self):
        return self._sources

    @property
    def all_variables(self):
        _all = []
        for _source in self._data_sources.values():
            _all += list(_source.keys())
        return list(set(_all))

    @property
    def templates(self):
        _all = {}
        for _source in self._data_sources.values():
            for _key, _val in _source.templates.items():
                _all[_key] = _val
        return _all

    @property
    def common_variables(self):
        return list(set.intersection(
            *[set(list(_s.keys())) for _s in self._data_sources.values()]
        ))

    def figure(self):
        xplot = figure(width=self.width, height=self.height, tools=self._tools)
        xplot.toolbar.active_inspect = None
        xplot.toolbar.logo = None
        return xplot

    def draw(self):
        xplot = self.figure()

        # Determine x and y variable
        x_var = self.common_variables[0]
        y_var = self.common_variables[1]
        size = 30

        _i = 0
        # TODO To be able to change x, y, size and color, we need to create a "temporary"
        # TODO ColumnDataSource with 'x', 'y', 'size', and 'color' parameters, which we then
        # TODO interchange with the drop down Select buttons. See Fossagrim.misc_plots.plot_collected_stand_data()
        # TODO for inspiration
        for _name, _source in self.sources.items():
            xplot.scatter(x=x_var, y=y_var, source=_source, fill_color=cnames[_i], marker=markers[_i],
                          legend_label=_name, size=size)
            _i += 1

        # Axes
        for _var, _axis in zip([x_var, y_var], [xplot.xaxis, xplot.yaxis]):
            # _axis.axis_label_text_font_size = '10px'
            # _axis.major_label_text_font_size = '10px'
            _axis.axis_label_standoff = 0
            _axis.axis_label = '{} [{}]'.format(
                self.templates[_var].name, self.templates[_var].units)
        for _var, _range in zip([x_var, y_var], [xplot.x_range, xplot.y_range]):
            if self.templates[_var].min is not None:
                _range.start = self.templates[_var].min
            if self.templates[_var].max is not None:
                _range.end = self.templates[_var].max



        # Legend
        if len(xplot.legend) > 0:
            xplot.legend.click_policy = 'hide'
            xplot.legend.location = 'top_right'
            # xplot.legend.label_text_font_size = '8pt'

        return xplot

    def show(self, out_file):
        from bokeh.io import output_file
        output_file(out_file)
        xplot = self.draw()
        show(xplot)


class TestCases(unittest.TestCase):
    def test_data_source(self):
        from blixt_rp.core.core import Template
        md1 = np.linspace(1000., 2000., 500)
        md2 = np.linspace(800., 2500., 800)
        l1_1 = np.random.normal(10., 1., 500)
        l1_2 = np.random.normal(10., 1., 800)
        l2_1 = np.random.normal(100., 1., 500)
        l2_2 = np.random.normal(100., 1., 800)
        l3 = np.random.normal(50., 1., 500)
        l4 = np.random.normal(70., 1., 800)

        template_md = Template(name='md', units='m', marker='circle', fill_color='red')
        template_one = Template(name='var_one', units='m', min=5., max=15., marker='circle', fill_color='red')
        template_two = Template(name='var_two', units='m/s', min=90., max=110., marker='square', fill_color='blue')
        template_three = Template(name='var_three', units='kg', min=10., max=80., marker='triangle', fill_color='yellow')
        template_four = Template(name='var_four', units='feet', min=30., max=90., marker='hex', fill_color='green')

        data_one = dict(md=md1, var_one=l1_1, var_two=l2_1, var_three=l3)
        data_two = dict(md=md2, var_one=l1_2, var_two=l2_2, var_four=l4)
        templates_one = {_x.name: _x for _x in [template_md, template_one, template_two, template_three]}
        templates_two = {_x.name: _x for _x in [template_md, template_one, template_two, template_four]}

        ds1 = DataSource(data=data_one, templates=templates_one)
        ds2 = DataSource(data=data_two, templates=templates_two)
        xp = CrossPlotter(dict(one=ds1, two=ds2))
        # print(xp.all_variables)
        # print(xp.common_variables)
        # print(xp.templates)
        xp.show('C:\\Users\marte\Downloads\plot.html')

    def test_intervals(self):
        from blixt_rp.core.core import Intervals
        project_table = "C:\\Users\\marte\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx"
        wis = Intervals()
        wis.read_blixt_tops(project_table)
        wis_table = WorkingIntervalsTable(wis)
        print(wis_table.source.data)

    def test_from_well(self):
        from blixt_rp.core.project_new import Project
        from blixt_rp.core.core import LogTable, Intervals

        project_table = "C:\\Users\\marte\\PycharmProjects\\blixt_rp\\excels\\project_table_new.xlsx"

        project = Project(
            name='testing',
            working_dir='C:\\Users\\marte\\PycharmProjects\\blixt_rp',
            project_table=project_table
        )
        log_table = LogTable({'Density': 'rho_dry', 'P velocity': 'vp_dry', 'S velocity': 'vs_dry',
                              'Porosity': 'PHIE', 'Volume': 'VCL'})

        wis = Intervals()
        wis.read_blixt_tops(project.project_table)

        project.load_all_wells(log_table=log_table)
        print([w.name for w in project.wells])
        xp = CrossPlotter(
            {w.name: w.data_source() for w in project.wells}
        )
        xp.show('C:\\Users\marte\Downloads\plot.html')
