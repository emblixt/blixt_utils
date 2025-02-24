import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import (ColumnDataSource, DataTable, TableColumn, NumberEditor, CheckboxEditor, StringEditor, Text,
                          Patch)
from bokeh.layouts import column

output_file('C:\\Users\marte\Documents\plot.html')

def test1():
    # Create a ColumnDataSource with sample data
    table_data = {'x': [1, 2],
            'y': [6, 7],
            'name': ['Point 1', 'Point 2'],
            'visible': [True, True]}
    table_source = ColumnDataSource(table_data)

    _y = np.linspace(2, 10, 100)
    line_data = {
        'x1': (_y - 6.)**2,
        'x2': 10. - (_y - 6.)**2,
        'y': _y
    }
    line_source = ColumnDataSource(line_data)

    # Create a figure and add a circle glyph
    p = figure(title="Simple Bokeh Example")
    # You could perphaps update the line_source with these arrays?
    _x = np.hstack((line_source.data['x1'], line_source.data['x2']))
    _y = np.hstack((line_source.data['y'], line_source.data['y'][::-1]))
    # patch_glyph = Patch(x=_x, y=_y, fill_color='#a6cee3')
    # p.add_glyph(glyph=patch_glyph)
    p.patch(_x, _y, fill_color='#a6cee3')

    p.scatter(x='x', y='y', source=table_source, size=10, color='navy')
    p.line(x='x1', y='y', source=line_source, line_color='red', legend_label='Curve 1')
    p.line(x='x2', y='y', source=line_source, line_color='blue', legend_label='Curve 2')

    # Create a DataTable
    columns = [
        TableColumn(field='visible', title='Show:', editor=CheckboxEditor()),
        TableColumn(field='name', title='Name', editor=StringEditor()),
        TableColumn(field="x", title="X", editor=NumberEditor()),
        TableColumn(field="y", title="Y", editor=NumberEditor())
    ]
    data_table = DataTable(source=table_source, columns=columns, width=400, height=200, editable=True)
    text_glyph = Text(x='x', y='y', text='name', text_alpha='visible')
    p.add_glyph(table_source, text_glyph)

    # p.legend.click_policy = 'hide'
    # Combine the plot and table in a layout
    layout = column(p, data_table)

    # Show the result
    show(layout)


if __name__ == '__main__':
    test1()
