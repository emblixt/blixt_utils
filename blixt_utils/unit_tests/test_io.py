import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from blixt_utils.io.io import read_general_ascii_GENERAL as read_file, project_wells_new
from blixt_utils.io.io import (get_las_well_info, get_las_curve_info, get_las_header, get_las_names_units,
                               get_las_start_data_line)
test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_utils\\blixt_utils\\unit_tests',
    'blixt_rp\\test_data'))

files = {
    'las_file1':
        [os.path.join(test_file_dir, "L-30.las"), 'space', 53,
         ["DEPTH", "CALD", "CALS", "DEPT", "DRHO", "DT", "GRD", "GRS", "ILD", "ILM", "LL8", "NPHILS", "NPHISS",
          "RHOB", "SP"],
         ["FT",  "IN",  "IN", "FT", "G/CC",  "US/F", "GAPI", "GAPI",  "OHMM", "OHMM", "OHMM", "V/V", "V/V",
          "G/CC", "MV"]
         ],
    'petrel_well_tops_file':
        ["C:\\Users\\marte\\Downloads\\Well tops.txt", 'space', 17, None, None ],
    'rokdoc_tops_file':
        [os.path.join(test_file_dir, "RokDocTops.csv"),';', 5,
         4,
         ['', '', 'm', 'm', 'ms', 'm', 'm', 'm']],
    'another_tops_file':
        [os.path.join(test_file_dir, "strat_litho_wellbore.csv"), ',', 1,
         0,
         None],
    'well_path_file1':
        ["C:\\Users\\marte\\Downloads\\25_2_6___wellpath.txt", 'space', 1,
         None,
         None],
    'well_path_file2':
        [os.path.join(test_file_dir, "Well C wellpath.txt"), 'tab', 1,
         0,
         ['', '', 'm', 'm', 'm', 'm', 'deg', 'deg', 'm', 'm', '']],
    'checkshot_file':
        [os.path.join(test_file_dir, "Petrel_checkshots.ascii"), 'space', 15,
         ['X', 'Y', 'Z', 'TWT picked', 'MD', 'Well', 'Average velocity', 'Interval velocity', 'TWT'],
         ['m', 'm', 'm', 'ms', 'm', '', 'm/s', 'm/s', 'ms']
         ]
}


class IOTestCase(unittest.TestCase):

    def test_read_file(self):
        for _name in list(files.keys()):
            if _name in ['petrel_well_tops_file', 'well_path_file1']:
                continue
            print(_name)
            data, units = (
                read_file(files[_name][0], files[_name][1], files[_name][2], files[_name][3],  files[_name][4], encoding='utf-8-sig'))
            print(list(data.keys())[0], data[list(data.keys())[0]][:10])
            print(units)
            self.assertIsInstance(data, dict, 'Test data {}'.format(_name))
            self.assertIsInstance(units, dict, 'Test units {}'.format(_name))


    def test_get_las(self):
        fname = files['las_file1'][0]
        # for _x in get_las_well_info(fname):
        #     print(_x )
        #     break
        for _x in get_las_curve_info(fname):
            print(_x)
            break
        # for _x in get_las_header(fname):
        #     print(_x )
        #     break
        names, units = get_las_names_units(fname)
        print(names)
        print(units)
        data_row = get_las_start_data_line(fname)
        print(data_row)

        self.assertTrue(len(names) == len(units))
        self.assertIsInstance(data_row, int)


    def test1(self):
        print(test_file_dir)
        las_file = os.path.join(test_file_dir, 'Well A.las')
        project_table = os.path.dirname(__file__).replace('blixt_utils\\blixt_utils\\unit_tests', 'blixt_rp\\excels\\project_table.xlsx')
        print(project_table)
        working_dir = os.path.dirname(__file__).replace('blixt_utils\\blixt_utils\\unit_tests', '')
        wells = project_wells_new(project_table, working_dir)
        print(list(wells.keys()))


    def test2(self):
        typical_rename_string = 'LFP_VP_VIRGIN->VP_VIRG, LFP_VS_VIRGIN->VS_VIRG, LFP_AI_VIRGIN->AI_VIRG'
        print(interpret_rename_string(typical_rename_string))


    def test_read_checkshot_or_wellpath(self):
        from blixt_rp.core.well import Project
        dir_path = os.path.dirname(os.path.realpath(__file__)).replace('\\blixt_utils\\blixt_utils\\unit_tests', '')
        project_table = os.path.join(dir_path, 'blixt_rp', 'excels', 'project_table.xlsx')
        wp = Project(project_table=project_table)
        wells = wp.load_all_wells()
        this_well = list(wells.keys())[0]
        read_checkshot_or_wellpath(wp.project_table, this_well, "Checkshots")


