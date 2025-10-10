import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from blixt_utils.io.io import read_general_ascii_GENERAL as read_file, project_wells_new
from blixt_utils.io.io import (get_las_well_info, get_las_curve_info, get_las_header, get_las_names_units,
                               get_las_start_data_line, read_sums_and_averages)
test_file_dir = str(os.path.dirname(__file__).replace(
    'blixt_utils\\blixt_utils\\unit_tests',
    'blixt_rp\\test_data'))

files = {
    'las_file1':
        [os.path.join(test_file_dir, "L-30.las"), 'space', 53,
         ["DEPTH", "CALD", "CALS", "DEPT", "DRHO", "DT", "GRD", "GRS", "ILD", "ILM", "LL8", "NPHILS", "NPHISS",
          "RHOB", "SP"], None,
         ["FT",  "IN",  "IN", "FT", "G/CC",  "US/F", "GAPI", "GAPI",  "OHMM", "OHMM", "OHMM", "V/V", "V/V",
          "G/CC", "MV"]
         ],
    'petrel_well_tops_file':
        ["C:\\Users\\marte\\Downloads\\Well tops.txt", 'space', 17, None, None ],
    'rokdoc_tops_file':
        [os.path.join(test_file_dir, "RokDocTops.csv"),';', 5,
         4,
         None,
         ['', '', 'm', 'm', 'ms', 'm', 'm', 'm']],
    'another_tops_file':
        [os.path.join(test_file_dir, "strat_litho_wellbore.csv"), ',', 1,
         0,
         None,
         None],
    'well_path_file1':
        ["C:\\Users\\marte\\Downloads\\25_2_6___wellpath.txt", 'space', 1,
         None,
         None,
         None],
    'well_path_file2':
        [os.path.join(test_file_dir, "Well C wellpath.txt"), 'tab', 1,
         0,
         None,
         ['', '', 'm', 'm', 'm', 'm', 'deg', 'deg', 'm', 'm', '']],
    'well_path_file3':
        ["S:\\Well\\UTM31_North_Sea_All\\Q-33\\33_9-17\\33_9_17___wellpath.txt", 'tab', 9,
         ['TVD', 'MD', 'INC'],
         [4, 1, 3],
         ['m', 'm', 'deg']],
    'checkshot_file':
        [os.path.join(test_file_dir, "Petrel_checkshots.ascii"), 'space', 15,
         ['X', 'Y', 'Z', 'TWT picked', 'MD', 'Well', 'Average velocity', 'Interval velocity', 'TWT'],
         None,
         ['m', 'm', 'm', 'ms', 'm', '', 'm/s', 'm/s', 'ms']],
    'checkshot_file_again':
        [os.path.join(test_file_dir, "Petrel_checkshots.ascii"), 'space', 15,
         ['X', 'TWT picked', 'Well'],
         [0, 3, 5],
         ['m', 'ms', '']],
    'checkshot_file2':
        [os.path.join(test_file_dir, "Well A checkshot.txt"), 'space', 4,
         ['md', 'owt'],
         [0, 1],
         ['m', 'ms']]
}


class IOTestCase(unittest.TestCase):

    def test_read_file(self):
        for _name in list(files.keys()):
            if _name in ['petrel_well_tops_file', 'well_path_file1']:
                continue
            print('\n', _name)
            data, units = read_file(
                    files[_name][0],
                    files[_name][1],
                    files[_name][2],
                    var_names=files[_name][3],
                    var_columns=files[_name][4],
                    var_units=files[_name][5],
                    encoding='utf-8-sig')
            print('  ', list(data.keys())[0], data[list(data.keys())[0]][:10])
            print('  ', units)
            self.assertIsInstance(data, dict, 'Test data {}'.format(_name))
            self.assertIsInstance(units, dict, 'Test units {}'.format(_name))

    def test_log_curve_read_file(self):
        from blixt_rp.core.log_curve_new import read_general_ascii as lc_read_file, LogCurve
        for _name in list(files.keys()):
            if _name in ['petrel_well_tops_file', 'well_path_file1']:
                continue
            print('\n', _name)

            log_curves = lc_read_file(
                    files[_name][0],
                    files[_name][1],
                    files[_name][2],
                    var_names=files[_name][3],
                    var_columns=files[_name][4],
                    var_units=files[_name][5],
                    encoding='utf-8-sig')
            if log_curves is not None:
                for _key in list(log_curves.keys()):
                    print(log_curves[_key].name)
                    print(log_curves[_key].min, log_curves[_key].max)
                    self.assertIsInstance(log_curves[_key], LogCurve, '{} is not a LogCurve'.format(_key))

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

    def test_rename_log_name(self):
        from blixt_utils.io.io import rename_log_name
        rename_logs = {'phie': ['CPI_PHIE', 'PHIE_smooth'],
                       'vcl': 'VCLAY',
                       'gr': ['GR_LFP']}
        for old_name in ['CPI_PHIE', 'PHIE_smooth', 'vclay', 'VCLAY', 'GR_LFP', 'XXX']:
            print('{} -> {}'.format(old_name, rename_log_name(old_name, rename_logs)))
        print('{} -> {}'.format('TEST', rename_log_name('TEST', None)))


    def test_well_reader(self):
        from blixt_utils.io.io import well_reader
        file_name = files['las_file1'][0]
        with open(file_name, "r", encoding='UTF8') as f:
            lines = f.readlines()
        # null_val, generated_keys, well_dict = well_reader(lines, file_format='las', rename_logs={'den': ['rhob']})
        # null_val, generated_keys, well_dict = well_reader(lines, file_format='las', rename_logs={'den': ['XXX']})
        # null_val, generated_keys, well_dict = well_reader(lines, file_format='las', rename_logs={'DEN': ['RHOB']})
        # dept and depth are already among the logs, so the operation below will remove one of the depth logs
        null_val, generated_keys, well_dict = well_reader(lines, file_format='las', rename_logs={'dept': ['depth']})
        print(null_val)
        print(generated_keys)
        for _key in list(well_dict.keys()):
            print('Key: ', _key)
            if isinstance(well_dict[_key], str):
                print(' - ', well_dict[_key])
            elif isinstance(well_dict[_key], dict):
                print(' - ', list(well_dict[_key].keys()))
            print(' - X -')


    def test_project_wells_new(self):
        project_table = test_file_dir.replace('test_data', 'excels\\project_table_new.xlsx')
            # result = project_wells_new(project_table, test_file_dir.replace('test_data', ''))
        result = project_wells_new(project_table, test_file_dir.replace('test_data', ''), do_rename=True)
        for _key in list(result.keys()):
            print(_key)
            for _i in list(result[_key].keys()):
                print(' -', _i)
                print('  -', result[_key][_i])
        self.assertIsInstance(result, dict)


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

    def test_sums_and_averages(self):
        excel_file = "C:\\Users\\emb\\OneDrive - Petrolia NOCO AS\\Technical work\PL1221\\SumsAndAverages.xlsx"
        result = read_sums_and_averages(excel_file)
        for _key in list(result.keys()):
            print(_key, result[_key])
        self.assertTrue(True)
