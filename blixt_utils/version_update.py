import os
import re
from datetime import datetime as datetime

VERSIONFILE = "_version.py"
TMPFILE = "_version_tmp.py"


def get_current_version():
    verstr = 'unknown'
    with open(VERSIONFILE, "r") as f:
        verstrline = f.read().strip()
        pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
        mo = pattern.search(verstrline)
    if mo:
        verstr = mo.group(1)
        print("Version "+verstr)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
    return verstr


def write_version_to_tmp(version):
    with open(VERSIONFILE, "r") as orig:
        with open(TMPFILE, "w") as out:
            for line in orig.readlines():
                if '__version__ =' in line:
                    out.write('__version__ = "{}"'.format(version))
                else:
                    out.write(line)
                

if __name__ == "__main__":
    year_str = str(datetime.now().year)
    current_version = get_current_version()
    current_version_no = int(current_version.split('.')[-1])
    current_yday = current_version.split('.')[1]
    yday_str = str(datetime.now().timetuple().tm_yday)
    if yday_str == current_yday:
        version_no = str(current_version_no + 1)  # increase version number if a build from today exists
    else:
        version_no = '0'

    write_version_to_tmp('{}.{}.{}'.format(year_str, yday_str, version_no))
    os.replace(TMPFILE, VERSIONFILE)
