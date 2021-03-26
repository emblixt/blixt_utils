from setuptools import setup
import re

verstr = 'unknown'
VERSIONFILE = "blixt_utils/_version.py"
with open(VERSIONFILE, "r") as f:
    verstrline = f.read().strip()
    pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
    mo = pattern.search(verstrline)
if mo:
    verstr = mo.group(1)
    print("Version "+verstr)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name='blixt_utils',
    version=verstr,
    packages=[
        'blixt_utils',
        'blixt_utils.io',
        'blixt_utils.misc',
        'blixt_utils.plotting',
        'blixt_utils.signal_analysis',
        'blixt_utils.training_data'
    ],
    url='https://github.com/emblixt/blixt_utils',
    license='Apache 2.0',
    author='Erik Mårten Blixt',
    author_email='marten.blixt@gmail.com',
    description='Utility tools often used in my libraries',
    long_description='',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.16.0',
        'matplotlib>=3.0.2',
        'segyio>=1.8.3',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
)
