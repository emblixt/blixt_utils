from setuptools import setup, find_packages
import re

#with open('seismiqb/__init__.py', 'r') as f:
#    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='blixt_utils',
    version=0.1,
    #packages=['io', 'misc', 'plotting', 'signal_analysis'],
    #packages=find_packages(),
    packages=['blixt_utils'],
    py_modules=['main', 'utils'],
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
