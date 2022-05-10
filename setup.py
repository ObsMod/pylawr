from setuptools import setup, find_packages
import glob
import pathlib

BASE = pathlib.Path(__file__).parent.resolve()

NAME = "pylawr"
AUTHOR = "pylawr developers"
AUTHOR_EMAIL = "finn.burgemeister@uni-hamburg.de"
DESCRIPTION = "pylawr is a Python package to load, process and plot weather " \
              "radar data"
LONG_DESCRIPTION = (BASE / 'README.md').read_text(encoding='utf-8')
KEYWORDS = 'Meteorology, Weather Radar'
URL = "https://wetterradar.uni-hamburg.de"
DOWNLOAD_URL = "https://github.com/ObsMod/pylawr"
LICENSE = "MIT"
CLASSIFIERS = [
    'Development Status :: 4 - BETA',

    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',

    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Atmospheric Science',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    "Programming Language :: Python :: 3.10",
    'Programming Language :: Python :: 3 :: Only',
]
MAJOR = 0
MINOR = 4
PATCH = 0
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"

setup(
    name=NAME,

    version=VERSION,

    description=DESCRIPTION,

    long_description=LONG_DESCRIPTION,

    long_description_content_type='text/markdown',

    url=DOWNLOAD_URL,  # Optional

    author=AUTHOR,  # Optional

    author_email=AUTHOR_EMAIL,  # Optional

    classifiers=CLASSIFIERS,

    keywords=KEYWORDS,

    #package_dir={'': 'pylawr'},  # Optional

    packages=find_packages(exclude=['docs', 'tests.*']),

    python_requires='>=3.7, <4',

    scripts=glob.glob("bin/pylawr*"),
)
