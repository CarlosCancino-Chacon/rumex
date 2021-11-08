#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "rumex"
DESCRIPTION = "A rule-based system for generating expressive performances"
KEYWORDS = "music performance midi"
URL = "https://github.com/neosatrapahereje/rumex"
EMAIL = "carloscancinochacon@gmail.com"
AUTHOR = "Carlos Cancino-Chacón"
REQUIRES_PYTHON = ">=3.5"
VERSION = "0.1.0"

# What packages are required for this module to be executed?
REQUIRED = ["numpy", "scipy", "partitura", "basismixer"]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

SCRIPTS = [
    # "bin/RuMExRender",
    os.path.join("bin", "RuMExRender")
]


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="ext/markdown",
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"rumex": []},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    scripts=SCRIPTS,
    include_package_data=True,
    license='GPLv3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
