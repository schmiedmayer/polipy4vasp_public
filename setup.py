#!/usr/bin/env python

import setuptools
from numpy.distutils.core import setup, Extension
import site
import sys

ext = [
    Extension(
        name="vasa",
        sources=["polipy4vasp/asa/asa.f90"],
        f2py_options=[],
    )
]

setup(ext_modules=ext)
