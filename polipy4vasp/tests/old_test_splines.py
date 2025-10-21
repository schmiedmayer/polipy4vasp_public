#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 10:51:42 2021

@author: carolin
"""

import pytest
from pytest import approx
import numpy as np
import polipy4vasp.splines as splines
from polipy4vasp.main import Setup
from scipy.special import spherical_jn
from scipy.special import spherical_in

def test_cutoff_function():
    Rcut = 6
    r = np.linspace(0, Rcut, 50)
    fcut = splines.cutoff_function(Rcut, r)
    # Do we need different test functions or are two comparisons ok?
    assert fcut[0] == 1, 'Cutoff function is not 1 at x = 0'
    assert fcut[-1] == approx(0, abs = 1e-6), 'Cutoff function is not approximatly 0 at x = Rcut' # TODO: Isn't 0.0002 actually a large error!

def test_q_nl():
    Rcut = 6
    r = np.linspace(0, Rcut, 50)
    Nmax = 2
    Lmax = 3
    q = splines.get_qnl(Rcut,Nmax,Lmax)
    for i in range(Lmax + 1):
        assert spherical_jn(i, q[i, :] * Rcut) == approx(0, abs = 1e-14), 'A calculated root is not small enough'
        assert spherical_jn(i, q[i, :] * r[-1]) == approx(0, abs = 1e-14), 'A radial basis functions is not zero at Rcut'

def test_derivative_h_nl():
    setting = Setup(Rcut = 6,
                    SigmaAtom = 0.3)
    h = splines.get_splines(setting, 10)
    diff = []
    #np.random.seed(0)
    for rcut in np.random.random(100) * 2 + 4 - 0.0003:
        r_var = np.linspace(-0.0003, 0.0003, 7) + rcut
        a , b = np.polyfit(r_var, h(r_var, 0)[1, 2], 1) # y = a*x + b
        diff.append(abs(a - h(rcut, 1)[1, 2]))
    #print('mean difference of derivatives: ', np.mean(diff), '\tstd.dev: ', np.std(diff))
    assert np.mean(diff) == approx(0, abs = 2e-8), 'The difference of the analytic and numeric derivatives is too large'