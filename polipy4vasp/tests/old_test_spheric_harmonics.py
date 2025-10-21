#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:38:08 2021

@author: carolin
"""

from vasa import asa
import pytest
import numpy as np
import polipy4vasp.sph as sph

def test_dericative_spheric_harmonics():
    Lmax = 4
    Rcut = 5
    L = 4
    ati = 2
    atj = 1
    dx = 1e-6
    np.random.seed(0)
    sph.setup_asa(Lmax)

    r = np.random.rand(5, 2, 3) * Rcut
    normR = np.linalg.norm(r, axis = -1)
    nr = r / normR[..., np.newaxis]
    y, dy = sph.sph_harm(Lmax, nr)
    dy = dy / normR[..., np.newaxis]
    #print(dy[L, ati, atj, x])# L, i, j, xyz

    for i in range(3):
        x = i
        rm = np.copy(r)
        rm[ati, atj, x] -= dx
        rm /= np.linalg.norm(rm, axis = -1)[..., np.newaxis]
        rp = np.copy(r)
        rp[ati, atj, x] += dx
        rp /= np.linalg.norm(rp, axis = -1)[..., np.newaxis]
        ym, _ = sph.sph_harm(Lmax, rm)
        yp, _ = sph.sph_harm(Lmax, rp)
        d = (yp - ym) / (2 * dx)
        #print(d[L, ati, atj])
        #print(dy[Lmax, ati, atj, x])
        assert abs(d[L, ati, atj] - dy[Lmax, ati, atj, x]) < 1e-12, 'The derivative of the spheric harmonics is wrong'
