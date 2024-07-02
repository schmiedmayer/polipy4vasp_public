#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:38:10 2021

@author: carolin
"""
import pytest
from pytest import approx
import numpy as np
from copy import deepcopy
from polipy4vasp.main import Setup, ML_ForceField
from polipy4vasp.ML_AB_reader import Configuration
from polipy4vasp.sph import setup_asa
from polipy4vasp.splines import get_splines
from polipy4vasp.preprocess import get_weights, AU_to_Ang
from polipy4vasp.descriptors import get_Descriptor
from polipy4vasp.globals import setup_globals

dx = 0.0001

settings = Setup()
setup_asa(settings.Lmax)
glob = setup_globals(settings)
h = get_splines(settings)
pos = np.random.rand(30,3)*10
lattice = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
Natom = pos.shape[0]
maxtype = 1
conf = Configuration(natom = Natom,
                     lattice = lattice,
                     atompos = pos,
                     atomtype = np.random.randint(0,maxtype,Natom),
                     atomname = ['Si'],
                     maxtype = maxtype)
desc = get_Descriptor(settings,glob,conf,h)

mlff = ML_ForceField(settings = settings,
                     info = settings,
                     glob = glob,
                     lrc = desc.lc,
                     w = [np.random.rand(np.sum(conf.atomtype == i)) for i in range(maxtype)],
                     shift_ene = 0,
                     shift_stress = 0,
                     h = h)

def test_all_forces():
    #analytical
    aF = mlff.predict(conf,False,False)[1]
    
    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    
    pE = mlff.predict(pconf,False,False)[0]
    
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mE = mlff.predict(mconf,False,False)[0]
    
    #Calculate finite differences
    nF = -(pE-mE)/(2*AU_to_Ang(dx))
    
    #assert
    cF = aF[n,a] - nF
    assert cF == approx(0,abs=5*10**-5,rel=10**-4) , 'Some thing wrong in force predicton!'
