#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
from pytest import approx
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from polipy4vasp.main import Setup, ML_ForceField
from polipy4vasp.ML_AB_reader import Configuration
from polipy4vasp.sph import setup_asa
from polipy4vasp.splines import get_splines
from polipy4vasp.preprocess import get_weights
from polipy4vasp.descriptors import get_Descriptor
from polipy4vasp.globals import setup_globals

dx = 0.001

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
                     glob = glob,
                     lrc = desc.lc,
                     w = [np.random.rand(np.sum(conf.atomtype == i)) for i in range(maxtype)],
                     shift_ene = 0,
                     shift_stress = 0,
                     h = h,
                     info = settings)

trans = R.from_euler('xyz', 2*np.pi*np.random.rand(3))
antitrans=trans.inv()

def test_forc_rot():
    E, F = mlff.predict(conf,False,False)
    
    conf.atompos = trans.apply(conf.atompos)
    conf.lattice = trans.apply(conf.lattice)
    
    rotE, rotF = mlff.predict(conf,False,False)

    assert rotE-E == approx(0,abs=10**-10,rel=10**-10) , 'Energies are not rotational invariant!'
    assert antitrans.apply(rotF) - F == approx(0,abs=10**-10,rel=10**-10) , 'Forces are not rotational invariant!'
