#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pytest import approx
import numpy as np
from copy import deepcopy

from polipy4vasp.kernel import K, K_s, get_dK
from polipy4vasp.kernel import radial_part, dKxdp, angular_part, combine_derivertives
from polipy4vasp.kernel import radial_part, dKxdp, angular_part, combine_derivertives


from polipy4vasp.descriptors import get_AllDescriptors, get_Descriptor
from polipy4vasp.sph import setup_asa

from test_utils import dx, generate_Settings, generate_Symmetric_Settings, generate_Globals, generate_Configurations, generate_Splines, make_num_diff, lm_to_l_and_m

setup_asa(6)
Settings = generate_Settings()
Globals = generate_Globals(Settings)
Splines = generate_Splines(Settings)
#SymSettings = generate_Symmetric_Settings()
#SymGlobals = generate_Globals(SymSettings)
#SymSplines = generate_Splines(SymSettings)
Configurations = generate_Configurations()
#lamb = [0,1,2]
#TenGlobals = [generate_Globals(Settings,lamb=l,ten=True) for l in lamb]
#TenSplines = [generate_Splines(Settings,lamb=l) for l in lamb]
#TenSymGlobals = [generate_Globals(SymSettings,lamb=l,ten=True) for l in lamb]
#TenSymSplines = [generate_Splines(SymSettings,lamb=l) for l in lamb]

def test_K_s():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for desc in get_AllDescriptors(sett,glob,Configurations,h):
            for lc in desc.lc:
                assert np.allclose(np.diag(K(lc,lc,sett)), K_s(lc,sett), equal_nan = True), 'Kernel: "K" and "K_s" are not consistent'

def test_dK():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for desc in get_AllDescriptors(sett,glob,Configurations,h):
            for lc in desc.lc:
                lrc = desc.lc[np.random.randint(len(desc.lc))]
                ref_K = K(lc,lrc,sett)
                KK, dK = get_dK(lc,lrc,sett)
                assert np.allclose(ref_K, KK, equal_nan = True), 'Kernel: "K" and "get_dK" are not consistent with respect to "K"'
                indx, plc, mlc = make_num_diff(lc,dx)
                num_dK = (K(plc,lrc,sett) - K(mlc,lrc,sett)) / (dx * 2)
                assert np.allclose(num_dK[indx[0]], dK[indx[0],:,indx[1]], equal_nan = True, atol = dx), '"get_dK": Derivative diverges from numeric differences!'

def test_radial_part():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            pconf = deepcopy(conf)
            mconf = deepcopy(conf)
            indx, ppos, mpos = make_num_diff(conf.atompos,dx)
            JJ = conf.atomtype[indx[0]]
            pconf.atompos = ppos
            mconf.atompos = mpos
            pdesc = get_Descriptor(sett,glob,pconf,h)
            mdesc = get_Descriptor(sett,glob,mconf,h)
            
            desc = get_Descriptor(sett,glob,conf,h)
            lrc = desc.lc[np.random.randint(len(desc.lc))]
            L = sett.Nmax2*desc.maxtype
            
            num_dphi = [(K(plc[:,:L],lrc[:,:L],sett) - K(mlc[:,:L],lrc[:,:L],sett)).sum(0) / (dx *2 ) for plc, mlc in zip(pdesc.lc, mdesc.lc)]
    
            for J, lc in enumerate(desc.lc):
                _, dK = get_dK(lc[:,:L],lrc[:,:L],sett)
                dphi = radial_part(desc,dK,len(lrc),J) * sett.Beta
                assert np.allclose(num_dphi[J], dphi[indx[0],indx[1]], equal_nan = True, atol = dx), '"radial_part": Derivative diverges from numeric differences!'
                
def test_angular_part():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            pconf = deepcopy(conf)
            mconf = deepcopy(conf)
            indx, ppos, mpos = make_num_diff(conf.atompos,dx)
            JJ = conf.atomtype[indx[0]]
            pconf.atompos = ppos
            mconf.atompos = mpos
            pdesc = get_Descriptor(sett,glob,pconf,h)
            mdesc = get_Descriptor(sett,glob,mconf,h)
            
            desc = get_Descriptor(sett,glob,conf,h)
            lrc = desc.lc[np.random.randint(len(desc.lc))]
            L = sett.Nmax2*desc.maxtype
            
            num_dphi = [(K(plc[:,L:],lrc[:,L:],sett) - K(mlc[:,L:],lrc[:,L:],sett)).sum(0) / (dx *2 ) for plc, mlc in zip(pdesc.lc, mdesc.lc)]
            
            for J, lc in enumerate(desc.lc):
                _, dK = get_dK(lc[:,L:],lrc[:,L:],sett)
                dphi = angular_part(sett,glob,desc,dK,len(lrc),J) * (1-sett.Beta)
                assert np.allclose(num_dphi[J], dphi[indx[0],indx[1]], equal_nan = True, atol = dx), '"angular_part": Derivative diverges from numeric differences!'
                
def test_combine_derivertives():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            pconf = deepcopy(conf)
            mconf = deepcopy(conf)
            indx, ppos, mpos = make_num_diff(conf.atompos,dx)
            JJ = conf.atomtype[indx[0]]
            pconf.atompos = ppos
            mconf.atompos = mpos
            pdesc = get_Descriptor(sett,glob,pconf,h)
            mdesc = get_Descriptor(sett,glob,mconf,h)
            
            desc = get_Descriptor(sett,glob,conf,h)
            lrc = desc.lc[np.random.randint(len(desc.lc))]
            
            num_dphi = [(K(plc,lrc,sett) - K(mlc,lrc,sett)).sum(0) / (dx *2 ) for plc, mlc in zip(pdesc.lc, mdesc.lc)]
            
            for J, lc in enumerate(desc.lc):
                _, dphi = combine_derivertives(sett,glob,desc,lrc,len(lrc),J)
                assert np.allclose(num_dphi[J], dphi[indx[0],indx[1]], equal_nan = True, atol = dx), '"combine_derivertives": Derivative diverges from numeric differences!'




if __name__ == '__main__':
    test_K_s()
    test_dK()
    test_radial_part()
    test_angular_part()
    test_combine_derivertives()
    