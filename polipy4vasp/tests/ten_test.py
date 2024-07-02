#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pytest import approx
import numpy as np
from copy import deepcopy
from polipy4vasp.positions import conf2nn_conf
from polipy4vasp.descriptors import *
from polipy4vasp.tensor_descriptors import *
from polipy4vasp.main import Setup, ML_ForceField
from polipy4vasp.ML_AB_reader import Configuration
from polipy4vasp.sph import setup_asa
from polipy4vasp.splines import get_splines
from polipy4vasp.globals import setup_globals
from polipy4vasp.preprocess import AU_to_Ang

import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
trans = R.from_euler('xyz', 2*np.pi*(1-2*np.random.rand(3)))
antitrans=trans.inv()

settings = Setup(Nmax=8,Lmax=4,lamb=1,Zeta=0,Beta=[0.6,0.4])
setup_asa(settings.Lmax)
glob = setup_globals(settings)
h = get_splines(settings)
pos = np.random.rand(30,3)*10
#pos = np.array([[0.1,0.1,0.1],[3.1,0.1,0.1],[0.1,0.1,3.1]])
lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])*10
Natom = pos.shape[0]
maxtype = 2
conf = Configuration(natom = Natom,
                     lattice = lattice,
                     atompos = pos,
                     atomtype = np.random.randint(0,maxtype,Natom),
                     atomname = ['Si'],
                     maxtype = maxtype)
bufnn = conf2nn_conf(conf,settings.Rcut)

dx = 1e-6

def ultimate_test():
    desc = get_Descriptor(settings,glob,conf,h)
    mlff = ML_ForceField(info=None,
                         settings = settings,
                         glob=glob,
                         lrc = desc.lc,
                         w = [np.random.rand((2 * settings.lamb + 1) * np.sum(conf.atomtype == J)) for J in range(conf.maxtype)],
                         shift_ene = None,
                         shift_stress = None,
                         h = h)
    
    aZ = mlff.predict(conf, write_output = False, deriv = True)[1]
    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pP = mlff.predict(pconf, write_output = False, deriv = False)
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mP = mlff.predict(mconf, write_output = False, deriv = False)
    #Calculate finite differences
    nZ = (pP-mP)/(2 * dx)
    #assert
    cZ = AU_to_Ang(aZ[n,a]) - nZ
    assert cZ == approx(0,abs=10**-7,rel=10**-6) , 'Derivertiv not working!'

    
def test_ten_dKxdp():
    nnconf = conf2nn_conf(conf,settings.Rcut)
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    buf_dK = [get_ten_dK(desc,lrc[J],J)[1] for J in range(conf.maxtype)]
    adK = [ten_dKxdp(settings,glob,desc,buf_dK[J],J) for J in range(conf.maxtype)]
    nnconf = conf2nn_conf(conf,settings.Rcut)
    cTilde, dcTilde = get_cTilde(settings,nnconf,h)
    c, _, _, _ = get_c(settings,nnconf,cTilde,dcTilde)
    p, _ = get_ten_p(settings,glob,nnconf,c)
    lc = np.moveaxis(p.reshape(2*settings.lamb+1,-1, nnconf.natom), 1, 2)
    do = [nnconf.centraltype == J for J in range(nnconf.maxtype)]
    lrc = [lc[:,d] for d in do]
#    for J, lr in enumerate(lrc):
#        print(np.argwhere(desc.lc[J][:,settings.Nmax*conf.maxtype:]!=lr))
            
    n = np.random.randint(settings.Nmax)
    l = np.random.randint(settings.Lmax+1)
    m = np.random.randint(-l,l+1)
    lm = l*(l+1)+m
    J = np.random.randint(nnconf.maxtype)
    Jp = np.random.randint(nnconf.maxtype)
    a = np.random.randint(np.sum(conf.atomtype == Jp))
#    a = np.random.randint(nnconf.natom)
    select = np.arange(conf.natom)[do[Jp]][a]
    #numeric
    #+
    pc = deepcopy(c)
    pc[n,lm,J,select] += dx
    pp, _ = get_ten_p(settings,glob,nnconf,pc)
    pp = np.moveaxis(pp.reshape(2*settings.lamb+1,-1, nnconf.natom), 1, 2)
    pp = [pp[:,d] for d in do]
    pdesc = deepcopy(desc)
    pdesc.lc = pp
    
    pK = [get_ten_dK(pdesc,lrc[J],J)[0] for J in range(conf.maxtype)]
    #-
    mc = deepcopy(c)
    mc[n,lm,J,select] -= dx
    mp, _ = get_ten_p(settings,glob,nnconf,mc)
    mp = np.moveaxis(mp.reshape(2*settings.lamb+1,-1, nnconf.natom), 1, 2)
    mp = [mp[:,d] for d in do]
    mdesc = deepcopy(desc)
    mdesc.lc = mp
    
    mK = [get_ten_dK(mdesc,lrc[J],J)[0] for J in range(conf.maxtype)]
    #finite differences
    ndK = [(pK[J]-mK[J])/(2*dx) for J in range(nnconf.maxtype)]
    for Jpp, bdK in enumerate(ndK):
        if Jpp == Jp : 
            cdK = (np.sum(bdK,axis = 0)-adK[Jpp][n,lm,J,a,:])
            assert cdK == approx(0,abs=10**-7,rel=10**-6) , 'Derivertiv of PHI not working!'
            

def test_ten_dKxdpxdc():
    nnconf = conf2nn_conf(conf,settings.Rcut)
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    buf_dK = [get_ten_dK(desc,lrc[J],J)[1] for J in range(conf.maxtype)]
    adK = [ten_dKxdpxdc(settings,glob,desc,buf_dK[J],J) for J in range(conf.maxtype)]
    nnconf = conf2nn_conf(conf,settings.Rcut)
    cTilde, dcTilde = get_cTilde(settings,nnconf,h)
    c, _, _, _ = get_c(settings,nnconf,cTilde,dcTilde)
    p, _ = get_ten_p(settings,glob,nnconf,c)
    lc = np.moveaxis(p.reshape(2*settings.lamb+1,-1, nnconf.natom), 1, 2)
    do = [nnconf.centraltype == J for J in range(nnconf.maxtype)]
    lrc = [lc[:,d] for d in do]
#    for J, lr in enumerate(lrc):
#        print(np.argwhere(desc.lc[J][:,settings.Nmax*conf.maxtype:]!=lr))
    n = np.random.randint(nnconf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pdesc = get_Descriptor(settings,glob,pconf,h)    
    pK = [get_ten_dK(pdesc,lrc[J],J)[0] for J in range(conf.maxtype)]
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mdesc = get_Descriptor(settings,glob,mconf,h)   
    mK = [get_ten_dK(mdesc,lrc[J],J)[0] for J in range(conf.maxtype)]
    #finite differences
    ndK = [(pK[J]-mK[J])/(2*dx) for J in range(nnconf.maxtype)]
    for Jpp, bdK in enumerate(ndK):
        cdK = (np.sum(bdK,axis = 0)-adK[Jpp][n,a])
        assert cdK == approx(0,abs=10**-7,rel=10**-6) , 'dKxdpxdc not working!'

def test_dphi():
    settings.Zeta=0
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    adphi =get_ten_dphi(settings,glob,desc,lrc)
    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pdesc = get_Descriptor(settings,glob,pconf,h)
    pphi = get_ten_phi(settings,pdesc,lrc)
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mdesc = get_Descriptor(settings,glob,mconf,h)
    mphi = get_ten_phi(settings,mdesc,lrc)
    #finite differences
    ndphi = (pphi-mphi)/(2*dx)
    #assert
    cdphi = adphi[n,a] - ndphi
    assert cdphi == approx(0,abs=10**-7,rel=10**-6) , 'Derivertiv of phi not working!'
    
def test_rot():
    desc = get_Descriptor(settings,glob,conf,h)
    conf.atompos = trans.apply(conf.atompos)
    conf.lattice = trans.apply(conf.lattice)
    rotdesc = get_Descriptor(settings,glob,conf,h)
    for J in range(conf.maxtype):
        ndesc = desc.lc[J]
        Ntype = np.sum(conf.atomtype == J)
        rdesc = np.moveaxis(np.roll(antitrans.apply(np.roll(np.moveaxis(rotdesc.lc[J].reshape(3,-1),0,1),shift=1,axis=1)),shift=-1,axis=1),1,0).reshape(3,Ntype,-1)
        cdesc = ndesc - rdesc
        assert cdesc == approx(0,abs=10**-7,rel=10**-6) , 'Symetries not working!'
        
def test_norm_dK():
    settings.Zeta=1
    nnconf = conf2nn_conf(conf,settings.Rcut)
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    adK = [get_ten_dnormK(settings,desc,lrc[J],J) for J in range(conf.maxtype)]
    l = np.random.randint(2*settings.lamb+1)
    J = np.random.randint(nnconf.maxtype)
    a = np.random.randint(len(desc.lc[J][l,:,0]))
    n = np.random.randint(len(desc.lc[J][l,0,:]))
    pdesc = deepcopy(desc)
    pdesc.lc[J][l,a,n] += dx
    pK = [get_ten_normK(pdesc,lrc[J],J) for J in range(conf.maxtype)]
    
    mdesc = deepcopy(desc)
    mdesc.lc[J][l,a,n] -= dx
    mK = [get_ten_normK(mdesc,lrc[J],J) for J in range(conf.maxtype)]
    
    ndK = (pK[J]-mK[J])/(2*dx)
    ddK = ndK[a,:,l,:]-adK[J][l,a,l,:,:,n]
    assert ddK == approx(0,abs=10**-7,rel=10**-6) , 'Derivertiv not working!'
    
def test_norm_dphi():
    settings.Zeta=1
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    adphi =get_ten_dphi(settings,glob,desc,lrc)
    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pdesc = get_Descriptor(settings,glob,pconf,h)
    pphi = get_ten_phi(settings,pdesc,lrc)
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mdesc = get_Descriptor(settings,glob,mconf,h)
    mphi = get_ten_phi(settings,mdesc,lrc)
    #finite differences
    ndphi = (pphi-mphi)/(2*dx)
    #assert
    cdphi = adphi[n,a] - ndphi
    assert cdphi == approx(0,abs=10**-7,rel=10**-6) , 'Derivertiv of phi not working!'

# ultimate_test()
# test_ten_dKxdp()
# test_ten_dKxdpxdc()
# test_dphi()
# test_norm_dK()
# test_norm_dphi()
# test_rot()
