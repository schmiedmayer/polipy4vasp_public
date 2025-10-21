#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:04:55 2021

@author: carolin
"""

import pytest
from pytest import approx
import numpy as np
from copy import deepcopy
from polipy4vasp.positions import conf2nn_conf
from polipy4vasp.descriptors import *
from polipy4vasp.main import Setup
from polipy4vasp.ML_AB_reader import Configuration
from polipy4vasp.sph import setup_asa
from polipy4vasp.splines import get_splines
from polipy4vasp.kernel import K, get_dK, radial_part, dKxdp, angular_part
from polipy4vasp.desing import get_PHI
from polipy4vasp.globals import setup_globals

settings = Setup(Nmax=8,Lmax=4,Rcut=4,Zeta=0,SigmaExp=0.4,Kernel='poli')
setup_asa(settings.Lmax)
glob = setup_globals(settings)
h = get_splines(settings)
pos = np.random.rand(30,3)*10
#pos = np.array([[1,1,1],[1.3,1,1],[1,1.3,1],[1,1,1.3]])
#pos = np.array([[0,0,0],[5.5,0,0],[0,5.5,0],[5.5,5.5,0]])
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

dx = 0.000001

def test_dcTilde():
    #analytical
    nnconf = conf2nn_conf(conf,settings.Rcut)
    c, adcTilde = get_cTilde(settings,nnconf,h)

    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pnnconf = conf2nn_conf(pconf,settings.Rcut)
    pcTilde, _ = get_cTilde(settings,pnnconf,h)
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mnnconf = conf2nn_conf(mconf,settings.Rcut)
    mcTilde, _ = get_cTilde(settings,mnnconf,h)
    #finite differences
    ndcTilde = (pcTilde-mcTilde)/(2*dx)
    #assert 
    cdcTilde = -np.nan_to_num(adcTilde[:,:,n,:,a]) - np.nan_to_num(ndcTilde[:,:,n,:])
    nn = np.sort(nnconf.nnl[n,:nnconf.nnn[n]])
    nndcTilde = adcTilde[:,:,nnconf.nnl == n,a] - np.sum(np.nan_to_num(ndcTilde[:,:,nn,:]),axis=-1)
    assert cdcTilde == approx(0,abs=10**-8,rel=10**-6) , 'Derivertiv of central atom cTilde not working!'
    assert nndcTilde == approx(0,abs=10**-8,rel=10**-6) , 'Derivertiv of nearest neighbor cTilde not working!'

def test_dc():
    sym_fac = np.empty((settings.Lmax+1)*(settings.Lmax+1))
    for l in range(settings.Lmax+1):
        sym_fac[l*l:(l+1)*(l+1)] = -(-1)**l
    sym_fac = sym_fac[np.newaxis,:,np.newaxis]
    #analytical
    nnconf = conf2nn_conf(conf,settings.Rcut)
    cTilde, dcTilde = get_cTilde(settings,nnconf,h)
    _, adc, aself_dc, nndo = get_c(settings,nnconf,cTilde,dcTilde)
    n = np.random.randint(nnconf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pnnconf = conf2nn_conf(pconf,settings.Rcut)
    cTilde, dcTilde = get_cTilde(settings,pnnconf,h)
    pc, _, _, _ = get_c(settings,pnnconf,cTilde,dcTilde)
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mnnconf = conf2nn_conf(mconf,settings.Rcut)
    cTilde, dcTilde = get_cTilde(settings,mnnconf,h)
    mc, _, _, _ = get_c(settings,mnnconf,cTilde,dcTilde)
    #finite differences
    ndc = (pc-mc)/(2*dx)
    #assert   
    nn = np.arange(nnconf.nnn[n])
    cdc = adc[:,:,n,nn,a] - sym_fac*ndc[:,:,nnconf.centraltype[n],nnconf.nnl[n,:nnconf.nnn[n]]]
    cself_dc = aself_dc[:,:,:,n,a]-ndc[:,:,:,n]
    assert cdc == approx(0,abs=10**-8,rel=10**-5) , 'Derivertiv of c not working!'
    assert cself_dc == approx(0,abs=10**-8,rel=10**-5) , 'Derivertiv of self c not working!'
    
def test_dp():
    #analytical
    nnconf = conf2nn_conf(conf,settings.Rcut)
    cTilde, dcTilde = get_cTilde(settings,nnconf,h)
    c, _, _, _ = get_c(settings,nnconf,cTilde,dcTilde)
    _, adp = get_p(settings,glob,nnconf,c)
    n = np.random.randint(settings.Nmax)
    l = np.random.randint(settings.Lmax+1)
    m = np.random.randint(-l,l+1)
    lm = l*(l+1)+m
    J = np.random.randint(nnconf.maxtype)
    a = np.random.randint(nnconf.natom)
    #numeric
    #+
    pc = deepcopy(c)
    pc[n,lm,J,a] += dx
    pp, _ = get_p(settings,glob,nnconf,pc)
    #-
    mc = deepcopy(c)
    mc[n,lm,J,a] -= dx
    mp, _ = get_p(settings,glob,nnconf,mc)
    #finite differences
    ndp = (pp-mp)/(2*dx)
    adp[n,lm,J,a] *= 2
    #assert

    cdp = adp[:,lm,:,a] - ndp[n,:,l,J,:,a]
    scdp = adp[:,lm,:,a] - ndp[:,n,l,:,J,a]
    assert cdp == approx(0,abs=10**-8,rel=10**-5) , 'Derivertiv of p not working!'
    assert scdp == approx(0,abs=10**-8,rel=10**-5) , 'Derivertiv of p not working!'
    

def test_dK():
    #analytical
    nnconf = conf2nn_conf(conf,settings.Rcut)
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    adK = [get_dK(desc,lrc[J],settings,J)[1] for J in range(conf.maxtype)]
    
    Jp = np.random.randint(conf.maxtype)
    n = np.random.randint(np.sum(conf.atomtype == Jp))
    a = np.random.randint(settings.Nmax*conf.maxtype+settings.Nmax*conf.maxtype*settings.Nmax*conf.maxtype*(settings.Lmax+1))
    
    #numeric
    #+
    pdesc = deepcopy(desc)
    pdesc.lc[Jp][n,a] += dx
    pK = [get_dK(pdesc,lrc[J],settings,J)[0] for J in range(conf.maxtype)]
    #-
    mdesc = deepcopy(desc)
    mdesc.lc[Jp][n,a] -= dx
    mK = [get_dK(mdesc,lrc[J],settings,J)[0] for J in range(conf.maxtype)]
    for J, K in enumerate(adK):
        #finite differences
        ndK = (pK[J]-mK[J])/(2*dx)
        #assert
        if J == Jp :
            cdK = K[n,:,a] - ndK[n,:]
            assert cdK == approx(0,abs=10**-8,rel=10**-6) , 'Derivertiv of K not working!'
        else:
            assert ndK == approx(0,abs=10**-8,rel=10**-6) , 'Derivertiv of K not working!'
        
def test_radial_part():
    settings.Beta = [2,0]
    nnconf = conf2nn_conf(conf,settings.Rcut)
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    buf_dK = [get_dK(desc,lrc[J],settings,J)[1] for J in range(conf.maxtype)]
    arad_dK = [radial_part(desc,buf_dK[J][:,:,:settings.Nmax*desc.maxtype],len(lrc[J]),J) for J in range(conf.maxtype)]
    
    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pdesc = get_Descriptor(settings,glob,pconf,h)
    pK = [get_dK(pdesc,lrc[J],settings,J)[0] for J in range(conf.maxtype)]
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mdesc = get_Descriptor(settings,glob,mconf,h)
    mK = [get_dK(mdesc,lrc[J],settings,J)[0] for J in range(conf.maxtype)]
    for J, K in enumerate(arad_dK):
        ndPHI = (pK[J]-mK[J])/(2*dx)
        cdPHI = settings.Beta[0]*K[n,a] - np.sum(ndPHI,axis = 0)
        assert cdPHI == approx(0,abs=10**-8,rel=10**-6) , 'Derivertiv of radial part not working!'

def test_angular_part():
    settings.Beta = [0,2]
    nnconf = conf2nn_conf(conf,settings.Rcut)
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [deepcopy(desc.lc[J]) for J in range(conf.maxtype)]
    buf_dK = [get_dK(desc,lrc[J],settings,J)[1] for J in range(conf.maxtype)]
    aang_dK = [angular_part(settings,glob,desc,buf_dK[J][:,:,settings.Nmax*desc.maxtype:],len(lrc[J]),J) for J in range(conf.maxtype)]
    test_dK = [dKxdp(settings,glob,desc,buf_dK[J][:,:,settings.Nmax*desc.maxtype:],J) for J in range(conf.maxtype)]
    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pdesc = get_Descriptor(settings,glob,pconf,h)
    pK = [get_dK(pdesc,lrc[J],settings,J)[0] for J in range(conf.maxtype)]
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mdesc = get_Descriptor(settings,glob,mconf,h)
    mK = [get_dK(mdesc,lrc[J],settings,J)[0] for J in range(conf.maxtype)]
    for J, K in enumerate(aang_dK):
        ndPHI = (pK[J]-mK[J])/(2*dx)
        cdPHI = settings.Beta[1]*K[n,a] - np.sum(ndPHI,axis = 0)
        assert cdPHI == approx(0,abs=10**-8,rel=10**-6) , 'Derivertiv of angular part not working!'

def test_dPHI():
    settings.Beta = [1,1]
    desc = get_Descriptor(settings,glob,conf,h)
    lrc = [desc.lc[J] for J in range(conf.maxtype)]
    adPHI = get_PHI(settings,glob,[desc],lrc,conf.maxtype)[1:]/settings.Wforc
    n = np.random.randint(conf.natom)
    a = np.random.randint(3)
    #numeric
    #+
    pconf = deepcopy(conf)
    pconf.atompos[n,a] += dx
    pdesc = get_Descriptor(settings,glob,pconf,h)
    pPHI = get_PHI(settings,glob,[pdesc],lrc,conf.maxtype)[0]
    #-
    mconf = deepcopy(conf)
    mconf.atompos[n,a] -= dx
    mdesc = get_Descriptor(settings,glob,mconf,h)
    mPHI = get_PHI(settings,glob,[mdesc],lrc,conf.maxtype)[0]
    #finite differences
    ndPHI = -(pPHI-mPHI)*conf.natom/(2*dx*settings.Wene)
    #assert
    cdPHI = adPHI[3*n+a,:] - ndPHI
    assert cdPHI == approx(0,abs=10**-7,rel=10**-6) , 'Derivertiv of PHI not working!'
    
        
# test_dcTilde()
# test_dc()
# test_dp()
# test_dK()
# test_radial_part()
# test_angular_part()
# test_dPHI()

def array_maker():
    p = np.chararray([8,8,5,3,3,30],itemsize = 100)
    for n in range(8):
        for npp in range(8):
            for l in range(5):
                for J in range(3):
                    for Jp in range(3):
                        for Na in range(30):
                            p[n,npp,l,J,Jp,Na] = str(n) + "n" + str(npp) + "n'" + str(l) + "l" + str(J) + "J" + str(Jp) + "J'" + str(Na) + "Na"
    return np.moveaxis(p.reshape(-1, 30), 0, -1)

#t = array_maker()