#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:29:07 2021

@author: carolin
"""
import pytest
import numpy as np
import polipy4vasp.positions as positions
from polipy4vasp.ML_AB_reader import Configuration

pos = np.array([[0, 0, 0], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], [0.1, 0, 0], np.random.rand(3)])
lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Rcut = [0.5, 1, 1.5, 2]
Natom = pos.shape[0]
conf = Configuration(natom = Natom,
                     lattice = lattice,
                     atompos = pos,
                     atomtype = np.zeros(Natom),
                     atomname = ['Si'],
                     maxtype = 1)

def get_vecrij_1(configuration,rlattice,Rcut):
    n = np.ceil(np.linalg.norm(rlattice,axis=1)*Rcut,out=np.zeros(3,int),casting='unsafe') +1
    vecrij = np.zeros([2*n[0]+1,2*n[1]+1,2*n[2]+1,configuration.natom,configuration.natom,3])
    ix = -1
    for nx in range(-n[0],n[0]+1):
        ix += 1
        iy = -1
        for ny in range(-n[1],n[1]+1):
            iy += 1
            iz = -1
            for nz in range(-n[2],n[2]+1):
                iz += 1
                vecrij[ix][iy][iz] = configuration.atompos[np.newaxis,:,:]-(configuration.atompos[:,np.newaxis,:] + nx*configuration.lattice[0] + ny*configuration.lattice[1] + nz*configuration.lattice[2]) 
    rij = np.linalg.norm(vecrij,axis=-1)
    return vecrij, rij, n

def conf2nn_conf_1(configuration,Rcut):
    rlattice, v = positions.get_rlattice_and_v(configuration.lattice)
    vecrij, rij, n = get_vecrij_1(configuration,rlattice,Rcut)
    nn_conf = positions.reduce_rij(configuration,vecrij,rij,Rcut,n)
    nn_conf.volume = v
    return nn_conf

def test_NNL():
    for rcut in Rcut:
        nn_conf = positions.conf2nn_conf(conf,rcut)
        nn_conf_1 = conf2nn_conf_1(conf, rcut)
        assert np.all(nn_conf.nnn == nn_conf_1.nnn), 'nnn not working'
        assert np.all(nn_conf.nnl == nn_conf_1.nnl), 'nnl not working'
        assert np.all(np.nan_to_num(nn_conf.nnr) == np.nan_to_num(nn_conf_1.nnr)), 'nnr not working'
        assert np.all(np.nan_to_num(nn_conf.nnvecr) == np.nan_to_num(nn_conf_1.nnvecr)), 'nnr not working'
