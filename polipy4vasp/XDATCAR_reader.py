#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===============================================================
XDATCAR file reader routine (:mod:`polipy4vasp.XDATCAR_reader`)
===============================================================

.. currentmodule:: polipy4vasp.XDATCAR_reader
    
"""

import numpy as np

from polipy4vasp.ML_AB_reader import Configuration
from polipy4vasp.preprocess import conver_conf

def read_XDATCAR(filename='XDATCAR'):
    r'''
    Reads a vasp XDATCAR file.

    Arguments
    ---------
    filename : str, optional
        Name and path of the XDATCAR file, default = "XDATCAR"

    Returns
    -------
    configurations : list
        List of atomic configurations
    '''
    with open(filename,'r') as file:
        XDATCAR = file.readlines()

    out = []
    sysname = XDATCAR[0].replace('\n','').strip()
    bP = float(XDATCAR[1])
    a1 = bP*np.array(XDATCAR[2].split(),dtype=np.float64)
    a2 = bP*np.array(XDATCAR[3].split(),dtype=np.float64)
    a3 = bP*np.array(XDATCAR[4].split(),dtype=np.float64)
    atomname = [i for i in XDATCAR[5].strip().split()]
    NatomType = [int(i) for i in XDATCAR[6].strip().split()]
    natom = sum(NatomType)
    lattice = np.array([a1,a2,a3])
    atomtype = np.hstack([np.ones(m,dtype=np.int32)*I for I, m in enumerate(NatomType)])
    maxtype = len(atomname)
    
    # XDATCAR for ISIF=!3
    if XDATCAR[8+natom][:21] == "Direct configuration=" :
        XDATCAR = XDATCAR[7:]
        n = len(XDATCAR)
        POS = np.array_split(XDATCAR, n//(natom+1))
        for P in POS:
            pos = []
            for p in P[1:]:
                buf = np.array(p.split(),dtype=np.float64)
                pos.append(a1*buf[0]+a2*buf[1]+a3*buf[2])
            atompos = np.array(pos)
            out.append(conver_conf(Configuration(natom = natom,
                                                 lattice = lattice,
                                                 atompos = atompos,
                                                 atomtype = atomtype,
                                                 atomname = atomname,
                                                 maxtype = maxtype,
                                                 sysname = sysname)))
    # XDATCAR for ISIF==3
    else:
        n = len(XDATCAR)
        POS = np.array_split(XDATCAR, n//(natom+8))
        for P in POS:
            sysname = P[0].replace('\n','').strip()
            bP = float(P[1])
            a1 = bP*np.array(P[2].split(),dtype=np.float64)
            a2 = bP*np.array(P[3].split(),dtype=np.float64)
            a3 = bP*np.array(P[4].split(),dtype=np.float64)
            atomname = [i for i in P[5].strip().split()]
            NatomType = [int(i) for i in P[6].strip().split()]
            natom = sum(NatomType)
            lattice = np.array([a1,a2,a3])
            atomtype = np.hstack([np.ones(m,dtype=np.int32)*I for I, m in enumerate(NatomType)])
            maxtype = len(atomname)
            pos = []
            for p in P[8:]:
                buf = np.array(p.split(),dtype=np.float64)
                pos.append(a1*buf[0]+a2*buf[1]+a3*buf[2])
            atompos = np.array(pos)
            out.append(conver_conf(Configuration(natom = natom,
                                                 lattice = lattice,
                                                 atompos = atompos,
                                                 atomtype = atomtype,
                                                 atomname = atomname,
                                                 maxtype = maxtype,
                                                 sysname = sysname)))
    return out
