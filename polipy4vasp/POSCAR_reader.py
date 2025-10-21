#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .ML_AB_reader import Configuration
         
def read_POSCAR(filename='POSCAR'):
    r'''
    Reads a vasp POSCAR file.
    
    Arguments
    ---------
    filename : str, optional
        Name and path of the POSCAR file, default = "POSCAR"
        
    Returns
    -------
    configuration : Configuration
        An atomic configuration
    '''
    f = open(filename,'r')
    POSCAR = []
    for line in f:
        POSCAR.append(line)
    f.close()
    system_name = POSCAR[0].replace('\n','').strip()
    #system_name = system_name + ' ' * (40-len(system_name)) + '\n'
    bP = float(POSCAR[1])
    a1 = bP*np.array(POSCAR[2].split(),dtype=np.float64)
    a2 = bP*np.array(POSCAR[3].split(),dtype=np.float64)
    a3 = bP*np.array(POSCAR[4].split(),dtype=np.float64)
    atomname = [i for i in POSCAR[5].strip().split()]
    NatomType = [int(i) for i in POSCAR[6].strip().split()]
    n = sum(NatomType)
    direct = False
    if 'D' == POSCAR[7][0] or 'd' == POSCAR[7][0]:
        direct = True
    Pos =[]
    for i in range(n):
        buf = np.array(POSCAR[8+i].split(),dtype=np.float64)
        if direct:
            buf = a1*buf[0]+a2*buf[1]+a3*buf[2]
        Pos.append(buf)
    NType = []
    for I, m in enumerate(NatomType):
        NType.append(np.ones(m,dtype=np.int32)*I)
    return Configuration(natom = n,
                         lattice = np.array([a1,a2,a3]),
                         atompos = np.array(Pos),
                         atomtype = np.hstack(NType),
                         atomname = atomname,
                         maxtype = len(atomname),
                         sysname = system_name)
