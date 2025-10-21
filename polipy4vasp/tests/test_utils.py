#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from polipy4vasp.main import Setup
from polipy4vasp.globals import setup_globals
from polipy4vasp.splines import get_splines
from polipy4vasp.POSCAR_reader import read_POSCAR

dx = 1e-6

def generate_Settings(lamb=0):
    r'''
    Generates a list of Settings for testing.

    Parameter
    ---------
    lamb: int, optional
        Rank of tensor. Default = 0.
    
    Returns
    -------
    List of Settings

    '''
    
    set1 = Setup(Rcut2 = 8.,
                 Rcut3 = 5.5,
                 SigmaAtom = 0.5,
                 Beta = 0.9,
                 Nmax2 = 12,
                 Nmax3 = 6,
                 Lmax = 4,
                 Kernel = "poli",
                 Zeta = 4,
                 SigmaExp = 0.4)
    
    set2 = Setup(Rcut2 = 4.,
                 Rcut3 = 5.,
                 SigmaAtom = 0.4,
                 Beta = 0.4,
                 Nmax2 = 6,
                 Nmax3 = 8,
                 Lmax = 5,
                 Kernel = "linear",
                 Zeta = 4,
                 SigmaExp = 0.4)
    
    set3 = Setup(Rcut2 = 6.,
                 Rcut3 = 6.,
                 SigmaAtom = 0.6,
                 Beta = 0.5,
                 Nmax2 = 7,
                 Nmax3 = 7,
                 Lmax = 3,
                 Kernel = "gaus",
                 Zeta = 4,
                 SigmaExp = 0.4)
    
    # set4 = Setup(Rcut2 = 9.,
    #              Rcut3 = 4.2,
    #              SigmaAtom = 0.3,
    #              Beta = 0.8,
    #              Nmax2 = 10,
    #              Nmax3 = 6,
    #              Lmax = 4,
    #              Kernel = "ngaus",
    #              Zeta = 4,
    #              SigmaExp = 0.4)
    
    Settings = [set1, set2, set3]#, set4]
    return Settings

def generate_Symmetric_Settings(lamb=0):
    r'''
    Generates a list of Settings for testing with symmetric Nmax and Rcut.

    Parameter
    ---------
    lamb: int, optional
        Rank of tensor. Default = 0.
    
    Returns
    -------
    List of Settings

    '''
    
    set1 = Setup(Rcut2 = 5.5,
                 Rcut3 = 5.5,
                 SigmaAtom = 0.5,
                 Beta = 0.9,
                 Nmax2 = 8,
                 Nmax3 = 8,
                 Lmax = 4,
                 Kernel = "poli",
                 Zeta = 4,
                 SigmaExp = 0.4)
    
    set2 = Setup(Rcut2 = 5.,
                 Rcut3 = 5.,
                 SigmaAtom = 0.4,
                 Beta = 0.4,
                 Nmax2 = 6,
                 Nmax3 = 6,
                 Lmax = 5,
                 Kernel = "linear",
                 Zeta = 4,
                 SigmaExp = 0.4)
    
    set3 = Setup(Rcut2 = 6.,
                 Rcut3 = 6.,
                 SigmaAtom = 0.6,
                 Beta = 0.5,
                 Nmax2 = 7,
                 Nmax3 = 7,
                 Lmax = 3,
                 Kernel = "gaus",
                 Zeta = 4,
                 SigmaExp = 0.4)
    
    Settings = [set1, set2, set3]
    return Settings

def generate_Globals(Settings,lamb=0,ten=False):
    r'''
    Generates Globals for Testing.

    Parameters
    ----------
    Settings : list
        list of settings
    lamb: int, optional
        Rank of tensor. Default = 0.
    ten: bool, optional
        Checks if tensor training is activated. Default = False

    Returns
    -------
    Globals : list
        list of Globals.

    '''
    class Type:
        def __init__(self,lamb):
            self.lamb = lamb
    class buf:
        def __init__(self,lamb,ten):
            if ten:
                self._ten=True
                self.Type = Type(lamb)
            else :
                self._ten=False
            
    TenClass = buf(lamb,ten)
    Globals = [setup_globals(sett,TenClass) for sett in Settings]
    return Globals

def generate_Splines(Settings,lamb=0):
    r'''
    Generates Cubic splines for Testing.

    Parameters
    ----------
    Settings : list
        list of settings
    lamb: int, optional
        Rank of tensor. Default = 0.

    Returns
    -------
    Splines : list
        list of splines

    '''
    Splines = [get_splines(sett,lamb) for sett in Settings]
    return Splines 

def generate_Configurations():
    r'''
    Returns a list of Configurations for testing.

    Returns
    -------
    Configurations : list
        List of Configurations

    '''
    conf1 = read_POSCAR('POSCARS/POSCAR_mol')
    conf2 = read_POSCAR('POSCARS/POSCAR_bulk_small')
    conf3 = read_POSCAR('POSCARS/POSCAR_bulk_big')
    
    Configurations = [conf1,conf2,conf3]
    
    return Configurations

def make_num_diff(array,dx=1e-6):
    r'''
    Sets up finite differences for an arbitrary array.

    Parameters
    ----------
    array : ndarray
        The array used for finite differences.
    dx : scalar, optional
        Finite difference. The default is 1e-5.

    Returns
    -------
    i : tuple
        Random indices.
    parray : ndarray
        Array with positiv shift.
    marray : ndarry
        Array with negativ shift.

    '''
    i = tuple([np.random.randint(s) for s in array.shape])
    parray = deepcopy(array)
    marray = deepcopy(array)
    parray[i]+= dx
    marray[i]-= dx
    return i, parray, marray

def l_and_m_to_lm(l,m):
    r'''
    Computes the compund index :math:`lm` via
    
    .. math:: l(l+1)+m.

    Parameters
    ----------
    l : scalar, int
        Angular quantum number :math:`l`
    m : scalar, int
        Magnetic quantum number :math:`l`

    Returns
    -------
    lm : scalar, int
        Compund index :math:`lm`.

    '''
    if abs(m) > l :
        raise ValueError('abs(m) > l!')
    lm = l*(l+1) + m
    return lm

def lm_to_l_and_m(lm):
    r'''
    

    Parameters
    ----------
    lm : scalar, int
        Compund index :math:`lm`.

    Returns
    -------
    l : scalar, int
        Angular quantum number :math:`l`
    m : scalar, int
        Magnetic quantum number :math:`l`

    '''
    l = 0
    while (l+1)*(l+1) <= lm: l += 1
    m = lm - l*(l+1)
    return l, m

def rot_matrix(a=90,b=0,c=0,deg=True):
    r'''
    Generates a matrix for general rotations in 3D.

    Parameters
    ----------
    a : float, optional
        :math:`\alpha` yaw. The default is 90.
    b : float, optional
        :math:`\beta` pitch. The default is 0.
    c : float, optional
        :math:`\gamma` roll. The default is 0.
    deg : bool, optional
        Uses DEG as units for the angles if set to `True` otherwise RAD are used. The default is True.

    Returns
    -------
    R : ndarray
        General 3D rotation matrix.

    '''
    if deg :
        f = np.pi / 180
        a *= f
        b *= f
        c *= f
        
    Rz = np.array([
         [np.cos(a),-np.sin(a),0],
         [np.sin(a),np.cos(a),0],
         [0,0,1]],dtype=np.float64)
    Ry = np.array([
         [np.cos(b),0,np.sin(b)],
         [0,1,0],
         [-np.sin(b),0,np.cos(b)]],dtype=np.float64)
    Rx = np.array([
         [1,0,0],
         [0,np.cos(c),-np.sin(c)],
         [0,np.sin(c),np.cos(c)]],dtype=np.float64)
    
    R = Rz @ Ry @ Rx
    return R

def random_rot():
    r'''
    Generates a random matrix for rotations in 3D.

    Returns
    -------
    R : ndarray
        A random 3D rotation matrix.

    '''
    return rot_matrix(*np.random.rand(3)*2*np.pi,deg=False)
