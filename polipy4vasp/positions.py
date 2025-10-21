#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===================================================================
Nearest neighbor list setup routines (:mod:`polipy4vasp.positions`)
===================================================================

.. currentmodule:: polipy4vasp.positions
    
"""

import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class NN_Configuration:
    r'''
    Dataclass containing the information of a configuration in a nearest neighbor sense.
    
    Args:
        nnn (ndarray) : List containing the number of nearest neighbors for each atom
        nnl (ndarray) : Array containing the atom number of the nearest neighbors
        maxnn (int) : Maximal number of nearest neighbors
        nnr (ndarray) : Array of distances :math:`r_{ij}` to nearest neighbors
        nnvecr (ndarray) : Array of vectors :math:`\textbf{r}_{ij}` to nearest neighbors
        natom (int) : Number of atoms
        maxtype (int) : Number of different atom species
        centraltype (ndarray) : Atom species of the central atom
        nntype (ndarray) : Atom species of the nearest neighbors
        mult_per_img (bool) : Is ``True`` if multible periodic images are in the nearest neighbors list
        volume (scalar, optional) : Volume of the simulation box
    '''
    nnn: ndarray
    nnl: ndarray
    maxnn: int
    nnr: ndarray
    nnvecr: ndarray
    natom : int
    maxtype : int
    centraltype : ndarray
    nntype : ndarray
    mult_per_img : bool
    volume: float = None
    _lattice: ndarray = None

def get_rlattice_and_v(lattice):
    r'''
    Calulates the volume of the simulation box and the reciprocal lattice vectors
    
    Arguments
    ---------
    lattice : ndarray
        Array of lattice vectors 
        
    Returns
    -------
    rlattice : ndarray
        Reciprocal lattice vectors
    v       : float
        Volume of the simulation box
    '''
    v = np.dot(lattice[0],np.cross(lattice[1],lattice[2]))
    rlattice = np.linalg.inv(lattice).T
    return rlattice, v
        
def get_vecrij(configuration,rlattice,Rcut):
    r'''
    Calculates the distances to every atom and to the periodic images.
    
    Arguments
    ---------
    configuration : Configuration
        An atomic configuration
    rlattice   : ndarray
        Reciprocal lattice vectors
    Rcut      : scalar
        The cutoff radius :math:`R_\text{cut}`
    
    Returns
    -------
    vecrij : ndarray
        Distance vector :math:`\textbf{r}_{ij}` to every atom and its periodic images
    rij : ndarray
        Distances :math:`r_{ij}` to every atom and its periodic images
    n : ndarray
        Number of generated periodic images in each cartesian direction
     
    Notes
    -----
    The reciprocal lattice vectores are used to determine how many periodic images are needed according to
    
    .. math:: n_i = \lceil |\vec{b}_i| R_\text{cut} \rceil,
    
    where :math:`\vec{b}_i` are the reciprocal lattic vectors.
    '''
    # Calculating how many periodic images are needed
    n = np.ceil(np.linalg.norm(rlattice,axis=1)*Rcut,out=np.zeros(3,int),casting='unsafe')
    vecrij = np.zeros([2*n[0]+1,2*n[1]+1,2*n[2]+1,configuration.natom,configuration.natom,3])
    # Calculating the atomic distance vectors for all atoms and generated periodic images
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

def reduce_rij(configuration,vecrij,rij,Rcut,n):
    r'''
    Reduces :math:`\textbf{r}_{ij}` and :math:`r_{ij}`by discarding all atoms along the :math:`j` index which 
    are outside of one :math:`R_\text{cut}`. Returns an array only containing the distances to the 
    nearest neighbors (atoms within :math:`R_\text{cut}`) and a nearest neighbor list.
    
    Arguments
    ---------
    configuration : Configuration
        An atomic configuration
    vecrij : ndarray
        Distance vector :math:`\textbf{r}_{ij}` to every atom and its periodic images within the cutoff radius
    rij : ndarray
        Distances :math:`r_{ij}` to every atom and its periodic images within the cutoff radius
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`. Default 5 angstrom.
    n : ndarray
        Number of generated periodic images in each cartesian direction
        
    Returns
    -------
    nn_conf : NN_Configuration
        Configuration in a nearest neighbor layout
    '''
    def make2array(nnn,nnr,nnvecr,nnl,nntype,natom,maxnn):
        r'''
        Routine to store nearest neighbors into an ndarray if the number of nearest neighbors differs.
        '''
        outr = np.full([natom,maxnn],np.nan)
        outvecr = np.full([natom,maxnn,3],np.nan)
        outnn = np.full([natom,maxnn],np.nan,dtype=np.int32)
        outnntype = np.full([natom,maxnn],np.nan,dtype=np.int32)
        for i in range(natom):
            outr[i,:nnn[i]] = nnr[i]
            outvecr[i,:nnn[i],:] = nnvecr[i]
            outnn[i,:nnn[i]] = nnl[i]
            outnntype[i,:nnn[i]] = nntype[i]
        return outnn, outr, outvecr, outnntype
    
    # Determining indizes where rij < Rcut
    args = np.argwhere(np.logical_and(rij < Rcut,rij != 0)).T
    # periodic image indizes where rij < Rcut
    projections = args[0:3]
    # atomic indizes j where rij < Rcut
    args = args[3:]
    nnn = []
    nnl = []
    nnr = []
    nnvecr = []
    nntype = []
    for i in range(configuration.natom):
        arg = args[0] == i
        nnn.append(np.sum(arg))
        buf1 = args[1,arg]
        nntype.append(configuration.atomtype[buf1])
        nnl.append(buf1)
        buf2 = projections[...,arg]
        nnr.append(rij[buf2[0],buf2[1],buf2[2],i,buf1])
        nnvecr.append(vecrij[buf2[0],buf2[1],buf2[2],i,buf1,:])    
    maxnn = max(nnn)
    if sum(nnn)/configuration.natom == maxnn :
        nnl, nnr, nnvecr, nntype = np.array(nnl), np.array(nnr), np.array(nnvecr), np.array(nntype)
    else :
        nnl, nnr, nnvecr, nntype = make2array(nnn,nnr,nnvecr,nnl,nntype,configuration.natom,maxnn)
    mult_per_img = np.sum(np.linalg.norm(configuration.lattice,axis = 1) <= 2*Rcut) != 0 
    return NN_Configuration(nnn = np.array(nnn),
                            nnl = nnl,
                            maxnn = maxnn,
                            nnr = nnr,
                            nnvecr = nnvecr,
                            natom = configuration.natom,
                            maxtype = configuration.maxtype,
                            centraltype = configuration.atomtype,
                            nntype = nntype,
                            mult_per_img = mult_per_img)
               
def conf2nn_conf(configuration,Rcut):
    r'''
    Calculates the nearest neighbors of a given atomic configuration for the case that there are multible
    periodic images within the cutoff sphere. The nearest neighbors are all atoms :math:`j` which
    are within :math:`R_\text{cut}` around the central atom :math:`ì`.
    
    Arguments
    ---------
    configuration : Configuration
        An atomic configuration
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`. Default 5 angstrom.
    Returns
    -------
    nn_conf : NN_Configuration
        Configuration in a nearest neighbor layout
    '''
    rlattice, v = get_rlattice_and_v(configuration.lattice)
    vecrij, rij, n = get_vecrij(configuration,rlattice,Rcut)
    nn_conf = reduce_rij(configuration,vecrij,rij,Rcut,n)
    nn_conf.volume = v
    nn_conf._lattice = configuration.lattice
    return nn_conf

def conf2nn_conf_fast(configuration, Rcut):
    r'''
    Calculates the nearest neighbors of a given atomic configuration using the minimum image convetion.
    The nearest neighbors are all atoms :math:`j` which are within :math:`R_\text{cut}` around the
    central atom :math:`ì`.
    
    Arguments
    ---------
    configuration : Configuration
        An atomic configuration
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`. Default 5 angstrom.
    Returns
    -------
    nn_conf : NN_Configuration
        Configuration in a nearest neighbor layout
    '''
    rlattice, v = get_rlattice_and_v(configuration.lattice)
    rpos = configuration.atompos @ rlattice.T
    vecr = (np.mod(rpos[np.newaxis,:] - rpos[:,np.newaxis]+0.5, 1) - 0.5) @ configuration.lattice
    r = np.linalg.norm(vecr, axis=-1)
    mask = r <= Rcut
    np.fill_diagonal(mask, False)
    nnn = np.sum(mask,axis=1)
    maxnn = np.max(nnn)
    nnr = np.full([configuration.natom,maxnn],np.nan)
    nnvecr = np.full([configuration.natom,maxnn,3],np.nan)
    nnl = np.full([configuration.natom,maxnn],np.nan,dtype=np.int32)
    nntype = np.full([configuration.natom,maxnn],np.nan,dtype=np.int32)
    l = np.arange(configuration.natom)
    for i, (nn,m) in enumerate(zip(nnn,mask)):
        nnr[i,:nn] = r[i,m]
        nnvecr[i,:nn,:] = vecr[i,m,:]
        nnl[i,:nn] = l[m]
        nntype[i,:nn] = configuration.atomtype[m]
    
    return NN_Configuration(nnn = nnn,
                            nnl = nnl,
                            maxnn = maxnn,
                            nnr = nnr,
                            nnvecr = nnvecr,
                            natom = configuration.natom,
                            maxtype = configuration.maxtype,
                            centraltype = configuration.atomtype,
                            nntype = nntype,
                            mult_per_img = False,
                            volume = v,
                            _lattice = configuration.lattice)

def check_mult_per_img(lattice, Rcut):
    r'''
    Checks weather multible periodic images are within the cutoff sphere.

    Parameters
    ----------
    lattice : ndarray
        The lattic vectors.
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`. Default 5 angstrom.

    Returns
    -------
    check: bool
        Returns `True` if multible images are within the cutoff sphere.

    '''
    
    cross = np.cross([lattice[0],lattice[0],lattice[1]],
                     [lattice[1],lattice[1],lattice[2]])
    V = np.dot(lattice[2], cross[0])
    norm = np.linalg.norm(cross,axis=-1)
    return np.any((V/norm) <= 2*Rcut)
    
    
    

def get_nn_conf(configuration,Rcut):
    r'''
    Calculates the nearest neighbors of a given atomic configuration. The nearest neighbors are all atoms :math:`j` which
    are within :math:`R_\text{cut}` around the central atom :math:`ì`.
    
    Arguments
    ---------
    configuration : Configuration
        An atomic configuration
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`. Default 5 angstrom.
    Returns
    -------
    nn_conf : NN_Configuration
        Configuration in a nearest neighbor layout
    '''
    if check_mult_per_img(configuration.lattice, Rcut):
        return conf2nn_conf(configuration,Rcut)
    else:
        return conf2nn_conf_fast(configuration, Rcut)

def reduce_nn_conf(nn_conf,Rcut):
    r'''
    Reduces an excisting NN_Configuration to an NN_Configuration with :math:`R_\text{cut}`.
    
    Arguments
    ---------
    nn_conf : NN_Configuration
        Configuration in a nearest neighbor layout
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`.
    Returns
    -------
    nn_conf : NN_Configuration
        Configuration in a nearest neighbor layout
    '''
    mask = nn_conf.nnr < Rcut
    nnn = np.sum(mask,axis=1)
    maxnn = np.max(nnn)
    natom = nn_conf.natom
    nnr = np.full([natom,maxnn],np.nan)
    nnvecr = np.full([natom,maxnn,3],np.nan)
    nnl = np.full([natom,maxnn],np.nan,dtype=np.int32)
    nntype = np.full([natom,maxnn],np.nan,dtype=np.int32)
    for i, (nn,m) in enumerate(zip(nnn,mask)):
        nnr[i,:nn] = nn_conf.nnr[i,m]
        nnvecr[i,:nn,:] = nn_conf.nnvecr[i,m,:]
        nnl[i,:nn] = nn_conf.nnl[i,m]
        nntype[i,:nn] = nn_conf.nntype[i,m]
    
    if nn_conf.mult_per_img:
        mult_per_img = check_mult_per_img(nn_conf._lattice, Rcut)
    else :
        mult_per_img = False
    return NN_Configuration(nnn = nnn,
                            nnl = nnl,
                            maxnn = maxnn,
                            nnr = nnr,
                            nnvecr = nnvecr,
                            natom = natom,
                            maxtype = nn_conf.maxtype,
                            centraltype = nn_conf.centraltype,
                            nntype = nntype,
                            mult_per_img = mult_per_img,
                            volume = nn_conf.volume)

def get_nn_confs(settings,configuration):
    r'''
    Calculates the nearest neighbors of a given atomic configuration. The nearest neighbors are all atoms :math:`j` which
    are within :math:`R^{(2)}_\text{cut}` and  :math:`R^{(3)}_\text{cut}` for two- and three-body descriptors around 
    the central atom :math:`ì`.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    configuration : Configuration
        An atomic configuration

    Returns
    -------
    nn_conf2 : NN_Configuration
        Configuration in a nearest neighbor layout for the radial descriptor (two-body).
    nn_conf3 : NN_Configuration
        Configuration in a nearest neighbor layout for the angular descriptor (three-body).

    '''
    
    if settings.Rcut2 > settings.Rcut3:
        nn_conf2 = get_nn_conf(configuration,settings.Rcut2)
        nn_conf3 = reduce_nn_conf(nn_conf2,settings.Rcut3)
    elif settings.Rcut3 > settings.Rcut2:
        nn_conf3 = get_nn_conf(configuration,settings.Rcut3)
        nn_conf2 = reduce_nn_conf(nn_conf3,settings.Rcut2)
    else:
        nn_conf2 = get_nn_conf(configuration,settings.Rcut2)
        nn_conf3 = deepcopy(nn_conf2)
        
    return nn_conf2, nn_conf3

def LR_nn_conf(conf):
    # CF
    r'''
    Fakes a nearest neighbor configuration to be able to use the existing routines.

    Parameters
    ----------
    conf : Configuration
        An atomic configuration.

    Returns
    -------
    NN_Configuration
        Fake nearest neighbor configuration.

    '''
    natoms = conf.natom
    nnn = np.zeros(natoms, dtype = np.int32) + natoms - 1
    nnl = np.array([np.arange(natoms) for _ in range(natoms)])[~np.eye(natoms, dtype=bool)].reshape(natoms,-1)
    return NN_Configuration(nnn = nnn,
                            nnl = nnl,
                            maxnn = natoms - 1, #no self interaction
                            nnr = None,
                            nnvecr =None,
                            natom = natoms,
                            maxtype = conf.maxtype,
                            centraltype = conf.atomtype,
                            nntype = conf.atomtype[nnl].reshape(natoms,-1),
                            mult_per_img = None)
