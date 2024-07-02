#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy


def pre_process(settings,data):
    r"""
    This routine pre processes the training data by calculating the energy per
    atom, shifting it by the mean and deviding it by the variance.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    data : Training_Data
        Contains all vital information from the ML_AB file
        
    Returns
    -------
    shift_ene : scalar
        Shift of the total energy traning data
    shift_stress : scalar
        Shift of the stress tensor traning data
    """
    l_natom = np.array([conf.natom for conf in data.configurations])
    toteneperatom = data.energies/l_natom
    shift_ene = np.mean(toteneperatom)
    shift_stress = np.mean(data.stresstensors)
    data.energies = toteneperatom - shift_ene
    data.stresstensors -= shift_stress
    settings.Wene = settings.Wene/np.std(data.energies)
    settings.Wforc = settings.Wforc/np.std(np.vstack(data.forces))
    settings.Wstress = settings.Wstress/np.std(data.stresstensors)
    return shift_ene, shift_stress

def ten_pre_process(settings,data):
    r"""
    This routine pre processes the training data by calculating the energy per
    atom, shifting it by the mean and deviding it by the variance.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    data : Training_Data
        Contains all vital information from the ML_AB file
        
    Returns
    -------
    None
    """
    
    if settings.Charges == None:
        types = np.array(data.atomtypes)
        Charges = np.zeros(len(data.atomtypes))
        n_mean = np.zeros(len(data.atomtypes))
        
        for conf, ten in zip(data.configurations,data.tensors):
            for J, aJ in enumerate(conf.atomname):
                JJ = np.argwhere(aJ == types)[0,0]
                Charges[JJ] += np.diagonal(ten[conf.atomtype == J],axis1=-2,axis2=-1).mean()
                n_mean[JJ] += 1
        
        Charges /= n_mean
    else:
        Charges = np.array(settings.Charges)
    settings.Charges = Charges
        
           
def get_weights(settings,maxtype):
    r"""
    This routine generates the vector used for weighting the angular and radial
    descriptors.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    maxtype  : int
        Number of different species
        
    Returns
    -------
    weights : ndarray
        Vector used for weighting the angular and radial descriptors
    """
    M = settings.Nmax*maxtype
    w=np.hstack((np.ones(M)*settings.Beta[0],np.ones(M*M*(settings.Lmax+1))*settings.Beta[1]))
    return w

def norm_lc(lc):
    r"""
    This routine normalizes the local configurations
    
    .. math ::  \textbf{X}_i\cdot \textbf{X}_b \longleftarrow \textbf{X}_i\cdot \textbf{W} \cdot \textbf{X}_b.
    
    Arguments
    ---------
    lc     : ndarray
        Array containing the local configurations
    weights : ndarray
        The weighting vector local configurations
        
    Returns
    -------
    hatlc : ndarray
        Normalized local configurations :math:`\hat{\textbf{X}}_i`
    abslc: ndarray
        Absolut of the local configurations :math:`||\hat{\textbf{X}}_i||`
    """
    abslc = np.linalg.norm(lc, axis=-1)
    hatlc = lc/abslc[...,np.newaxis]
    return hatlc, abslc

def split_vali_Data(Data,percent):
    r'''
    Splits a data set into a training set and a validation set

    Arguments
    ---------
    Data : Training_Data
        Contains all vital information from the ML_AB file
    percent : float
        Percentege of total data used as validation Data 

    Returns
    -------
    Data : Training_Data
        Training set
    vali_Data : Training_Data
        Validation set

    '''
    vali_Data = deepcopy(Data)
    nvali = int(np.round(Data.nconf*percent))
    ntrain= Data.nconf - nvali
    print('Splitting data into '+str(ntrain)+' training points and '+str(nvali)+' validation points.')
    #random
    #vali_idx = np.arange(Data.nconf,dtype=np.int64)
    #np.random.shuffle(vali_idx)
    #vali_idx = vali_idx[:nvali]
    #linear
    vali_idx = np.linspace(0,Data.nconf-1,nvali,dtype=np.int64)
    vali_sel = np.zeros(Data.nconf,dtype=bool)
    vali_sel[vali_idx] = True
    
    vali_Data.configurations = [conf for i, conf in zip(vali_sel,Data.configurations) if i]
    vali_Data.nconf = nvali
    vali_Data.energies = Data.energies[vali_sel]
    vali_Data.forces = [conf for i, conf in zip(vali_sel,Data.forces) if i]
    vali_Data.stresstensors = Data.stresstensors[vali_sel]
    
    Data.configurations = [conf for i, conf in zip(vali_sel,Data.configurations) if not i]
    Data.nconf = ntrain
    Data.energies = Data.energies[~vali_sel]
    Data.forces = [conf for i, conf in zip(vali_sel,Data.forces) if not i]
    Data.stresstensors = Data.stresstensors[~vali_sel]
    
    return Data, vali_Data

def Ang_to_AU(to_convert):
    r'''
    Convertes Angstrom to Bohr radii. Needed for comparebility with vasp.
    
    Arguments
    ---------
    to_convert : ndarray
        Value to convert in Angstrom
    
    Returns
    -------
    out : ndarray
        Value converted to Bohr radii
    '''
    converter = 1.889726125
    return converter*to_convert

def AU_to_Ang(to_convert):
    r'''
    Convertes Bohr radii to Angstrom. Needed for comparebility with vasp.
    
    Arguments
    ---------
    to_convert : ndarray
        Value to convert in Bohr radii
    
    Returns
    -------
    out : ndarray
        Value converted to Angstrom
    '''
    converter = 1.889726125
    return to_convert/converter

def eV_to_AU(to_convert):
    r'''
    Convertes eV to Hartree energies. Needed for comparebility with vasp.
    
    Arguments
    ---------
    to_convert : ndarray
        Value to convert in eV
    
    Returns
    -------
    out : ndarray
        Value converted to Hartree energies
    '''
    converter = 0.036749322176
    return converter*to_convert

def eVperAng_to_AU(to_convert):
    r'''
    Convertes eV per Angstom to Hartree energies per Bohr radii. Needed for comparebility with vasp.
    
    Arguments
    ---------
    to_convert : ndarray
        Value to convert in eV per Angstrom
    
    Returns
    -------
    out : ndarray
        Value converted to Hartree energies per Bohr radii
    '''
    converter = 0.0194469038078203
    return converter*to_convert

def AU_to_eVperAng(to_convert):
    r'''
    Convertes Hartree energies per Bohr radii to eV per Angstom. Needed for comparebility with vasp.
    
    Arguments
    ---------
    to_convert : ndarray
        Value to convert in Hartree energies per Bohr radii
    
    Returns
    -------
    out : ndarray
        Value converted to eV per Angstrom
    '''
    converter = 0.0194469038078203
    return to_convert/converter
    
def AU_to_eV(to_convert):
    r'''
    Convertes eV to Hartree energies. Needed for comparebility with vasp.
    
    Arguments
    ---------
    to_convert : ndarray
        Value to convert in Hartree energies
    
    Returns
    -------
    out : ndarray
        Value converted to eV
    '''
    converter = 0.036749322176
    return to_convert/converter

def conver_set(settings):
    r'''
    Convertes the Setup to atomic units.
    
    Arguments
    ---------
    settings : Setup
        Setup to convert
    Returns
    -------
    out : ndarray
        Converted Setup
    '''
    conset = deepcopy(settings)
    conset.Rcut = Ang_to_AU(conset.Rcut)
    conset.SigmaAtom = Ang_to_AU(conset.SigmaAtom)
    return conset
    
def conver_conf(conf):
    r'''
    Convertes a atomic configuration to atomic units.
    
    Arguments
    ---------
    conf : Configuration
        Configuration to convert
    '''
    conf.lattice = Ang_to_AU(conf.lattice)
    conf.atompos = Ang_to_AU(conf.atompos)
    return conf

def calc_ionic_polarisation(conf,intr_charge):
    intr_charge = np.array(intr_charge)
    return np.sum(intr_charge[conf.atomtype][:,np.newaxis]*conf.atompos,axis=0)

def polarisation_to_minimgcon(P,conf):
    P = np.linalg.inv(conf.lattice).T @ P
    tolarge = P > 0.5
    tosmall = P <= -0.5
    while ~np.all(~tolarge) or ~np.all(~tosmall):
        P[tolarge] -= 1
        P[tosmall] += 1
        tolarge = P > 0.5
        tosmall = P <= -0.5
    return conf.lattice @ P

def fix_polarisation_timeseries(Ps,configurations):
    lattice  = [AU_to_Ang(conf.lattice) for conf in configurations]
    rlattice = [np.linalg.inv(l).T for l in lattice]
    rPs      = [l @ P for l, P in zip(rlattice,Ps)]
    for i in range(1,len(rPs)):
        dP = rPs[i] - rPs[i-1]
        mask = 0.5 < np.abs(dP)
        while np.any(mask) :
            rPs[i][mask] -= np.sign(dP[mask])
            dP = rPs[i] - rPs[i-1]
            mask = 0.5 < np.abs(dP)
    return np.array([l @ P for l, P in zip(lattice,rPs)])

def calc_ionic_borncharges(conf,intr_charge):
    Z = np.empty((conf.natom,3,3))
    for J in range(conf.maxtype):
        Z[conf.atomtype == J] = np.diag([intr_charge[J]]*3)[None]
    return Z
    
    
