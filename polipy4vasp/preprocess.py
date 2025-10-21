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
    post_process_args : tupel
        Arguments needed for postprocessing.
    pre_process_args : tupel
        Arguments needed for preprocessing.
    """
    l_natom = np.array([conf.natom for conf in data.configurations])
    toteneperatom = data.energies/l_natom
    shift_ene = np.mean(toteneperatom)
    shift_stress = np.mean(data.stresstensors)
    Wene = settings.Waderiv/np.std(toteneperatom)
    Wforc = 1/np.std(np.vstack(data.forces))
    Wstress = 1/np.std(data.stresstensors)
    return (shift_ene, shift_stress), [l_natom,shift_ene,shift_stress,Wene,Wforc,Wstress]

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
    post_process_args : tupel
        Arguments needed for postprocessing.
    pre_process_args : scaler
        Scalar for preprocessing.
    """
    
    post_process_args, pre_process_args = data.Type.preprocess(settings,data)
    
    return post_process_args, pre_process_args
           

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
    print('Split data into '+str(ntrain)+' training configurations and '+str(nvali)+' validation configurations.')
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
    Data.configurations = [conf for i, conf in zip(vali_sel,Data.configurations) if not i]
    Data.nconf = ntrain
    
    if Data._ten :
        vali_Data.tensors = [ten for i, ten in zip(vali_sel,Data.tensors) if i]
        
        Data.tensors = [ten for i, ten in zip(vali_sel,Data.tensors) if not i]
    else :
        vali_Data.energies = Data.energies[vali_sel]
        vali_Data.forces = [forc for i, forc in zip(vali_sel,Data.forces) if i]
        vali_Data.stresstensors = Data.stresstensors[vali_sel]
        
        Data.energies = Data.energies[~vali_sel]
        Data.forces = [forc for i, forc in zip(vali_sel,Data.forces) if not i]
        Data.stresstensors = Data.stresstensors[~vali_sel]
    
    return Data, vali_Data


def fix_polarisation_timeseries(Ps,configurations):
    lattice  = [conf.lattice for conf in configurations]
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
