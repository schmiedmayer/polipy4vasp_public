#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================
Routines for computing the desing matrix (:mod:`polipy4vasp.desing`)
====================================================================

.. currentmodule:: polipy4vasp.desing
    
"""

import numpy as np

from .kernel import combine_derivertives,predict_combine_derivertives
from tqdm import tqdm
from joblib import Parallel, delayed
        
def get_phi(settings,glob,descriptor,lrc,maxtype,w=None,predict=False):
    r'''
    Constructes part of the desing matrix :math:`\Phi` or performec the prdiction for one structure if the weights :math:`\omega_{i_B}` are known. 

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    descriptor : Descriptor
        Dataclass containing the descripror, its deriverive, and al vital information for describing the data structure.
    lrc : list
        Local refference configuration
    maxtype : int
        Number of different atom types.
    w : list, optional
        The weights :math:`\omega_{i_B}` used for predicting. The default is None.
    predict : TYPE, optional
        If set to ``True`` this routine will predict the energie and the forces. The default is False.

    Returns
    -------
    E : array
        Energy of one structure.
    F: array
        Forces of one structure.

    '''
    if predict :
        E, F = predict_combine_derivertives(settings,glob,descriptor,lrc[0],0,w[0])
        for J in range(1,maxtype):
            buf_E, buf_F = predict_combine_derivertives(settings,glob,descriptor,lrc[J],J,w[J])
            E += buf_E
            F += buf_F
        return E, -F
    else :
        Ephi = []
        Fphi = []
        for J in range(maxtype):
            nlrc = len(lrc[J])
            E, F = combine_derivertives(settings,glob,descriptor,lrc[J],nlrc,J)
            Ephi.append(E)
            Fphi.append(F)
        return np.concatenate(Ephi,0), -np.concatenate(Fphi,2)


def get_PHI(settings,glob,descriptors,lrc,maxtype):
    r'''
    Constructes the desing matrix :math:`\Phi` needed for training.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    descriptors : list
        List of dataclass Descripror.
    lrc : list
        Local refference configuration
    maxtype : int
        Number of different atom types.

    Returns
    -------
    Phi : array
        The desing matrix

    '''
    nlrc = sum([len(rc) for rc in lrc])
    buf = Parallel(n_jobs=settings.ncore,require='sharedmem')(delayed(get_phi)(settings,glob,desc,lrc,maxtype) for desc in tqdm(descriptors, desc='Bulding PHI'))
    EPHI = np.vstack([b[0] for b in buf])
    FPHI = np.vstack([b[1].reshape(-1,nlrc) for b in buf])
    return np.vstack([EPHI*settings.Wene,FPHI*settings.Wforc])
