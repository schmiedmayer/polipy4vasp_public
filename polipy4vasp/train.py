#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eigvals, block_diag

def get_w(settings,PHI,Y,nlrc):
    r'''
    Calculates the weights :math:`\omega` by solving the equation
    
    .. math:: \Phi \omega = y .
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    PHI : ndarray
        Desing matrix :math:`\Phi`
    Y : ndarray
        Ab initio training data :math:`y`.
    nlrc : list
        Number if local refference configurations per atomic species.

    Returns
    -------
    ws : list
        Weights :math:`\omega`.
    singular : ndarray
        Singular vallues.
    out : ndarray
        The predicted :math:`\Phi \omega = y`.

    '''
    if PHI.shape[0] <= PHI.shape[1] : print('WARNING: Desing Matrix over defined reduce LRC!!!')
    w, _, _, singular = np.linalg.lstsq(PHI,Y,rcond=-1)
    out = [np.dot(PHI,w),Y]
    if settings.lamb == None : n = 1
    else : n = 2 * settings.lamb + 1
    ws = []
    i = 0
    for j in nlrc:
        ws.append(w[n*i:n*(j+i)])
        i += j
    return ws, singular, out

def get_Y(settings,data):
    r'''
    Returns the large :math:`\textbf{Y}` vector conaining the first principle results needed foor fitting.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    data : Training_Data
        Contains all vital information from the ML_AB file
    
    Returns
    -------
    Y : ndarray
        Vector :math:`\textbf{Y}` conaining the first principle results
    '''
    forc = np.hstack(np.vstack(data.forces))   
    return np.hstack([data.energies*settings.Wene,forc*settings.Wforc])
