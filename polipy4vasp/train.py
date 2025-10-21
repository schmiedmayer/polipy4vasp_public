#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def get_w(settings,data,PHI,Y,nlrc):
    r'''
    Calculates the weights :math:`\omega` by solving the equation
    
    .. math:: \Phi \omega = y .
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    data : Training_Data
        Contains all vital information from the ML_AB file.
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
    out : ndarray
        The predicted :math:`\Phi \omega = y`.

    '''
    def solve(phi,y,atomtype=None):
        if phi.shape[0] <= phi.shape[1] : print('WARNING: Desing Matrix over defined reduce LRC!!!')
        w, _, _, singular = np.linalg.lstsq(phi,y,rcond=-1)
        condnum = np.max(singular)/np.min(singular)
        if type(atomtype) == str: print("Condition number of PHI for atom type %2s: %8.1E" % (atomtype, condnum))
        else: print("Condition number of PHI : %8.1E" % condnum)
        o = [np.dot(phi,w),y]
        return w, o
    
    ws = []
    if data._ten :
        if data.Type.summ :
            w, out = solve(PHI,Y)
            n = 2 * data.Type.lamb + 1
            i = 0
            for j in nlrc:
                ws.append(w[n*i:n*(j+i)])
                i += j
        else :
           out = []
           for phi, y, atomname in zip(PHI,Y,data.atomname):
               w, o = solve(phi, y, atomname)
               ws.append(w)
               out.append(o)
    else :
        w, out = solve(PHI,Y)
        i = 0
        for j in nlrc:
            ws.append(w[i:(j+i)])
            i += j
            
    return ws, out

def get_Y(data,pre_process_args):
    r'''
    Returns the large :math:`\textbf{Y}` vector conaining the first principle results needed foor fitting.
    
    Arguments
    ---------
    data : Training_Data
        Contains all vital information from the ML_AB file
    pre_process_args : tupel
        Arguments needed for preprocessing
    
    Returns
    -------
    Y : ndarray
        Vector :math:`\textbf{Y}` conaining the first principle results
    '''
    forc = np.hstack(np.vstack(data.forces))*pre_process_args[4]
    return np.hstack([(data.energies/pre_process_args[0]-pre_process_args[1])*pre_process_args[3],forc])
