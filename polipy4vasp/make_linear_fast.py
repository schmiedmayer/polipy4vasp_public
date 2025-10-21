#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .tensor_descriptors import ten_dKxdpxdc_predict

def make_fast(settings,data,lrc,w):
    r"""
    When using linear Kernels this routine makes the prediction faster by making
    the contrection over the weights bevorehand.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    data : Training_Data
        Contains all vital information from the ML_AB file.
    lrc : list
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`.
    w : list
        The fited weights.

    Returns
    -------
    lrc : list
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`.
    w : int
        0.

    """
    if settings.Kernel == 'linear':
        if data._ten :
            lrc = [np.dot(np.moveaxis(l[0],(0,1,2),(2,1,0)).reshape(l[0].shape[2],-1),ww) for l, ww in zip(lrc,w)]
            w   = 0
        else :
            lrc = [np.dot(ww,l) for l, ww in zip(lrc,w)]
            w   = 0
        
    return lrc,w
        
def fast_prediction(desc,lrc,summ):
    r"""
    Fast telsorial prediction for a linear kernel

    Parameters
    ----------
    desc : Descriptor
        Class containing all the user defined settings for training the MLFF.
    lrc : list
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`.
    summ : bool
        Calculate sum over atoms.

    Returns
    -------
    T : ndarray
        Predicted tensor.

    """
    if summ:
        return np.sum([np.dot(np.sum(lc,axis=1),l) for lc, l in zip(desc.lc,lrc)],axis=0)
    else :
        return np.hstack([np.dot(lc,l) for lc, l in zip(desc.lc,lrc)])
    

def fast_prediction_grad(settings,glob,desc,lrc):
    r"""
    Fast telsorial derivative prediction for a linear kernel

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    glob : Globals
        Class containing all precomputet coefficients.
    desc : Descriptor
        Class containing all the user defined settings for training the MLFF.
    lrc : list
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`.

    Returns
    -------
    dT : ndarray
        Derivative of predicted tensor.

    """
    dT = ten_dKxdpxdc_predict(settings,glob,desc,lrc[0],0)
    for J in range(1,desc.maxtype):
        dT += ten_dKxdpxdc_predict(settings,glob,desc,lrc[J],J)
    return dT
