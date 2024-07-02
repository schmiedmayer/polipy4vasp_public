#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .tensor_descriptors import ten_dKxdpxdc_predict

def make_fast(settings,lrc,w):
    r"""
    When using linear Kernels this routine makes the prediction faster by making
    the contrection over the weights bevorehand.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
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
    if settings.lamb == None:
        lrc = lrc
        w   = w
    else:
        if settings.lamb >= 0 and settings.Zeta == 0:
            lrc = [np.dot(np.moveaxis(l,(0,1,2),(2,1,0)).reshape(l.shape[2],-1),ww) for l, ww in zip(lrc,w)]
            w   = 0
        if settings.lamb == None and settings.Kernel == 'linear' :
            lrc = [np.dot(ww,l) for l, ww in zip(lrc,w)]
            w   = 0
    return lrc,w
        
def fast_prediction_P(desc,lrc):
    r"""
    Fast telsorial prediction for a linear kernel

    Parameters
    ----------
    desc : Descriptor
        Class containing all the user defined settings for training the MLFF.
    lrc : list
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`.

    Returns
    -------
    P : ndarray
        Predicted tensor.

    """
    P = np.sum([np.dot(np.sum(lc,axis=1),l) for lc, l in zip(desc.lc,lrc)],axis=0)
    return P

def fast_prediction_Z(settings,glob,desc,lrc):
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
    Z : ndarray
        Derivative of predicted tensor.

    """
    Z = np.zeros((desc.natom,3,2*settings.lamb+1))
    for J in range(desc.maxtype):
        Z += ten_dKxdpxdc_predict(settings,glob,desc,lrc[J],J)
    return Z
