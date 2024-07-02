#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
============================================
Spherical harmonics (:mod:`polipy4vasp.sph`)
============================================

.. currentmodule:: polipy4vasp.sph
    
"""

import numpy as np

from vasa import asa


def setup_asa(Lmax):
    r'''
    Wraped FORTRAN routine "ylm3st" from the VASP module "asa". Calculates the real Clepsh Gordan coefficients
    needed for calculating the real spherical harmonics.
    
    Arguments
    ---------
    Lmax : int
        Maximal angular quantum number :math:`l_\text{max}`
    
    
    .. warning::
        Needs to be run before using `sph_harm` to avoid an Error!
    '''
    asa.ylm3st(Lmax)

def sph_harm(Lmax,hatvecr):
    r'''
    Wraped FORTRAN routine "setylm_grad" from the VASP module "asa". Calculates the real spherical harmonics
    :math:`Y_{lm}(\hat{\textbf{r}}_{ij})` and its derivative for all :math:`l` and :math:`m` up to :math:`l_\text{max}`.
    The parameter :math:`\hat{\textbf{r}}_{ij}` has to lie on the unit sphere. To obtain the final derivative one has to
    devide the output by :math:`r_{ij}`
    
    Arguments
    ---------
    Lmax : int
        Maximal angular quantum number :math:`l_\text{max}`
    hatvecr : ndarray
        :math:`\hat{\textbf{r}}_{ij}`
    
    Returns
    -------
    Y : ndarray
        Real spherical harmonics :math:`Y_{lm}(\hat{\textbf{r}}_{ij})`
    dY : ndarray
        Derivaive of real sperical harmonics
        
    Notes
    -----
    Since :math:`-l \le m \le l` a compound index is used. The index is calculated via
    
    .. math:: l(l+1)+m.
        
    
    '''
    (Mshape,Nshape,_) = hatvecr.shape
    if Nshape > 0:
        Y, dY = asa.setylm_grad(Mshape,Nshape,Lmax,hatvecr)
        dY = np.moveaxis(dY,[1,2,3],[3,1,2])
        
    else:
        lm = (Lmax+1)*(Lmax+1)
        Y = np.zeros([lm,Mshape,Nshape])
        dY = np.zeros([lm,Mshape,Nshape,3])
    return Y, dY
