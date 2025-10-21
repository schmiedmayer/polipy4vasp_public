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
        Needs to be run before using `sph_harm` or `sph_harm_grad` to avoid an Error!
    '''
    asa.ylm3st(Lmax)

def sph_harm_grad(Lmax,hatvecr):
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

def sph_harm(Lmax,hatvecr):
    r'''
    Wraped FORTRAN routine "setylm" from the VASP module "asa". Calculates the real spherical harmonics
    :math:`Y_{lm}(\hat{\textbf{r}}_{ij})`for all :math:`l` and :math:`m` up to :math:`l_\text{max}`.
    The parameter :math:`\hat{\textbf{r}}_{ij}` has to lie on the unit sphere. 
    
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

        
    Notes
    -----
    Since :math:`-l \le m \le l` a compound index is used. The index is calculated via
    
    .. math:: l(l+1)+m.
        
    
    '''
    (Mshape,Nshape,_) = hatvecr.shape
    if Nshape > 0:
        Y = asa.setylm(Mshape,Nshape,Lmax,hatvecr)
        
    else:
        lm = (Lmax+1)*(Lmax+1)
        Y = np.zeros([lm,Mshape,Nshape])
    return Y

def sph_harm_grad_lamb(lamb,hatvecr):
    r'''
    Wraped FORTRAN routine "setylm_grad_lamb". Calculates the real spherical harmonics
    :math:`Y_{\lambda m}(\hat{\textbf{r}}_{ij})` and its derivative for all :math:`m` for :math:`l=\lambda`.
    The parameter :math:`\hat{\textbf{r}}_{ij}` has to lie on the unit sphere. To obtain the final derivative one has to
    devide the output by :math:`r_{ij}`
    
    Arguments
    ---------
    lamb : int, scalar, optional
        Order of the tensor :math:`\lambda`. Default 0.
    hatvecr : ndarray
        :math:`\hat{\textbf{r}}_{ij}`
    
    Returns
    -------
    Y : ndarray
        Real spherical harmonics :math:`Y_{lm}(\hat{\textbf{r}}_{ij})`
    dY : ndarray
        Derivaive of real sperical harmonics
        
    
    '''
    (Mshape,Nshape,_) = hatvecr.shape
    if Nshape > 0:
        Y, dY = asa.setylm_grad_lamb(Mshape,Nshape,lamb,hatvecr)
        dY = np.moveaxis(dY,[1,2,3],[3,1,2])
        
    else:
        lm = (lamb+1)*(lamb+1)
        Y = np.zeros([lm,Mshape,Nshape])
        dY = np.zeros([lm,Mshape,Nshape,3])
    return Y, dY

def sph_harm_lamb(lamb,hatvecr):
    r'''
    Wraped FORTRAN routine "setylm_lamb". Calculates the real spherical harmonics
    :math:`Y_{\lambda m}(\hat{\textbf{r}}_{ij})` for all :math:`m` for :math:`l=\lambda`.
    The parameter :math:`\hat{\textbf{r}}_{ij}` has to lie on the unit sphere.
    
    Arguments
    ---------
    lamb : int, scalar, optional
        Order of the tensor :math:`\lambda`. Default 0.
    hatvecr : ndarray
        :math:`\hat{\textbf{r}}_{ij}`
    
    Returns
    -------
    Y : ndarray
        Real spherical harmonics :math:`Y_{lm}(\hat{\textbf{r}}_{ij})`        
    
    '''
    (Mshape,Nshape,_) = hatvecr.shape
    if Nshape > 0:
        Y = asa.setylm_lamb(Mshape,Nshape,lamb,hatvecr)
        
    else:
        lm = (lamb+1)*(lamb+1)
        Y = np.zeros([lm,Mshape,Nshape])
    return Y
    