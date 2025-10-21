#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================
Descriptor calculation routines (:mod:`polipy4vasp.descriptors`)
================================================================

.. currentmodule:: polipy4vasp.descriptors
    
"""


import numpy as np
from dataclasses import dataclass
from joblib import Parallel, delayed
from tqdm import tqdm

from .sph import sph_harm, sph_harm_grad, sph_harm_lamb, sph_harm_grad_lamb
from .positions import get_nn_confs, LR_nn_conf
from .LR_descriptors import get_c_LR, get_c_LR_grad

@dataclass
class Descriptor:
    r'''
    Dataclass containing the descripror, its deriverive, and al vital information for describing the data
    structure.
    
    args:
        lc (list) : The local configurations :math:`\textbf{X}_i`.
        dc2 (list) : Derivertives of two-body expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{jJ}_{n\lambda m}`.
        dc (list) : Derivertives of three-body expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{jJ}_{nlm}`.
        self_dc2 (list) : Derivertives of two-body expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{iJ}_{n\lambda m}`.
        self_dc (list) : Derivertives of three-body expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{iJ}_{nlm}`.
        dp (list) : Array of derivertives of the three body expansions coefficients :math:`\frac{\mathrm{d}}{\mathrm{d} c^{jJ}_{nlm}}p^{iJJ'}_{nn'l}`.
        natom (int) : Number of atoms.
        centraltype (list) : Type of the central atom.
        derivtype2 (list) : Atom type of non zero derivertive for two-body.
        derivtype (list) : Atom type of non zero derivertive for three-body.
        derivl2 (list) : Derivertive list for two-body.
        derivl (list) : Derivertive list for three-body.
        derivloc (list) : List containing the location of non zero derivertivs for three-body.
        maxderiv (int) : Maximal number of  nearest neighbors for three-body.
        maxtype (int) : Maximal number of types.
        llc (list) : The linear (:math:`\lambda=0`) local configurations :math:`\textbf{X}_i`. Only used for :math:`\lambda>0` and :math:`\zeta>1`.
        
    notes:
        The nearest neighbors are needed for the derivertives, since all other atom derivertives are zero only
        those in the nearest neighbor list are stored.
        
    '''
    lc : list
    dc2 : list
    dc : list
    self_dc2 : list
    self_dc : list
    dp : list
    natom : int
    centraltype : list
    derivtype2 : list
    derivtype : list
    derivl2 : list
    derivl : list
    derivloc : list
    maxderiv : int
    maxtype : int
    llc : list = None
    
    def __eq__(self,descriptor):
        r'''
        Checkes if two Descriptor are equal.

        Parameters
        ----------
        descriptor : Descriptor
            Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure

        Returns
        -------
        bool
            Returns ``True`` if the two descriptors are equal.

        '''
        for a, b in zip(self.lc,descriptor.lc):
            if not np.all(a == b) :
                print('lc')
                return False
        for a, b in zip(self.self_dc,descriptor.self_dc):
            if not np.all(a == b) :
                print('self_dc')
                return False
        for a, b in zip(self.dp,descriptor.dp):
            if not np.all(a == b) :
                print('dp')
                return False
        for a, b in zip(self.centraltype,descriptor.centraltype):
            if not np.all(a == b) :
                print('centraltype')
                return False
        for a, b in zip(self.derivtype,descriptor.derivtype):
            if not np.all(a == b) :
                print('derivtyper')
                return False
        for a, b in zip(self.derivl,descriptor.derivl):
            if not np.all(a == b) :
                print('derivl')
                return False
        if self.llc == None :
            if self.llc != descriptor.llc:
                return False
        else :
            for a, b in zip(self.llc,descriptor.llc):
                if not np.all(a == b) :
                    return False
        if self.maxderiv != descriptor.maxderiv :
            return False
        if self.maxtype != descriptor.maxtype :
            return False
        if self.natom != descriptor.natom :
            return False
        return True
        


def get_cTilde_grad(settings,nnconf,h):
    r'''
    This routine calculates the pair expansion coefficients :math:`\tilde{c}^{ij}_{nlm}` defined as
    
    .. math:: \tilde{c}^{ij}_{nlm} = Y_{lm}(\hat{\textbf{r}}_{ij}) h_{nl}(r_{ij}),
    
    and its derivertive
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d}r^\alpha_i} \tilde{c}^{ij}_{nlm} = -\left(\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}Y_{lm}(\hat{\textbf{r}}_{ij})\right) h_{nl}(r_{ij}) + Y_{lm}(\hat{\textbf{r}}_{ij})\left(\frac{\mathrm{d}}{\mathrm{d}r_{ij}}h_{nl}(r_{ij})\right)\frac{r_i^\alpha-r_j^\alpha}{r_{ij}}.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    cTilde : ndarray
        Array of pair expansion coefficients :math:`\tilde{c}^{ij}_{nlm}`
    dcTilde : ndarray
        Array of the derivertive of pair expansion coefficients :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}\tilde{c}^{ij}_{nlm}`
    '''
    drij = nnconf.nnvecr/nnconf.nnr[:,:,np.newaxis]
    Y, dY = sph_harm_grad(settings.Lmax,drij)
    dY = dY/nnconf.nnr[np.newaxis,:,:,np.newaxis]
    shnl = h(nnconf.nnr)
    sdhnl = h(nnconf.nnr,1)[:,:,:,:,np.newaxis]*drij[np.newaxis,np.newaxis,:,:,:]
    hnl = np.empty([settings.Nmax3,(settings.Lmax+1)*(settings.Lmax+1),nnconf.natom,nnconf.maxnn])
    dhnl = np.empty([settings.Nmax3,(settings.Lmax+1)*(settings.Lmax+1),nnconf.natom,nnconf.maxnn,3])
    for l in range(settings.Lmax+1):
        hnl[:,l*l:(l+1)*(l+1),:,:] = shnl[:,l,:,:][:,np.newaxis,:,:]
        dhnl[:,l*l:(l+1)*(l+1),:,:,:] = sdhnl[:,l,:,:,:][:,np.newaxis,:,:,:]
    cTilde = Y[np.newaxis,:,:,:]*hnl
    dcTilde = Y[np.newaxis,:,:,:,np.newaxis]*dhnl+dY[np.newaxis,:,:,:,:]*hnl[:,:,:,:,np.newaxis]
    return cTilde, dcTilde

def get_cTilde(settings,nnconf,h):
    r'''
    This routine calculates the pair expansion coefficients :math:`\tilde{c}^{ij}_{nlm}` defined as
    
    .. math:: \tilde{c}^{ij}_{nlm} = Y_{lm}(\hat{\textbf{r}}_{ij}) h_{nl}(r_{ij}).
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    cTilde : ndarray
        Array of pair expansion coefficients :math:`\tilde{c}^{ij}_{nlm}`
    '''
    drij = nnconf.nnvecr/nnconf.nnr[:,:,np.newaxis]
    Y = sph_harm(settings.Lmax,drij)
    shnl = h(nnconf.nnr)
    hnl = np.empty([settings.Nmax3,(settings.Lmax+1)*(settings.Lmax+1),nnconf.natom,nnconf.maxnn])
    for l in range(settings.Lmax+1):
        hnl[:,l*l:(l+1)*(l+1),:,:] = shnl[:,l,:,:][:,np.newaxis,:,:]
    cTilde = Y[np.newaxis,:,:,:]*hnl
    return cTilde

def get_cTilde_grad_2b(settings,nnconf,h):
    r'''
    This routine calculates the pair expansion coefficients :math:`\tilde{c}^{ij}_{n00}` for :math:`l=0` defined as
    
    .. math:: \tilde{c}^{ij}_{n00} = Y_{00}(\hat{\textbf{r}}_{ij}) h_{n0}(r_{ij}),
    
    and its derivertive
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d}r^\alpha_i} \tilde{c}^{ij}_{n00} = Y_{00}(\hat{\textbf{r}}_{ij})\left(\frac{\mathrm{d}}{\mathrm{d}r_{ij}}h_{n0}(r_{ij})\right)\frac{r_i^\alpha-r_j^\alpha}{r_{ij}}.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    cTilde : ndarray
        Array of pair expansion coefficients :math:`\tilde{c}^{ij}_{n00}`
    dcTilde : ndarray
        Array of the derivertive of pair expansion coefficients :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}\tilde{c}^{ij}_{n00}`
    '''
    fac = 0.2820947917738781434 #0.5/sqrt(pi)
    drij = nnconf.nnvecr/nnconf.nnr[:,:,np.newaxis]
    cTilde = h(nnconf.nnr)*fac
    dcTilde = (h(nnconf.nnr,1)*fac)[:,:,:,np.newaxis]*drij[np.newaxis,:,:,:]
    return cTilde, dcTilde

def get_cTilde_2b(settings,nnconf,h):
    r'''
    This routine calculates the pair expansion coefficients :math:`\tilde{c}^{ij}_{n00}` for :math:`l=0` defined as
    
    .. math:: \tilde{c}^{ij}_{n00} = Y_{00}(\hat{\textbf{r}}_{ij}) h_{n0}(r_{ij}).
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    cTilde : ndarray
        Array of pair expansion coefficients :math:`\tilde{c}^{ij}_{n00}`
    dcTilde : ndarray
        Array of the derivertive of pair expansion coefficients :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}\tilde{c}^{ij}_{n00}`
    '''
    fac = 0.2820947917738781434 #0.5/sqrt(pi)
    cTilde = h(nnconf.nnr)*fac
    return cTilde

def get_cTilde_ten_grad_2b(settings,nnconf,h,lamb):
    r'''
    This routine calculates the pair expansion coefficients :math:`\tilde{c}^{ij}_{n\lmabda m}` for :math:`l=\lambda` defined as
    
    .. math:: \tilde{c}^{ij}_{n\lmabda m} = Y_{\lmabda m}(\hat{\textbf{r}}_{ij}) h_{n\lmabda}(r_{ij}),
    
    and its derivertive
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d}r^\alpha_i} \tilde{c}^{ij}_{n\lmabda m} = -\left(\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}Y_{\lmabda m}(\hat{\textbf{r}}_{ij})\right) h_{n\lmabda }(r_{ij}) + Y_{\lmabda m}(\hat{\textbf{r}}_{ij})\left(\frac{\mathrm{d}}{\mathrm{d}r_{ij}}h_{n \lmabda}(r_{ij})\right)\frac{r_i^\alpha-r_j^\alpha}{r_{ij}}.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    lamb : int, scalar, optional
        Order of the tensor :math:`\lambda`.
    
    Returns
    -------
    cTilde : ndarray
        Array of pair expansion coefficients :math:`\tilde{c}^{ij}_{nlm}`
    dcTilde : ndarray
        Array of the derivertive of pair expansion coefficients :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}\tilde{c}^{ij}_{nlm}`
    '''
    drij = nnconf.nnvecr/nnconf.nnr[:,:,np.newaxis]
    Y, dY = sph_harm_grad_lamb(lamb,drij)
    dY = dY/nnconf.nnr[np.newaxis,:,:,np.newaxis]
    hnl = h(nnconf.nnr)
    dhnl = h(nnconf.nnr,1)[:,:,:,np.newaxis]*drij[np.newaxis,:,:,:]
    cTilde = Y[np.newaxis,:,:,:]*hnl[:,np.newaxis,:,:]
    dcTilde = Y[np.newaxis,:,:,:,np.newaxis]*dhnl[:,np.newaxis,:,:,:]+dY[np.newaxis,:,:,:,:]*hnl[:,np.newaxis,:,:,np.newaxis]
    return cTilde, dcTilde

def get_cTilde_ten_2b(settings,nnconf,h,lamb):
    r'''
    This routine calculates the pair expansion coefficients :math:`\tilde{c}^{ij}_{n\lmabda m}` for :math:`l=\lambda` defined as
    
    .. math:: \tilde{c}^{ij}_{n\lmabda m} = Y_{\lmabda m}(\hat{\textbf{r}}_{ij}) h_{n\lmabda}(r_{ij}),
    
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    lamb : int, scalar
        Order of the tensor :math:`\lambda`.
    
    Returns
    -------
    cTilde : ndarray
        Array of pair expansion coefficients :math:`\tilde{c}^{ij}_{n\lambda m}`

    '''
    drij = nnconf.nnvecr/nnconf.nnr[:,:,np.newaxis]
    Y = sph_harm_lamb(lamb,drij)
    hnl = h(nnconf.nnr)
    cTilde = Y[np.newaxis,:,:,:]*hnl[:,np.newaxis,:,:]
    return cTilde

def get_c_grad(settings,nnconf,h):
    r'''
    This routine calculates the expansions coefficients :math:`c^{iJ}_{nlm}` defined as
    
    .. math:: c^{iJ}_{nlm} = \sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{nlm},
    
    where :math:`J` indicates the species of the interacting atom. Also the derivertive is calculated
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}c^{iJ}_{nlm}&=\sum_{\substack{j\neq i \\ j\in J}}\left(\frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}\tilde{c}^{ij}_{nlm}\right),
    
    with the index :math:`k` indecating ether the central atom or a nearest neighbor.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    c : ndarray
        Array of expanssion coefficents :math:`c^{iJ}_{nlm}`
    dc : ndarray
        Array of derivertives of expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{jJ}_{nlm}`
    self_dc : ndarray
        Array of derivertives of expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{iJ}_{nlm}`
    nndo : list
        List of bool arrays for selekting atom nearest neighbor types
    '''
    cTilde,dcTilde = get_cTilde_grad(settings,nnconf,h)
    c = np.zeros([settings.Nmax3,(settings.Lmax+1)*(settings.Lmax+1),nnconf.maxtype,nnconf.natom])
    self_dc = np.zeros([settings.Nmax3,(settings.Lmax+1)*(settings.Lmax+1),nnconf.maxtype,nnconf.natom,3])
    nndo = []
    for J in range(nnconf.maxtype):
        ntypemask = nnconf.nntype == J
        for i, do in enumerate(ntypemask):
            c[:,:,J,i] = np.sum(cTilde[:,:,i,do],axis=-1)
            if nnconf.mult_per_img :
                self_dc[:,:,J,i,:] = -np.sum(dcTilde[:,:,i,do & (nnconf.nnl[i] != i),:],axis=-2)
            else :
                self_dc[:,:,J,i,:] = -np.sum(dcTilde[:,:,i,do,:],axis=-2)
        nndo.append(ntypemask)
    if nnconf.mult_per_img :
        mask = ~np.eye(nnconf.natom, dtype=bool)
        nnconf.nnn = np.ones(nnconf.natom,dtype = np.int32)*(nnconf.natom-1)
        buf_nnl = (np.arange(nnconf.natom,dtype = np.int32)[np.newaxis,:]*np.ones(nnconf.natom,dtype = np.int32)[:,np.newaxis])[mask].reshape(nnconf.natom,-1)
        nnconf.nntype = (nnconf.centraltype[np.newaxis,:]*np.ones(nnconf.natom)[:,np.newaxis])[mask].reshape(nnconf.natom,-1)
        dc = np.zeros([settings.Nmax3,(settings.Lmax+1)*(settings.Lmax+1),nnconf.natom,nnconf.natom-1,3])
        for j in range(nnconf.natom) :
            for i in range(nnconf.natom-1) :
                dc[:,:,j,i,:] = np.sum(dcTilde[:,:,j,nnconf.nnl[j] == buf_nnl[j][i],:], axis = 2)
        nnconf.nnl = buf_nnl
        nndo = [nnconf.nntype == J for J in range(nnconf.maxtype)]
    else :
        dc = dcTilde
    return c, dc, self_dc, nndo

def get_c(settings,nnconf,h):
    r'''
    This routine calculates the expansions coefficients :math:`c^{iJ}_{nlm}` defined as
    
    .. math:: c^{iJ}_{nlm} = \sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{nlm},
    
    where :math:`J` indicates the species of the interacting atom. 
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    c : ndarray
        Array of expanssion coefficents :math:`c^{iJ}_{nlm}`
    '''
    cTilde = get_cTilde(settings,nnconf,h)
    c = np.zeros([settings.Nmax3,(settings.Lmax+1)*(settings.Lmax+1),nnconf.maxtype,nnconf.natom])
    for J in range(nnconf.maxtype):
        ntypemask = nnconf.nntype == J
        for i, do in enumerate(ntypemask):
            c[:,:,J,i] = np.sum(cTilde[:,:,i,do],axis=-1)       
    return c

def get_c_grad_2b(settings,nnconf,h):
    r'''
    This routine calculates the expansions coefficients :math:`c^{iJ}_{n00}` defined as
    
    .. math:: c^{iJ}_{n00} = \sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{n00},
    
    where :math:`J` indicates the species of the interacting atom. Also the derivertive is calculated
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}c^{iJ}_{n00}&=\sum_{\substack{j\neq i \\ j\in J}}\left(\frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}\tilde{c}^{ij}_{n00}\right),
    
    with the index :math:`k` indecating ether the central atom or a nearest neighbor.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    c : ndarray
        Array of expanssion coefficents :math:`c^{iJ}_{n00}`
    dc : ndarray
        Array of derivertives of expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{jJ}_{n00}`
    self_dc : ndarray
        Array of derivertives of expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{iJ}_{n00}`
    nndo : list
        List of bool arrays for selekting atom nearest neighbor types
    '''
    cTilde,dcTilde = get_cTilde_grad_2b(settings,nnconf,h)
    c = np.zeros([settings.Nmax2,nnconf.maxtype,nnconf.natom])
    self_dc = np.zeros([settings.Nmax2,nnconf.maxtype,nnconf.natom,3])
    nndo = []
    for J in range(nnconf.maxtype):
        ntypemask = nnconf.nntype == J
        for i, do in enumerate(ntypemask):
            c[:,J,i] = np.sum(cTilde[:,i,do],axis=-1)
            if nnconf.mult_per_img :
                self_dc[:,J,i,:] = -np.sum(dcTilde[:,i,do & (nnconf.nnl[i] != i),:],axis=-2)
            else :
                self_dc[:,J,i,:] = -np.sum(dcTilde[:,i,do,:],axis=-2)     
        nndo.append(ntypemask)
    if nnconf.mult_per_img :
        mask = ~np.eye(nnconf.natom, dtype=bool)
        nnconf.nnn = np.ones(nnconf.natom,dtype = np.int32)*(nnconf.natom-1)
        buf_nnl = (np.arange(nnconf.natom,dtype = np.int32)[np.newaxis,:]*np.ones(nnconf.natom,dtype = np.int32)[:,np.newaxis])[mask].reshape(nnconf.natom,-1)
        nnconf.nntype = (nnconf.centraltype[np.newaxis,:]*np.ones(nnconf.natom)[:,np.newaxis])[mask].reshape(nnconf.natom,-1)
        dc = np.zeros([settings.Nmax2,nnconf.natom,nnconf.natom-1,3])
        for j in range(nnconf.natom) :
            for i in range(nnconf.natom-1) :
                dc[:,j,i,:] = np.sum(dcTilde[:,j,nnconf.nnl[j] == buf_nnl[j][i],:], axis = 1)
        nnconf.nnl = buf_nnl
        nndo = [nnconf.nntype == J for J in range(nnconf.maxtype)]
    else :
        dc = dcTilde
    return c, dc, self_dc, nndo

def get_c_2b(settings,nnconf,h):
    r'''
    This routine calculates the expansions coefficients :math:`c^{iJ}_{n00}` defined as
    
    .. math:: c^{iJ}_{n00} = \sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{n00},
    
    where :math:`J` indicates the species of the interacting atom.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    
    Returns
    -------
    c : ndarray
        Array of expanssion coefficents :math:`c^{iJ}_{n00}`

    '''
    cTilde = get_cTilde_2b(settings,nnconf,h)
    c = np.zeros([settings.Nmax2,nnconf.maxtype,nnconf.natom])

    for J in range(nnconf.maxtype):
        ntypemask = nnconf.nntype == J
        for i, do in enumerate(ntypemask):
            c[:,J,i] = np.sum(cTilde[:,i,do],axis=-1)
            
    return c

def get_c_ten_grad_2b(settings,nnconf,h,lamb):
    r'''
    This routine calculates the expansions coefficients :math:`c^{iJ}_{n\lambda m}` for :math:`l = \lambda` defined as
    
    .. math:: c^{iJ}_{n\lambda m} = \sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{n\lambda m},
    
    where :math:`J` indicates the species of the interacting atom. Also the derivertive is calculated
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}c^{iJ}_{n\lambda m}&=\sum_{\substack{j\neq i \\ j\in J}}\left(\frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}\tilde{c}^{ij}_{n\lambda m}\right),
    
    with the index :math:`k` indecating ether the central atom or a nearest neighbor.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    lamb : int, scalar
        Order of the tensor :math:`\lambda`.
    
    Returns
    -------
    c : ndarray
        Array of expanssion coefficents :math:`c^{iJ}_{n\lambda m}`
    dc : ndarray
        Array of derivertives of expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{jJ}_{n\lambda m}`
    self_dc : ndarray
        Array of derivertives of expanssion coefficents :math:`\frac{\mathrm{d}}{\mathrm{d}r^\alpha_i}c^{iJ}_{n\lambda m}`
    nndo : list
        List of bool arrays for selekting atom nearest neighbor types
    '''
    cTilde,dcTilde = get_cTilde_ten_grad_2b(settings,nnconf,h,lamb)
    c = np.zeros([settings.Nmax2,(2*lamb+1),nnconf.maxtype,nnconf.natom])
    self_dc = np.zeros([settings.Nmax2,(2*lamb+1),nnconf.maxtype,nnconf.natom,3])
    nndo = []
    for J in range(nnconf.maxtype):
        ntypemask = nnconf.nntype == J
        for i, do in enumerate(ntypemask):
            c[:,:,J,i] = np.sum(cTilde[:,:,i,do],axis=-1)
            if nnconf.mult_per_img :
                self_dc[:,:,J,i,:] = -np.sum(dcTilde[:,:,i,do & (nnconf.nnl[i] != i),:],axis=-2)
            else :
                self_dc[:,:,J,i,:] = -np.sum(dcTilde[:,:,i,do,:],axis=-2)       
        nndo.append(ntypemask)
    if nnconf.mult_per_img :
        mask = ~np.eye(nnconf.natom, dtype=bool)
        nnconf.nnn = np.ones(nnconf.natom,dtype = np.int32)*(nnconf.natom-1)
        buf_nnl = (np.arange(nnconf.natom,dtype = np.int32)[np.newaxis,:]*np.ones(nnconf.natom,dtype = np.int32)[:,np.newaxis])[mask].reshape(nnconf.natom,-1)
        nnconf.nntype = (nnconf.centraltype[np.newaxis,:]*np.ones(nnconf.natom)[:,np.newaxis])[mask].reshape(nnconf.natom,-1)
        dc = np.zeros([settings.Nmax2,(2*lamb+1),nnconf.natom,nnconf.natom-1,3])
        for j in range(nnconf.natom) :
            for i in range(nnconf.natom-1) :
                dc[:,:,j,i,:] = np.sum(dcTilde[:,:,j,nnconf.nnl[j] == buf_nnl[j][i],:], axis = 2)
        nnconf.nnl = buf_nnl
        nndo = [nnconf.nntype == J for J in range(nnconf.maxtype)]
    else :
        dc = dcTilde
    return c, dc, self_dc, nndo

def get_c_ten_2b(settings,nnconf,h,lamb):
    r'''
    This routine calculates the expansions coefficients :math:`c^{iJ}_{n\lambda m}` for :math:`l = \lambda` defined as
    
    .. math:: c^{iJ}_{n\lambda m} = \sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{n\lambda m},
    
    where :math:`J` indicates the species of the interacting atom. 
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    lamb : int, scalar
        Order of the tensor :math:`\lambda`.
    
    Returns
    -------
    c : ndarray
        Array of expanssion coefficents :math:`c^{iJ}_{nlm}`
    '''
    cTilde = get_cTilde_ten_2b(settings,nnconf,h,lamb)
    c = np.zeros([settings.Nmax2,(2*lamb+1),nnconf.maxtype,nnconf.natom])
    for J in range(nnconf.maxtype):
        ntypemask = nnconf.nntype == J
        for i, do in enumerate(ntypemask):
            c[:,:,J,i] = np.sum(cTilde[:,:,i,do],axis=-1)       
    return c

def get_p_grad(settings,glob,nnconf,c):
    r'''
    This routine calculates the three body expansions coefficients :math:`p^{iJJ'}_{nn'l}` defined as
    
    .. math:: p^{iJJ'}_{nn'l} = \sqrt{\frac{8\pi^2}{2l+1}}\sum_{m=-l}^{l}\left[c^{iJ}_{nlm}c^{iJ'}_{n'lm}-\delta_{JJ'}\sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{nlm}\tilde{c}^{ij}_{n'lm}\right],
    
    and its derivative with respect to :math:`c^{iJ}_{nlm}`.

    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    c : ndarray
        Array of expanssion coefficients :math:`c^{iJ}_{nlm}`

    Returns
    -------
    p : ndarray
        Array of the three body expansions coefficients :math:`p^{iJJ'}_{nn'l}`
    dp: ndarray
        Array of derivertives of the three body expansions coefficients :math:`\frac{\mathrm{d}}{\mathrm{d} c^{jJ}_{nlm}}p^{iJJ'}_{nn'l}`
    '''
    buf = np.multiply(c[:,np.newaxis,:,:,np.newaxis,:],c[np.newaxis,:,:,np.newaxis,:,:])
    p = np.empty([settings.Nmax3,settings.Nmax3,settings.Lmax+1,nnconf.maxtype,nnconf.maxtype,nnconf.natom])
    dp = np.empty_like(c)
    for l in range(settings.Lmax+1):
        p[:,:,l,:,:,:] = np.sum(buf[:,:,l*l:(l+1)*(l+1),:,:,:],axis=2)
        dp[:,l*l:(l+1)*(l+1),:,:]=glob.fac[0,l,0,0]*c[:,l*l:(l+1)*(l+1),:,:]
    p = glob.fac[np.newaxis,:,:,:,:,np.newaxis]*p
    return p, dp

def get_p(settings,glob,nnconf,c):
    r'''
    This routine calculates the three body expansions coefficients :math:`p^{iJJ'}_{nn'l}` defined as
    
    .. math:: p^{iJJ'}_{nn'l} = \sqrt{\frac{8\pi^2}{2l+1}}\sum_{m=-l}^{l}\left[c^{iJ}_{nlm}c^{iJ'}_{n'lm}-\delta_{JJ'}\sum_{\substack{j\neq i \\ j\in J}}\tilde{c}^{ij}_{nlm}\tilde{c}^{ij}_{n'lm}\right].

    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    c : ndarray
        Array of expanssion coefficients :math:`c^{iJ}_{nlm}`

    Returns
    -------
    p : ndarray
        Array of the three body expansions coefficients :math:`p^{iJJ'}_{nn'l}`
    '''
    buf = np.multiply(c[:,np.newaxis,:,:,np.newaxis,:],c[np.newaxis,:,:,np.newaxis,:,:])
    p = np.empty([settings.Nmax3,settings.Nmax3,settings.Lmax+1,nnconf.maxtype,nnconf.maxtype,nnconf.natom])
    for l in range(settings.Lmax+1):
        p[:,:,l,:,:,:] = np.sum(buf[:,:,l*l:(l+1)*(l+1),:,:,:],axis=2)
    p = glob.fac[np.newaxis,:,:,:,:,np.newaxis]*p
    return p

def get_p_ten_grad(settings,glob,nnconf,c,lamb):
    r'''
    This routine calculates the three body expansions coefficients :math:`p_{nn'll'}^{\mu iJJ'}` defined as
    
    .. math:: p_{nn'll'}^{\mu iJJ'} =\sqrt{\frac{8\pi^2}{2l+1}}\sum_{m=-l}^{l}\sum_{m'=-l'}^{l'}c^{iJ}_{nlm}c^{iJ'}_{n'l'm'}\langle\lambda\mu l' m'|lm\rangle,
    
    and its derivative with respect to :math:`c^{iJ}_{nlm}`.

    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    c : ndarray
        Array of expanssion coefficients :math:`c^{iJ}_{nlm}`
    lamb : int, scalar
        Order of the tensor :math:`\lambda`.

    Returns
    -------
    p : ndarray
        Array of the three body expansions coefficients :math:`p_{nn'll'}^{\mu iJJ'}`
    dp: ndarray
        Array of derivertives of the three body expansions coefficients :math:`\frac{\mathrm{d}}{\mathrm{d} c^{jJ}_{nlm}}p_{nn'll'}^{\mu iJJ'}`
    '''
    p = np.zeros([2*lamb+1,settings.Nmax3,settings.Nmax3,glob.len_m_index_sum,nnconf.maxtype,nnconf.maxtype,nnconf.natom])
    dp = np.zeros([2,2*lamb+1,settings.Nmax3,glob.max_len_cleb,nnconf.maxtype,nnconf.natom])
    for k, ind in enumerate(glob.c_index):
        buf = np.multiply(c[:,np.newaxis,ind[0],:,np.newaxis,:],c[np.newaxis,:,ind[1],np.newaxis,:,:])*glob.cleb[k]
        dp[0,k,:,:glob.len_cleb[k]]= c[:,ind[1]]*glob.cleb[k][:,0,:,:,0,:]
        dp[1,k,:,:glob.len_cleb[k]]= c[:,ind[0]]*glob.cleb[k][:,0,:,:,0,:]
        for l, m in enumerate(glob.m_index_sum[k]):
            p[k,:,:,l] = np.sum(buf[:,:,m],axis=2)
    return p, dp

def get_p_ten(settings,glob,nnconf,c,lamb):
    r'''
    This routine calculates the three body expansions coefficients :math:`p_{nn'll'}^{\mu iJJ'}` defined as
    
    .. math:: p_{nn'll'}^{\mu iJJ'} =\sqrt{\frac{8\pi^2}{2l+1}}\sum_{m=-l}^{l}\sum_{m'=-l'}^{l'}c^{iJ}_{nlm}c^{iJ'}_{n'l'm'}\langle\lambda\mu l' m'|lm\rangle.
    

    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    nnconf : NN_Configuration
        Class containing the information of a configuration in a nearest neighbor sense
    c : ndarray
        Array of expanssion coefficients :math:`c^{iJ}_{nlm}`
    lamb : int, scalar
        Order of the tensor :math:`\lambda`.

    Returns
    -------
    p : ndarray
        Array of the three body expansions coefficients :math:`p_{nn'll'}^{\mu iJJ'}`
    
    '''
    p = np.zeros([2*lamb+1,settings.Nmax3,settings.Nmax3,glob.len_m_index_sum,nnconf.maxtype,nnconf.maxtype,nnconf.natom])
    for k, ind in enumerate(glob.c_index):
        buf = np.multiply(c[:,np.newaxis,ind[0],:,np.newaxis,:],c[np.newaxis,:,ind[1],np.newaxis,:,:])*glob.cleb[k]
        for l, m in enumerate(glob.m_index_sum[k]):
            p[k,:,:,l] = np.sum(buf[:,:,m],axis=2)
    return p

def get_Descriptor(settings,glob,configuration,h):
    r'''
    This routine calculates the normalized weighted local configurations (descriptors) :math:`\hat{\textbf{X}}_i` which
    descibe the local enviroment arrount an atom using :math:`c^{iJ}_{n00}` and :math:`p^{iJJ'}_{nn'l}`
    
    .. math:: \textbf{X}_i&=\begin{pmatrix}c^{i1}_1 \\ c^{i1}_2 \\\vdots \\ c^{i2}_1 \\c^{i2}_2 \\\vdots \\ p^{i11}_{110} \\ p^{i11}_{111} \\ \vdots \\ p^{i11}_{120} \\ p^{i11}_{121} \\\vdots \\p^{i12}_{110} \\ \vdots \\ p^{i22}_{110} \\ \vdots \end{pmatrix}
    
    Also the deriverive of the weighted local configurations is computed.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    configuration : Configuration
        An atomic configuration
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
        
    Returns
    -------
    descriptor : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    '''
    nnconf2, nnconf = get_nn_confs(settings,configuration)
    c2, dc2, self_dc2, nndo2 = get_c_grad_2b(settings,nnconf2,h[0])
    c, dc, self_dc, nndo = get_c_grad(settings,nnconf,h[1])
    do = [nnconf.centraltype == J for J in range(nnconf.maxtype)]
    p, dp = get_p_grad(settings,glob,nnconf,c)
    c2 = np.moveaxis(c2.reshape(-1, nnconf.natom), 0, -1)
    p = np.moveaxis(p.reshape(-1, nnconf.natom), 0, -1)
    lc = np.hstack((np.sqrt(settings.Beta)*c2, np.sqrt(1-settings.Beta)*p))
    dp = [dp[:,:,:,d] for d in do]
    lc = [lc[d] for d in do]
    return Descriptor(lc = lc,
                      dc2 = [dc2[:,d] for d in do],
                      dc = [dc[:,:,d] for d in do],
                      self_dc2= [self_dc2[:,:,d] for d in do],
                      self_dc= [self_dc[:,:,:,d] for d in do],
                      dp = dp,
                      natom = nnconf.natom,
                      centraltype = do,
                      derivtype2 = nndo2,
                      derivtype = nndo,
                      derivl2 = nnconf2.nnl,
                      derivl = nnconf.nnl,
                      derivloc = nnconf.nnn,
                      maxderiv = nnconf.maxnn,
                      maxtype = nnconf.maxtype)

def get_TenDescriptor_grad(settings,glob,configuration,h,lamb):
    r'''
    This routine calculates the tensorial local configurations (descriptors) :math:`\hat{\textbf{X}}^\lambda_i` which
    descibe the local enviroment arrount an atom using :math:`p^{\lambda iJJ'}_{nn'l}`
    
    .. math:: \textbf{X}_i&=\begin{pmatrix}c^{i1}_1 \\ c^{i1}_2 \\\vdots \\ c^{i2}_1 \\c^{i2}_2 \\\vdots \\ p^{i11}_{110} \\ p^{i11}_{111} \\ \vdots \\ p^{i11}_{120} \\ p^{i11}_{121} \\\vdots \\p^{i12}_{110} \\ \vdots \\ p^{i22}_{110} \\ \vdots \end{pmatrix}
    
    Also the deriverive of the local configurations is computed.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    configuration : Configuration
        An atomic configuration
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    lamb : int
        Order of the tensor.
        
    Returns
    -------
    descriptor : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    '''
    llc = None
    nnconf2, nnconf = get_nn_confs(settings,configuration)
    c2, dc2, self_dc2, nndo2 = get_cTilde_ten_grad_2b(settings,nnconf2,h[0],lamb)
    c, dc, self_dc, nndo = get_c_grad(settings,nnconf,h[1])
    do = [nnconf.centraltype == J for J in range(nnconf.maxtype)]
    p, dp = get_p_ten_grad(settings,glob,nnconf,c,lamb)
    c2 = np.moveaxis(np.moveaxis(c2, 1, 0).reshape(2*lamb+1,-1, nnconf.natom), 1, 2)
    p = np.moveaxis(p.reshape(2*lamb+1,-1, nnconf.natom), 1, 2)
    lc = np.concatenate((np.sqrt(settings.Beta)*c2, np.sqrt(1-settings.Beta)*p),axis = 2)
    dp = [dp[:,:,:,:,:,d] for d in do]
    lc = [lc[:,d] for d in do]
    if not settings.Kernel == 'linear' and settings.Zeta > 1 :
        llc = np.moveaxis(get_p_grad(settings,glob,nnconf,c)[0].reshape(-1, nnconf.natom), 0, -1)
        llc = [llc[d] for d in do]
    return Descriptor(lc = lc,
                      dc2 = [dc2[:,:,d] for d in do],
                      dc = [dc[:,:,d] for d in do],
                      self_dc2= [self_dc2[:,:,:,d] for d in do],
                      self_dc= [self_dc[:,:,:,d] for d in do],
                      dp = dp,
                      natom = nnconf.natom,
                      centraltype = do,
                      derivtype2 = nndo2,
                      derivtype = nndo,
                      derivl2 = nnconf2.nnl,
                      derivl = nnconf.nnl,
                      derivloc = nnconf.nnn,
                      maxderiv = nnconf.maxnn,
                      maxtype = nnconf.maxtype,
                      llc = llc)

def get_TenDescriptor(settings,glob,configuration,h,lamb):
    r'''
    This routine calculates the tensorial local configurations (descriptors) :math:`\hat{\textbf{X}}^\lambda_i` which
    descibe the local enviroment arrount an atom using :math:`p^{\lambda iJJ'}_{nn'l}`
    
    .. math:: \textbf{X}_i&=\begin{pmatrix}c^{i1}_1 \\ c^{i1}_2 \\\vdots \\ c^{i2}_1 \\c^{i2}_2 \\\vdots \\ p^{i11}_{110} \\ p^{i11}_{111} \\ \vdots \\ p^{i11}_{120} \\ p^{i11}_{121} \\\vdots \\p^{i12}_{110} \\ \vdots \\ p^{i22}_{110} \\ \vdots \end{pmatrix}.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    configuration : Configuration
        An atomic configuration
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    lamb : int
        Order of the tensor.
        
    Returns
    -------
    descriptor : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    '''
    llc = None
    nnconf2, nnconf = get_nn_confs(settings,configuration)
    c2 = get_c_ten_2b(settings,nnconf,h[0],lamb)
    if settings.LR :
        nnconf = LR_nn_conf(configuration)
        c = get_c_LR(settings,configuration)
    else :
        c = get_c(settings,nnconf,h[1])
    do = [nnconf.centraltype == J for J in range(nnconf.maxtype)]
    p = get_p_ten(settings,glob,nnconf,c,lamb)
    c2 = np.moveaxis(np.moveaxis(c2, 1, 0).reshape(2*lamb+1,-1, nnconf.natom), 1, 2)
    p = np.moveaxis(p.reshape(2*lamb+1,-1, nnconf.natom), 1, 2)
    lc = np.concatenate((np.sqrt(settings.Beta)*c2, np.sqrt(1-settings.Beta)*p),axis = 2)
    lc = [lc[:,d] for d in do]
    if not settings.Kernel == 'linear' and settings.Zeta > 1 :
        llc = np.moveaxis(get_p(settings,glob,nnconf,c).reshape(-1, nnconf.natom), 0, -1)
        llc = [llc[d] for d in do]
    return Descriptor(lc = lc,
                      dc2 = None,
                      dc = None,
                      self_dc2= None,
                      self_dc= None,
                      dp = None,
                      natom = nnconf.natom,
                      centraltype = do,
                      derivtype2 = None,
                      derivtype = None,
                      derivl2 = None,
                      derivl = None,
                      derivloc = None,
                      maxderiv = None,
                      maxtype = nnconf.maxtype,
                      llc = llc)
        

def get_AllDescriptors(settings,glob,configurations,h):
    r'''
    This routine calculates all the normalized weighted local configurations (descriptors) :math:`\hat{\textbf{X}}_i` which
    descibe the local enviroment arrount an atom. Also the deriverive of the weighted local configurations are computed. 
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    configurations : list
        list containing multible atomic configuration
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
        
    Returns
    -------
    descriptors : list
        List containing all the local configuration, its derivertive, and all necercery information for unraveling the data structure
    '''
    return [get_Descriptor(settings,glob,conf,h) for conf in tqdm(configurations , desc='Preparing descriptors')]
    #return Parallel(n_jobs=settings.ncore,require='sharedmem')(delayed(get_Descriptor)(settings,glob,conf,h) for conf in tqdm(configurations , desc='Preparing descriptors'))

def get_AllTenDescriptors(settings,glob,configurations,h,ten_type):
    r'''
    This routine calculates all the normalized weighted local configurations (descriptors) :math:`\hat{\textbf{X}}_i` which
    descibe the local enviroment arrount an atom. Also the deriverive of the weighted local configurations are computed. 
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    configurations : list
        list containing multible atomic configuration
    h : CubicSpline
        Cubic spline function generated by :func:`polipy4vasp.splines.get_splines`
    ten_type : Tensor_Type
        Information on the tensor to learn.
        
    Returns
    -------
    descriptors : list
        List containing all the local configuration, its derivertive, and all necercery information for unraveling the data structure
    '''
    if ten_type.deriv:
        return [get_TenDescriptor_grad(settings,glob,conf,h,ten_type.lamb) for conf in configurations]
    else :
        return [get_TenDescriptor(settings,glob,conf,h,ten_type.lamb) for conf in configurations]
