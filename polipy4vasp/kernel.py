#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .preprocess import norm_lc

def K(X_i,X_b,settings):
    r'''
    Routine for computing the polinomial kernel defined as [#]_
    
    .. math:: K(\textbf{X}_i,\textbf{X}_b) = \left(\hat{\textbf{X}}_i\cdot\hat{\textbf{X}}_b\right)^{\zeta}.
    
    Arguments
    ---------
    X_i : array
        Local configuration :math:`\textbf{X}_i`
    X_b : array
        Local configuration :math:`\textbf{X}_b` 
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    Returns
    -------
    K  : ndarray
        The polinomial kernel :math:`K(\textbf{X}_i,\textbf{X}_b)`
    
    References
    ----------
    .. [#] https://doi.org/10.1063/5.0009491
    '''
    if X_i.ndim == 1 : X_i = X_i[np.newaxis,:]
    if X_b.ndim == 1 : X_b = X_b[np.newaxis,:]
    if settings.Kernel == "poli" :
        hatX_i, _ = norm_lc(X_i)
        hatX_b, _ = norm_lc(X_b)
        K = np.dot(hatX_i,hatX_b.T)**settings.Zeta
    if settings.Kernel == "gaus" :
        K = np.exp(-np.linalg.norm((X_i[:,np.newaxis]-X_b[np.newaxis]),axis=-1)**2/settings.SigmaExp)
    if settings.Kernel == "linear" :
        K = np.dot(X_i,X_b.T)
    if settings.Kernel == "ngaus" :
        hatX_i, _ = norm_lc(X_i)
        hatX_b, _ = norm_lc(X_b)
        K = np.exp(-np.linalg.norm((hatX_i[:,np.newaxis]-hatX_b[np.newaxis]),axis=-1)**2/settings.SigmaExp)
    return K.squeeze()

def K_s(X,settings):
    r'''
    Routine for computing the polinomial kernel of the same structure defined as
    
    .. math:: K(\textbf{X},\textbf{X}) = \left(\hat{\textbf{X}}\cdot\hat{\textbf{X}}\right)^{\zeta}=1.
    
    Arguments
    ---------
    X : array
        Local configuration :math:`\textbf{X}`
 
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    Returns
    -------
    K  : ndarray
        The polinomial kernel :math:`K(\textbf{X},\textbf{X})`
    '''    
    if settings.Kernel == "poli" :
        K = np.ones(len(X))
    if settings.Kernel == "gaus" :
        K = np.ones(len(X))
    if settings.Kernel == "linear" :
        K = np.einsum('ij,ij->i',X,X)
    if settings.Kernel == "ngaus" :
        K = np.ones(len(X))
    return K

def get_dK(desc,lrc,settings,J):
    r'''
    Routine for computing the polinomial kernel defined as [#]_
    
    .. math:: K(\textbf{X}_i,\textbf{X}_b) = \left(\hat{\textbf{X}}_i\cdot\hat{\textbf{X}}_b\right)^{\zeta},
    
    and its derivertive
    
    .. math:: \nabla_{r_k^\alpha} \textbf{K}(\textbf{X}_i,\textbf{X}_b)=\textbf{K}(\textbf{X}_i,\textbf{X}_b)\zeta\frac{1}{\left|\textbf{X}_i\right|}\left[\frac{1}{\hat{\textbf{X}}_i\cdot\hat{\textbf{X}}_b}\left(\frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}\textbf{X}_i\right) \cdot \hat{\textbf{X}}_b-\hat{\textbf{X}}_i\cdot\left(\frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}\textbf{X}_i\right)\right].
    
    Arguments
    ---------
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    lrc  : ndarray
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`
    zeta : scalar
        Power of the polinomial kernel :math:`\zeta`
    J    : int
        Type index :math:`J` for specifing the atom type
        
    Returns
    -------
    K  : ndarray
        The polinomial kernel :math:`K(\textbf{X}_i,\textbf{X}_b)`
    dK : ndarray
        The derivertive of the polinomial kernel :math:`\nabla_{r_k^\alpha}K(\textbf{X}_i,\textbf{X}_b)`
    
    References
    ----------
    .. [#] https://doi.org/10.1063/5.0009491
    '''
    
    if settings.Kernel == "poli" :
        hatlc, abslc = norm_lc(desc.lc[J])
        hatlrc, _    = norm_lc(lrc)
        buf = np.dot(hatlc,hatlrc.T)
        K = buf**settings.Zeta
        dK = (K*(settings.Zeta/abslc)[:,np.newaxis])[:,:,np.newaxis]*((1/buf)[:,:,np.newaxis]*hatlrc[np.newaxis,:,:]-hatlc[:,np.newaxis,:])

    if settings.Kernel == "gaus" :
        buf = (desc.lc[J][:,np.newaxis]-lrc[np.newaxis])
        K = np.exp(-np.linalg.norm(buf,axis=-1)**2/settings.SigmaExp)
        dK = (-2*K/settings.SigmaExp)[:,:,np.newaxis]*buf

    if settings.Kernel == "linear" :
        K = np.dot(desc.lc[J],lrc.T)
        dK = np.empty([*K.shape,len(lrc[0])])
        dK[:,:,:] = lrc[np.newaxis,:,:]

    if settings.Kernel == "ngaus" :
        hatlc, abslc = norm_lc(desc.lc[J])
        hatlrc, _    = norm_lc(lrc)
        dhatlc = (np.eye())
        buf = (hatlc[:,np.newaxis]-hatlrc[np.newaxis])
        K = np.exp(-np.linalg.norm(buf,axis=-1)**2/settings.SigmaExp)
        dK = (-2*K/settings.SigmaExp)[:,:,np.newaxis]*buf

    return K, dK

def radial_part(desc,dK,nlrc,J):
    out = np.zeros((desc.natom,3,nlrc))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('ln,nda->dal',dK[i,:,Jp::desc.maxtype],desc.dc[J][:,0,i,do,:], optimize = True)
            #out[dl[i,do]] += np.tensordot(np.swapaxes(dK[i,:,Jp::desc.maxtype],0,1)[:,np.newaxis,np.newaxis,:],desc.dc[J][:,0,i,do,:,np.newaxis],axes=([0],[0]))
        out[desc.centraltype[J]] += np.einsum('iln,nia->ial',dK[:,:,Jp::desc.maxtype],desc.self_dc[J][:,0,Jp,:], optimize = True)    
    return out

def dKxdp(settings,glob,desc,dK,J):
    size = dK.shape[:2]
    dK = dK.reshape(*size,settings.Nmax,settings.Nmax,settings.Lmax+1,desc.maxtype,desc.maxtype)#[:,:,:,:,lm_to_l,:,:]
    out = np.zeros((settings.Nmax,(settings.Lmax+1)**2,desc.maxtype,*size))
    for lm, dp in enumerate(np.rollaxis(desc.dp[J], 1)):
        out[:,lm,...] = np.einsum('ilnpJq,nJi->pqil',dK[...,glob.lm_to_l[lm],:,:],dp, optimize = True) + np.einsum('ilpnqJ,nJi->pqil',dK[...,glob.lm_to_l[lm],:,:],dp, optimize = True)
    # lm = m, np = p, Jp = q

    return out

def angular_part(settings,glob,desc,dK,nlrc,J):
    buf = dKxdp(settings,glob,desc,dK,J)
    out = np.zeros((desc.natom,3,nlrc))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('nml,nmda->dal',buf[:,:,Jp,i,:],desc.dc[J][:,:,i,do,:], optimize = True)
        out[desc.centraltype[J]] += np.einsum('nmil,nmia->ial',buf[:,:,Jp,:,:],desc.self_dc[J][:,:,Jp,:], optimize = True)
    return out
 
    
def predict_radial_part(desc,dK,J,w):
    out = np.zeros((desc.natom,3))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('ln,l,nda->da',dK[i,:,Jp::desc.maxtype],w,desc.dc[J][:,0,i,do,:], optimize = True)
        out[desc.centraltype[J]] += np.einsum('iln,l,nia->ia',dK[:,:,Jp::desc.maxtype],w,desc.self_dc[J][:,0,Jp,:], optimize = True)    
    return out

def predict_dKxdp(settings,glob,desc,dK,J,w):
    size = dK.shape[:2]
    dK = dK.reshape(*size,settings.Nmax,settings.Nmax,settings.Lmax+1,desc.maxtype,desc.maxtype)#[:,:,:,:,lm_to_l,:,:]
    out = np.zeros((settings.Nmax,(settings.Lmax+1)**2,desc.maxtype,size[0]))
    for lm, dp in enumerate(np.rollaxis(desc.dp[J], 1)):
        out[:,lm,...] = np.einsum('ilnpJq,l,nJi->pqi',dK[...,glob.lm_to_l[lm],:,:],w,dp, optimize = True) + np.einsum('ilpnqJ,l,nJi->pqi',dK[...,glob.lm_to_l[lm],:,:],w,dp, optimize = True)
    # lm = m, np = p, Jp = q
    return out

def predict_angular_part(settings,glob,desc,dK,J,w):
    buf = predict_dKxdp(settings,glob,desc,dK,J,w)
    out = np.zeros((desc.natom,3))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('nm,nmda->da',buf[:,:,Jp,i],desc.dc[J][:,:,i,do,:], optimize = True)
        out[desc.centraltype[J]] += np.einsum('nmi,nmia->ia',buf[:,:,Jp,:],desc.self_dc[J][:,:,Jp,:], optimize = True)
    return out
    
def combine_derivertives(settings,glob,desc,lrc,nlrc,J): #(settings,glob,desc,lrc,zeta,nlrc,J)
    r'''
    Arguments
    ---------
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    lrc  : ndarray
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`
    zeta : scalar
        Power of the polinomial kernel :math:`\zeta`
    nlrc : int
        Number of local 
    J    : int
        Type index :math:`J` for specifing the atom type

    '''
    K, dK = get_dK(desc,lrc,settings,J)
    out = settings.Beta[0]*radial_part(desc,dK[:,:,:settings.Nmax*desc.maxtype],nlrc,J)
    out += settings.Beta[1]*angular_part(settings,glob,desc,dK[:,:,settings.Nmax*desc.maxtype:],nlrc,J)
    return np.sum(K,axis=0)/desc.natom, out

def predict_combine_derivertives(settings,glob,desc,lrc,J,w):
    r'''
    Arguments
    ---------
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    lrc  : ndarray
        The local refferenc configurations :math:`\hat{\textbf{X}}_b`
    zeta : scalar
        Power of the polinomial kernel :math:`\zeta`
    J    : int
        Type index :math:`J` for specifing the atom type

    '''
    K, dK = get_dK(desc,lrc,settings,J)
    out = settings.Beta[0]*predict_radial_part(desc,dK[:,:,:settings.Nmax*desc.maxtype],J,w)
    out += settings.Beta[1]*predict_angular_part(settings,glob,desc,dK[:,:,settings.Nmax*desc.maxtype:],J,w)
    return np.sum(np.matmul(K,w),axis=0)/desc.natom, out
