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
    return K

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

def get_dK(X_i,X_b,settings):
    r'''
    Routine for computing the polinomial kernel defined as [#]_
    
    .. math:: K(\textbf{X}_i,\textbf{X}_b) = \left(\hat{\textbf{X}}_i\cdot\hat{\textbf{X}}_b\right)^{\zeta},
    
    and its derivertive
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d} \textbf{X}_i}K(\textbf{X}_i,\textbf{X}_b)=\textbf{K}(\textbf{X}_i,\textbf{X}_b)\zeta\frac{1}{\left|\textbf{X}_i\right|}\left[\frac{1}{\hat{\textbf{X}}_i\cdot\hat{\textbf{X}}_b}\left(\frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}\textbf{X}_i\right) \cdot \hat{\textbf{X}}_b-\hat{\textbf{X}}_i\cdot\left(\frac{\mathrm{d}}{\mathrm{d} r_k^\alpha}\textbf{X}_i\right)\right].
    
    Arguments
    ---------
    X_i : array
        Local configuration :math:`\textbf{X}_i`
    X_b : array
        Local configuration :math:`\textbf{X}_b` 
    zeta : scalar
        Power of the polinomial kernel :math:`\zeta`
        
    Returns
    -------
    K  : ndarray
        The polinomial kernel :math:`K(\textbf{X}_i,\textbf{X}_b)`
    dK : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} \textbf{X}_i}K(\textbf{X}_i,\textbf{X}_b)`
    
    References
    ----------
    .. [#] https://doi.org/10.1063/5.0009491
    '''
    
    if settings.Kernel == "poli" :
        hatlc, abslc = norm_lc(X_i)
        hatlrc, _    = norm_lc(X_b)
        buf = np.dot(hatlc,hatlrc.T)
        K = buf**settings.Zeta
        dK = (K*(settings.Zeta/abslc)[:,np.newaxis])[:,:,np.newaxis]*((1/buf)[:,:,np.newaxis]*hatlrc[np.newaxis,:,:]-hatlc[:,np.newaxis,:])

    if settings.Kernel == "gaus" :
        buf = (X_i[:,np.newaxis]-X_b[np.newaxis])
        K = np.exp(-np.linalg.norm(buf,axis=-1)**2/settings.SigmaExp)
        dK = (-2*K/settings.SigmaExp)[:,:,np.newaxis]*buf

    if settings.Kernel == "linear" :
        K = np.dot(X_i,X_b.T)
        dK = np.empty([*K.shape,len(X_b[0])])
        dK[:,:,:] = X_b[np.newaxis,:,:]

    if settings.Kernel == "ngaus" :
        hatlc, abslc = norm_lc(X_i)
        hatlrc, _    = norm_lc(X_b)
        buf = (hatlc[:,np.newaxis]-hatlrc[np.newaxis])
        K = np.exp(-np.linalg.norm(buf,axis=-1)**2/settings.SigmaExp)
        dK = (-2*K/(settings.SigmaExp * abslc)[:,np.newaxis])[:,:,np.newaxis] * ( buf - (buf @ hatlc.T) @ hatlc) #stimmt nicht
    return K, dK

def radial_part(desc,dK,nlrc,J):
    r'''
    This routine computes the derivative of the kernel :math:`K` with respect to the atom possitions for
    the two-body part of the descriptor via the equation:
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d} r^\alpha_i} K^{(2)}(\textbf{X}_i,\textbf{X}_b)= \frac{\mathrm{d} K^{(2)}(\textbf{X}_i,\textbf{X}_b)}{\mathrm{d} \textbf{X}_i} \frac{\mathrm{d} \textbf{X}_i}{\mathrm{d} c^{jJ}_{n00}} \frac{\mathrm{d} c^{jJ}_{n00}}{\mathrm{d} r^\alpha_i}.

    Parameters
    ----------
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    nlrc : int
        Number of local refererenc configurations
    J    : int
        Type index :math:`J` for specifing the atom type

    Returns
    -------
    out : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} r^\alpha_i}K^{(2)}(\textbf{X}_i,\textbf{X}_b)`
    '''
    
    out = np.zeros((desc.natom,3,nlrc))
    dl = desc.derivl2[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype2[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('ln,nda->dal',dK[i,:,Jp::desc.maxtype],desc.dc2[J][:,i,do,:], optimize = True)
            #out[dl[i,do]] += np.tensordot(np.swapaxes(dK[i,:,Jp::desc.maxtype],0,1)[:,np.newaxis,np.newaxis,:],desc.dc2[J][:,i,do,:,np.newaxis],axes=([0],[0]))
        out[desc.centraltype[J]] += np.einsum('iln,nia->ial',dK[:,:,Jp::desc.maxtype],desc.self_dc2[J][:,Jp,:], optimize = True)    
    return out

def dKxdp(settings,glob,desc,dK,J):
    r'''
    Computes the derivative of kernel with respect to :math:`c^{iJ}_{nlm}` via 
    
    .. math:: \frac{\mathrm{d}}{\mathrm{d} c^{iJ}_{nlm}} K^{(3)}(\textbf{X}_i,\textbf{X}_b) = \frac{\mathrm{d} K^{(3)}(\textbf{X}_i,\textbf{X}_b)}{\mathrm{d} \textbf{X}_i} \frac{\mathrm{d} \textbf{X}_i}{\mathrm{d} p_{nn'll'}^{\mu iJJ'}} \frac{\mathrm{d} p_{nn'll'}^{\mu iJJ'}}{\mathrm{d} c^{iJ}_{nlm}}.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    J    : int
        Type index :math:`J` for specifing the atom type

    Returns
    -------
    out : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} c^{iJ}_{nlm}}K^{(3)}(\textbf{X}_i,\textbf{X}_b)`
    '''
    
    size = dK.shape[:2]
    dK = dK.reshape(*size,settings.Nmax3,settings.Nmax3,settings.Lmax+1,desc.maxtype,desc.maxtype)#[:,:,:,:,lm_to_l,:,:]
    out = np.zeros((settings.Nmax3,(settings.Lmax+1)**2,desc.maxtype,*size))
    for lm, dp in enumerate(np.rollaxis(desc.dp[J], 1)):
        out[:,lm,...] = np.einsum('ilnpJq,nJi->pqil',dK[...,glob.lm_to_l[lm],:,:],dp, optimize = True) + np.einsum('ilpnqJ,nJi->pqil',dK[...,glob.lm_to_l[lm],:,:],dp, optimize = True)
    # lm = m, np = p, Jp = q
    return out

def angular_part(settings,glob,desc,dK,nlrc,J):
    r'''
    This routine computes the derivative of the kernel :math:`K` with respect to the atom possitions for
    the three-body part of the descriptor via the equation:
        
    .. math:: \frac{\mathrm{d}}{\mathrm{d} c^{iJ}_{nlm}} K^{(3)}(\textbf{X}_i,\textbf{X}_b) = \frac{\mathrm{d} K^{(3)}(\textbf{X}_i,\textbf{X}_b)}{\mathrm{d} \textbf{X}_i} \frac{\mathrm{d} \textbf{X}_i}{\mathrm{d} p_{nn'll'}^{\mu iJJ'}} \frac{\mathrm{d} p_{nn'll'}^{\mu iJJ'}}{\mathrm{d} c^{iJ}_{nlm}} \frac{\mathrm{d} c^{iJ}_{nlm}}{\mathrm{d} r^\alpha_i}.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    nlrc : int
        Number of local refererenc configurations
    J    : int
        Type index :math:`J` for specifing the atom type

    Returns
    -------
    out : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} r^\alpha_i}K^{(3)}(\textbf{X}_i,\textbf{X}_b)`
    '''
    
    buf = dKxdp(settings,glob,desc,dK,J)
    out = np.zeros((desc.natom,3,nlrc))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('nml,nmda->dal',buf[:,:,Jp,i,:],desc.dc[J][:,:,i,do,:], optimize = True)
        out[desc.centraltype[J]] += np.einsum('nmil,nmia->ial',buf[:,:,Jp,:,:],desc.self_dc[J][:,:,Jp,:], optimize = True)
    return out
    
def combine_derivertives(settings,glob,desc,lrc,nlrc,J):
    r'''
    This routine computes the derivative of the kernel :math:`K` with respect to the atom possitions.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    nlrc : int
        Number of local refererenc configurations
    J    : int
        Type index :math:`J` for specifing the atom type

    Returns
    -------
    out : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} r^\alpha_i}K(\textbf{X}_i,\textbf{X}_b)`

    '''
    K, dK = get_dK(desc.lc[J],lrc,settings)
    out = np.sqrt(settings.Beta)*radial_part(desc,dK[:,:,:settings.Nmax2*desc.maxtype],nlrc,J)
    out += np.sqrt(1-settings.Beta)*angular_part(settings,glob,desc,dK[:,:,settings.Nmax2*desc.maxtype:],nlrc,J)
    return np.sum(K,axis=0), out

def predict_radial_part(desc,dK,J,w):
    r'''
    This routine computes the derivative of the kernel :math:`K` with respect to the atom possitions for
    the two-body part of the descriptor when predicting e.g. contracting the weights
    :math:`\omega` as soon as possible:
    
    .. math:: \sum_b \omega_b \frac{\mathrm{d}}{\mathrm{d} r^\alpha_i} K^{(2)}(\textbf{X}_i,\textbf{X}_b) =\sum_b \omega_b \frac{\mathrm{d} K^{(2)}(\textbf{X}_i,\textbf{X}_b)}{\mathrm{d} \textbf{X}_i} \frac{\mathrm{d} \textbf{X}_i}{\mathrm{d} c^{jJ}_{n00}} \frac{\mathrm{d} c^{jJ}_{n00}}{\mathrm{d} r^\alpha_i}.

    Parameters
    ----------
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    nlrc : int
        Number of local refererenc configurations
    J    : int
        Type index :math:`J` for specifing the atom type
    w    : ndarray
        The weights math:`\omega`.

    Returns
    -------
    out : ndarray
        The derivertive of the kernel allready contracted over :math:`\omega`
    '''
    out = np.zeros((desc.natom,3))
    dl = desc.derivl2[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype2[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('ln,l,nda->da',dK[i,:,Jp::desc.maxtype],w,desc.dc2[J][:,i,do,:], optimize = True)
        out[desc.centraltype[J]] += np.einsum('iln,l,nia->ia',dK[:,:,Jp::desc.maxtype],w,desc.self_dc2[J][:,Jp,:], optimize = True)    
    return out

def predict_dKxdp(settings,glob,desc,dK,J,w):
    r'''
    Computes the derivative of kernel with respect to :math:`c^{iJ}_{nlm}` when predicting e.g. contracting the weights
    :math:`\omega` as soon as possible via 
    
    .. math:: \sum_b \omega_b \frac{\mathrm{d}}{\mathrm{d} c^{iJ}_{nlm}} K^{(3)}(\textbf{X}_i,\textbf{X}_b) = \sum_b \omega_b \frac{\mathrm{d} K^{(3)}(\textbf{X}_i,\textbf{X}_b)}{\mathrm{d} \textbf{X}_i} \frac{\mathrm{d} \textbf{X}_i}{\mathrm{d} p_{nn'll'}^{\mu iJJ'}} \frac{\mathrm{d} p_{nn'll'}^{\mu iJJ'}}{\mathrm{d} c^{iJ}_{nlm}}.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    J    : int
        Type index :math:`J` for specifing the atom type
    w    : ndarray
        The weights math:`\omega`.

    Returns
    -------
    out : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} c^{iJ}_{nlm}}K^{(3)}(\textbf{X}_i,\textbf{X}_b)`
    '''
    
    size = dK.shape[:2]
    dK = dK.reshape(*size,settings.Nmax3,settings.Nmax3,settings.Lmax+1,desc.maxtype,desc.maxtype)#[:,:,:,:,lm_to_l,:,:]
    out = np.zeros((settings.Nmax3,(settings.Lmax+1)**2,desc.maxtype,size[0]))
    for lm, dp in enumerate(np.rollaxis(desc.dp[J], 1)):
        out[:,lm,...] = np.einsum('ilnpJq,l,nJi->pqi',dK[...,glob.lm_to_l[lm],:,:],w,dp, optimize = True) + np.einsum('ilpnqJ,l,nJi->pqi',dK[...,glob.lm_to_l[lm],:,:],w,dp, optimize = True)
    # lm = m, np = p, Jp = q
    return out

def predict_angular_part(settings,glob,desc,dK,J,w):
    r'''
    This routine computes the derivative of the kernel :math:`K` with respect to the atom possitions for
    the three-body part of the descriptor when predicting e.g. contracting the weights
    :math:`\omega` as soon as possible via the equation:
        
    .. math:: \sum_b \omega_b \frac{\mathrm{d}}{\mathrm{d} c^{iJ}_{nlm}} K^{(3)}(\textbf{X}_i,\textbf{X}_b) = \sum_b \omega_b \frac{\mathrm{d} K^{(3)}(\textbf{X}_i,\textbf{X}_b)}{\mathrm{d} \textbf{X}_i} \frac{\mathrm{d} \textbf{X}_i}{\mathrm{d} p_{nn'll'}^{\mu iJJ'}} \frac{\mathrm{d} p_{nn'll'}^{\mu iJJ'}}{\mathrm{d} c^{iJ}_{nlm}} \frac{\mathrm{d} c^{iJ}_{nlm}}{\mathrm{d} r^\alpha_i}.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    nlrc : int
        Number of local refererenc configurations
    J    : int
        Type index :math:`J` for specifing the atom type

    Returns
    -------
    out : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} r^\alpha_i}K^{(3)}(\textbf{X}_i,\textbf{X}_b)`
    '''
    
    buf = predict_dKxdp(settings,glob,desc,dK,J,w)
    out = np.zeros((desc.natom,3))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('nm,nmda->da',buf[:,:,Jp,i],desc.dc[J][:,:,i,do,:], optimize = True)
        out[desc.centraltype[J]] += np.einsum('nmi,nmia->ia',buf[:,:,Jp,:],desc.self_dc[J][:,:,Jp,:], optimize = True)
    return out

def predict_combine_derivertives(settings,glob,desc,lrc,J,w):
    r'''
    This routine is used to compute the derivative of the kernel :math:`K` with respect to the atom possitions
    when predicting. It contrects the weights :math:`\omega` at the erliest point for most computational
    efficency.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    glob : Globals
        Class containing all precomputet coefficients
    desc : Descriptor
        Class containing the local configuration, its derivertive, and all necercery information for unraveling the data structure
    dK : ndarray
        The derivertive of the kernel only containing the radial descriptor part i.e. the :math:`N_\text{max}J_\text{max}` elements.
    nlrc : int
        Number of local refererenc configurations.
    J    : int
        Type index :math:`J` for specifing the atom type.
    w    : ndarray
        The weights math:`\omega`.

    Returns
    -------
    out : ndarray
        The derivertive of the kernel :math:`\frac{\mathrm{d}}{\mathrm{d} r^\alpha_i}K(\textbf{X}_i,\textbf{X}_b)`

    '''
    K, dK = get_dK(desc.lc[J],lrc,settings)
    out = np.sqrt(settings.Beta)*predict_radial_part(desc,dK[:,:,:settings.Nmax2*desc.maxtype],J,w)
    out += np.sqrt(1-settings.Beta)*predict_angular_part(settings,glob,desc,dK[:,:,settings.Nmax2*desc.maxtype:],J,w)
    return np.sum(np.matmul(K,w),axis=0), out