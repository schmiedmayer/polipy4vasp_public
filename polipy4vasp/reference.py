#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.linalg import eigh
from scipy.linalg.lapack import dpstrf
import numpy as np

from .kernel import K, K_s

def K_norm(x,y,settings):
    r'''
    This routine computes the kenel norm defined as:
        
    .. math:: d(x,y) = K(x,x)+k(y,y)-2*K(x,y)

    Arguments
    ---------
    x : array
        Local configuration :math:`x`
    y : array
        Local configuration :math:`y` 
    settings : Setup
        Class containing all the user defined settings for training the MLFF

    Returns
    -------
    norm : ndarray
        Kenel norm.

    '''

    Kxy = K(x,y,settings)
    if x.ndim == 2: Kxx = K_s(x,settings)
    else: Kxx = K(x,x,settings)
    if y.ndim == 2: Kyy = K_s(y,settings)
    else: Kyy = K(y,y,settings)
    if Kxy.ndim == 2: norm = Kxx[:,np.newaxis] + Kyy[np.newaxis] - 2*Kxy
    else: norm = Kxx + Kyy - 2*Kxy
    return norm.squeeze()

def random_lrc(settings, J,lc):
    r'''
    Random selection of local configurations.

    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    J : int
        Atomic species.
    lc : ndarray
        Local configurations.

    Returns
    -------
    args : ndarray
        Integer array containing the index of the selected lrc.

    '''
    Nselect = int(settings.NLRC[J])
    if lc.ndim == 3: lc = lc[0,:,:]
    size = lc.shape
    buf = np.arange(size[0],dtype=np.int64)
    np.random.shuffle(buf)
    args = buf[:Nselect]
    return args

def iter_CUR(settings,J,lc,data):
    r'''
    This routine performes the CUR algorithm [#]_  iteratively for selecting local reference 
    configurations.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    J : int
        Atomic species.
    lc : ndarray
        Local configurations.
    data : Training_Data
        Contains all vital information from the ML_AB file
        
    Returns
    -------
    args : ndarray
        Indices of selected local refference configurations
        
    Notes
    -----
    The CUR algorithm works by first calculating the eigenvectors and eigenvallues.
    Since the computational effort increases cubicaly with the matrix size, the algorithm
    is perfored on smaller subsets
    
    .. math:: \textbf{U}^T\textbf{K}\textbf{U}=\textbf{L}=\mathrm{Diag}(l_1,\dots,l_{N_b}).
    
    The matrix :math:`\textbf{K}` contains the kernels for all possible combinations of LRC. It
    is diagonalized and the matrix :math:`\textbf{U}` is the eigenvector matrix
    
    .. math:: \begin{split}\textbf{U} &= (\textbf{u}_1,\dots,\textbf{u}_{N_b}),\\\textbf{u}_{\mu} &=\begin{pmatrix}u_{1 \mu}\\ \vdots \\u_{N_b \mu}\end{pmatrix}\end{split}
    
    with :math:`u_{\nu\mu}` being the :math:`\nu`-th entry of the :math:`\mu`-th eigenvector
    :math:`\textbf{u}_{\mu}`.
    
    LRC that correlate as little as possible should be kept. This is achieved via applying a 
    modified version of the CUR algorithm that calculates the score :math:`\omega_\mu`
    
    .. math:: \omega_\mu=\frac{1}{N_\text{low}}\sum_{\nu=1}^{N_b}\gamma_{\nu \mu}
    
    with :math:`N_\text{low}` being the total number of eigenvectors :math:`\textbf{u}_\nu` whose 
    eigenvalue :math:`l_\nu` is smaller than the threshold :math:`\epsilon_\text{CUR}` and
    
    .. math:: \gamma_{\nu \mu}=\begin{cases} u^2_{\mu\nu}&\quad \text{if } l_\nu < \epsilon_\text{CUR}\\ 0 &\quad \text{else}\end{cases}.
    
    Discard local configurations with lowest leverage score!
    
    References
    ----------
    .. [#] https://doi.org/10.1073/pnas.0803205106
    '''
    def _CUR(mat,EpsCur,Nselect):
        L, U  = eigh(mat)
        arg   = L < EpsCur*np.max(L)
        omega = np.sum(U[:,~arg]**2,axis=1)
        narg  = np.argsort(omega)
        if len(narg) <= Nselect:
            return narg
        else :
            return narg[-Nselect:]
    
    Nselect = int(settings.NLRC[J])
    if lc.ndim == 3: lc = lc[0,:,:]
    ll  = np.array_split(lc,data.nconf)
    mat = K(ll[0],ll[0],settings)
    arg =_CUR(mat,settings.EpsCur,Nselect)
    s   = len(ll[0])
    args= np.copy(arg)
    lrc = ll[0][arg]
    for l in ll[1:]:
        args= np.hstack([args,s+np.arange(len(l),dtype=np.int64)])
        s  += len(l)
        l   = np.vstack([lrc,l])
        mat = K(l,l,settings)
        arg = _CUR(mat,settings.EpsCur,Nselect)
        lrc = l[arg]
        args=args[arg]
    return args

def FPS(settings,J,lc,data):
    r'''
    This routine performes the farthes point sampling algorithm for selecting local reference 
    configurations.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    J : int
        Atomic species.
    lc : ndarray
        Local configurations.
    data : Training_Data
        Contains all vital information from the ML_AB file
        
    Returns
    -------
    args : ndarray
        Indices of selected local refference configurations
    '''
    
    if lc.ndim == 3: lc = lc[0,:,:]
    Nselect = int(settings.NLRC[J])
    args    = [0]
    buf     = K_norm(lc[args[-1]],lc,settings)
    args.append(np.argmax(buf))
    for _ in range(Nselect-2):
        buf = np.min([buf,K_norm(lc[args[-1]],lc,settings)],axis=0)
        args.append(np.argmax(buf))
    args = np.array(args)
    return args

        

def Kmedian_init(settings,J,lc,data,Nit=25):
    r'''
    Routine to performe the K-mean algorithm by Stuart P. Lloyd [#]_. It is used
    to split the local configuration to reduce the computational load for the CUR
    algorithm.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    J : int
        Atomic species.
    lc : ndarray
        Local configurations.
    data : Training_Data
        Contains all vital information from the ML_AB file
    Nit : int (optional)
        Number of iterations (Default 25)
    
    Returns
    -------
    args : ndarray
        Indices of selected local refference configurations
    
    References
    ----------
    .. [#] https://doi.org/10.1109%2FTIT.1982.1056489
    '''
    
    if lc.ndim == 3: lc = lc[0,:,:]
    Nselect = int(settings.NLRC[J])
    out = FPS(settings,J,lc,data)
    m = lc[out]
    index = np.arange(len(lc))
    for _ in range(Nit):
        out = []
        arg = []
        for c in lc:
            buf = K_norm(c,m, settings)
            arg.append(np.argmin(buf))
        arg = np.array(arg)
        for i in range(Nselect):
            sarg = arg == i
            nMat = K_norm(lc[sarg],lc[sarg],settings)
            if nMat.ndim == 0 : n = 0
            else: n    = np.argmin(np.sum(nMat,axis=1))
            m[i] = lc[sarg][n]
            out.append(index[sarg][n])
    return np.array(out)


def iter_pivot(settings,J,lc,data):
    r'''
    This routine performes the pivoted Cholescy selection for selecting local reference 
    configurations.
    
    Arguments    print(np.all(lrc==lc[args]))
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    J : int
        Atomic species.
    lc : ndarray
        Local configurations.
    data : Training_Data
        Contains all vital information from the ML_AB file
        
    Returns
    -------
    args : ndarray
        Indices of selected local refference configurations
    '''
    
    if lc.ndim == 3: lc = lc[0,:,:]
    Nselect = int(settings.NLRC[J])
    ll  = np.array_split(lc,data.nconf)
    lrc = ll[0]
    s   = len(ll[0])
    args= np.arange(s,dtype=np.int64)
    for l in ll[1:]:
        args= np.hstack([args,s+np.arange(len(l),dtype=np.int64)])
        s+= len(l)
        l   = np.vstack([lrc,l])
        if len(l) <= Nselect:
            arg = np.arange(len(l),dtype=np.int64)
        else :
            mat = K(l,l,settings)
            _, jpvt, _,_= dpstrf(mat)
            arg = jpvt[:Nselect]-1
        lrc = l[arg]
        args= args[arg]
    return args
   
    
def get_LRC(settings,descriptors,data):
    r'''
    This routine selects the local refferenc configuration (refferenc descriptors).
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    descriptors : list
        List containing all the local configuration, its derivertive, and all necercery information for unraveling the data structure
    data : Training_Data
        Contains all vital information from the ML_AB file
        
    Returns
    -------
    lrc : list
        The local refferenc configurations.
    nlrc : list
        The number of local refferenc configurations.
    indx : list
        Index of local refferenc configurations.
        
    Notes
    -----
    There are multible methods for selecting local refferenc configurations:
    
    0.  Selects the local refference configurations stored in the ML_AB file
    
    1.  Random Selection.
    
    2.  CUR.
    
    3.  FPS
    
    4.  K-Median
    
    5.  Pivoted Choelesky.
        
    
    '''
    lrc = []
    nlrc= []
    indx= []
    for J in range(data.maxtype):
        if settings.lamb == None :
            lc = np.vstack([des.lc[J] for des in descriptors])
        else :
            if settings.Zeta > 1 : llc = np.vstack([des.llc[J] for des in descriptors])
            lc = np.concatenate([des.lc[J] for des in descriptors],1)
        
        if settings.AlgoLRC == 0:
            #selectes the local refference configurations stored in the ML_AB file
            args = data.lrc[J]
        if settings.AlgoLRC == 1:
            #selectes the local refference configurations randomly
            args = random_lrc(settings, J,lc)
        if settings.AlgoLRC == 2:
            #Uses the CUR algorythm
            args = iter_CUR(settings,J,lc,data)
        if settings.AlgoLRC == 3:
            #Uses the FPS algorythm
            args = FPS(settings,J,lc,data)
        if settings.AlgoLRC == 4:
            #Uses the K-median algorythm
            args = Kmedian_init(settings,J,lc,data)
        if settings.AlgoLRC == 5:
            #Uses the pivoted Choelesky algorythm
            args = iter_pivot(settings,J,lc,data)
        
        if settings.lamb == None : lrc.append(lc[args])
        elif settings.Zeta < 2: lrc.append(lc[:,args])
        else : lrc.append([lc[:,args],llc[args]])
        nlrc.append(len(args))
        indx.append(args)
    return lrc, nlrc, indx

