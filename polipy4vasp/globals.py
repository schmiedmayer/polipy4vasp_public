#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
=====================================================================
Routines for calculatin global variables (:mod:`polipy4vasp.globals`)
=====================================================================

.. currentmodule:: polipy4vasp.globals
    
"""


import numpy as np
from dataclasses import dataclass
from vasa import asa

@dataclass
class Globals:
    r'''
    Dataclass containing all precomputet coefficients.
    
    args:
        lm_to_l (array) :
        fac (array) : Constant factor :math:`\sqrt{\frac{8\pi^2}{2l+1}}`.
        cleb (list) : The clebsch gordan coefficients. The default is None.
        c_index (list) : Indices of :math:`l(l+m)` and :math:`l'(l'+m')`. The default is None.
        len_cleb (list) : Number of clebsch gordan coefficients. The default is None.
        m_index_sum (list) : Indices for suming over all :math:`m` and :math:`m'` The default is None.
        ten_l_to_lm0 (list) : Indices for linking :math:`l` to the combined index :math:`lm`. The default is None.
        ten_l_to_lm1 (list) : Indices for linking :math:`l'` to the combined index :math:`l'm'`. The default is None.
        ten_lm_sum (list) : Used for summing :math:`lm` in the derivertive tensor contraction. The default is None.
        ten_lm_to_l (list) : Indices for linking :math:`lm` to the combined index :math:`l`. The default is None.
        len_m_index_sum (int) : Number of entries to sum in the derivertive tensor contraction. The default is None.
        max_len_cleb (int) : Number of nonzero clebsch gordan coefficients. The default is None.
        two_body_mask (array) : Mask for exdracting the two body The default is None.
    '''
    lm_to_l : np.array
    fac : np.array
    cleb : list = None
    c_index : list = None
    len_cleb : list = None
    m_index_sum : list = None
    ten_l_to_lm0 : list = None
    ten_l_to_lm1 : list = None
    ten_lm_sum : list = None
    ten_lm_to_l : list = None
    len_m_index_sum : int = None
    max_len_cleb : int = None
    two_body_mask : np.array = None
    
def setup_scalar_globals(Lmax):
    r'''
    Sets up the Constant factor :math:`\sqrt{\frac{8\pi^2}{2l+1}}` used for :math:`p^{iJJ'}_{nn'l}`.
    
    Arguments
    ---------
    Lmax : int
        Maximal angular quantum number :math:`l_\text{max}`
        
    Returns
    -------
    glob : Globals
        Class containing all precomputet coefficients
    
    
    .. warning::
        Needs to be run before using `get_c` and `get_p` to avoid an Error!
    '''
    lm_to_l = np.empty((Lmax+1)*(Lmax+1),dtype = np.int32)
    for l in range(Lmax+1):
        lm_to_l[l*l:(l+1)*(l+1)] = l
        
    fac = np.sqrt(8*np.pi**2/(2*np.arange(Lmax+1)+1))
    fac = fac[np.newaxis,:,np.newaxis,np.newaxis]
    
    return Globals(lm_to_l = lm_to_l,
                   fac     = fac)

def real_cleb(l1,l2,l3,m1,m2,m3):
    r'''
    Calculates the real clebsch gordan coefficient.
    
    ..math: C^{l_1m_1}_{l_2m_2,l_3m_3}

    Parameters
    ----------
    l1 : int
        :math:`l_1`
    l2 : int
        :math:`l_2`
    l3 : int
        :math:`l_3`
    m1 : int
        :math:`m_1`
    m2 : int
        :math:`m_2`
    m3 : int
        :math:`m_3`

    Returns
    -------
    c : float
        Real clebsch gordan coefficient.

    '''
    def fs(i):
        return 1-2*np.mod(i+20,2)   
    n1, n2, n3 = abs(m1), abs(m2), abs(m3)
    out, s1, s2, s3, t1, t2, t3 = 0, 0, 0, 0, 0, 0, 0
    if m1 <  0 : s1 = 1
    if m2 <  0 : s2 = 1
    if m3 <  0 : s3 = 1
    if m1 == 0 : t1 = 1
    if m2 == 0 : t2 = 1
    if m3 == 0 : t3 = 1
    if (n1+n2==-n3) : out += asa.clebg0(l1,l2,l3)
    if (n1+n2== n3) : out += asa.clebgo(l1,l2,l3, n1, n2, n3) * fs(n3+s3)
    if (n1-n2==-n3) : out += asa.clebgo(l1,l2,l3, n1,-n2,-n3) * fs(n2+s2)
    if (n1-n2== n3) : out += asa.clebgo(l1,l2,l3,-n1, n2,-n3) * fs(n1+s1)
    return out * fs(n3+(s1+s2+s3)/2) / (np.sqrt(2)**(1+t1+t2+t3))

def clebgo(glob,Lmax,lamb):
    r'''
    Calculates all precomputet coefficients for tensorial training.

    Parameters
    ----------
    glob : Globals
        Class containing all precomputet coefficients
    Lmax : int
        Maximal angular quantum number :math:`l_\text{max}`
    lamb : TYPE
        Order of the tensorial kernel :math:`\lambda`

    Returns
    -------
    None.
    
    .. warning::
        Needs to be run after `setup_scalar_globals`!
    '''
    cleb = []
    c_index, len_cleb = [], []
    m_index_sum = []
    ten_l_to_lm0, ten_l_to_lm1 = [], []
    ten_lm_sum = []
    ten_lm_to_l = []
    for kappa in range(-lamb,lamb+1):
        lay1 = []
        ind = []
        m_finder = 0
        lay2 = []
        ten_lay0, ten_lay1 = [], []
        c_lm_to_l = 0
        buf_lm_to_l = []
        for lp in range(Lmax+1):
            for l in range(abs(lamb - lp), lamb + lp + 1, 2):
                if l <= Lmax :
                    fac = np.sqrt(8*np.pi**2/(2*l+1))
                    m_sum = []
                    ten_lay01, ten_lay11 = [], []
                    for mp in range(-lp,lp+1):
                        if kappa*mp < 0 :
                            m = - abs(kappa) -abs(mp)
                            mpp = -abs(abs(kappa) - abs(mp))
                            if mpp == 0:
                                nm = 1
                            else:
                                nm = 2
                        elif kappa*mp == 0 :
                            m = kappa + mp
                            mpp = 0
                            nm = 1
                        else :
                            m = abs(kappa) + abs(mp)
                            mpp = abs(abs(kappa) - abs(mp))
                            nm = 2
                            
                        while nm > 0 :
                            if abs(m) <= l :

                                lay1.append(fac*real_cleb(lamb,lp,l,kappa,mp,m))
                                ind.append([(l*(l+1)+m),(lp*(lp+1)+mp)])
                                m_sum.append(m_finder)
                                m_finder += 1
                                ten_lay01.append((l*(l+1)+m))
                                ten_lay11.append((lp*(lp+1)+mp))

                            nm -= 1
                            m = mpp
                    lay2.append(m_sum)
                    buf_lm_to_l = [*buf_lm_to_l,*(c_lm_to_l*np.ones_like(m_sum,dtype=np.int32))]
                    c_lm_to_l  += 1
                    ten_lay0.append(ten_lay01)
                    ten_lay1.append(ten_lay11)
        m_index_sum.append(lay2)
        ten_l_to_lm0.append(ten_lay0)
        ten_l_to_lm1.append(ten_lay1)
        buf_lm_sum = []
        for lm in range((Lmax + 1) * (Lmax + 1)):
            buf_lm_sum.append((np.array(ind).T == lm)[np.newaxis])
        ten_lm_sum.append(buf_lm_sum)
        len_cleb.append(len(lay1))
        cleb.append(np.array(lay1)[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis])
        c_index.append(np.array(ind).T)
    buf = []
    for lp in range(Lmax+1):
        for l in range(abs(lamb - lp), lamb + lp + 1, 2):
            if l <= Lmax :
                buf.append([l,lp])
    buf = np.array(buf).T
    for l in range(Lmax+1):
        for m in range(-l,l+1):
            buffer = []
            amask = buf[0] == l
            bmask = buf[1] == l
            if np.any(amask):
                a = np.hstack(np.argwhere(amask))
                b = np.hstack(np.argwhere(bmask))
                for i, j in enumerate(a):
                    buffer.append([j,b[i]])
            ten_lm_to_l.append(buffer)
    for i, lm in enumerate(ten_lm_to_l) :
        len_l = len(lm)
        if len_l > 1 :
            for k, m in enumerate(m_index_sum) :
                out = []
                for l in lm :
                    buf = np.full(ten_lm_sum[k][i].shape,False)
                    for m0 in m[l[0]] :
                        buf[0,0,m0] = True
                    for m1 in m[l[1]] :
                        buf[0,1,m1] = True
                    out.append(np.logical_and(buf,ten_lm_sum[k][i]))
                ten_lm_sum[k][i] = np.vstack(out)
    glob.cleb = cleb
    glob.c_index = c_index
    glob.len_cleb = len_cleb
    glob.m_index_sum = m_index_sum
    glob.ten_l_to_lm0 = ten_l_to_lm0
    glob.ten_l_to_lm1 = ten_l_to_lm1
    glob.ten_lm_sum = ten_lm_sum
    glob.ten_lm_to_l = ten_lm_to_l
    glob.len_m_index_sum = len(m_index_sum[0])
    glob.max_len_cleb = max(len_cleb)
    glob.two_body_mask = glob.lm_to_l == lamb
    


def setup_globals(settings,data):
    r'''
    Sets up all precomputed factros.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
     data : Training_Data
        Contains all vital information from the ML_AB file
    
    Returns
    -------
    glob : Globals
        Class containing all precomputet coefficients
    
    
    .. warning::
        Needs to be run before using `get_c` and `get_p` to avoid an Error!
    '''
    glob = setup_scalar_globals(settings.Lmax)
    if data._ten : clebgo(glob,settings.Lmax,data.Type.lamb)
    return glob
