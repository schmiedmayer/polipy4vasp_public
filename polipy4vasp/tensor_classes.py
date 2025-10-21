#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import numpy as np
from typing import Callable


def _BEC_calc_ionic_polarisation(conf,intr_charge):
    intr_charge = np.array(intr_charge)
    return np.sum(intr_charge[conf.atomtype][:,np.newaxis]*conf.atompos,axis=0)

def _BEC_calc_ionic_borncharges(conf,intr_charge):
    Z = np.empty((conf.natom,3,3))
    for J in range(conf.maxtype):
        Z[conf.atomtype == J] = np.diag([intr_charge[J]]*3)[None]
    return Z

def _polarisation_to_minimgcon(P,conf):
    P = np.linalg.inv(conf.lattice).T @ P
    tolarge = P > 0.5
    tosmall = P <= -0.5
    while ~np.all(~tolarge) or ~np.all(~tosmall):
        P[tolarge] -= 1
        P[tosmall] += 1
        tolarge = P > 0.5
        tosmall = P <= -0.5
    return conf.lattice @ P

def BEC_preprocess(settings,data):
    if type(settings.TenArgs) == tuple:
        Charges = np.array(settings.TenArgs[0])
    else:
        types = np.array(data.atomname)
        Charges = np.zeros(len(data.atomname))
        n_mean = np.zeros(len(data.atomname))
        
        for conf, ten in zip(data.configurations,data.tensors):
            for J, aJ in enumerate(conf.atomname):
                JJ = np.argwhere(aJ == types)[0,0]
                buf_Charges = np.diagonal(ten[conf.atomtype == J],axis1=-2,axis2=-1)
                Charges[JJ] += buf_Charges.sum()
                n_mean[JJ] += buf_Charges.size
        
        Charges /= n_mean
    
    data.tensors = [np.roll(ten-np.mean(ten,axis=0)-_BEC_calc_ionic_borncharges(conf,Charges),shift=-1,axis=2) for ten, conf in zip(data.tensors, data.configurations)]
    return (Charges,), 1.


def BEC_postprocess(ten,conf,args):
    #return _polarisation_to_minimgcon(np.roll(ten,1) + _BEC_calc_ionic_polarisation(conf,args[0]),conf)
    return np.roll(ten,1) + _BEC_calc_ionic_polarisation(conf,args[0])

def BEC_d_postprocess(ten,conf,args):
    return np.roll(ten,shift=1,axis=2) + _BEC_calc_ionic_borncharges(conf,args[0])

def BEC_reader(path,natom):
    lines = []
    lstart = False
    c = 0
    stop = natom*4+2
    with open(path) as f :
        for l in f :
            if " BORN EFFECTIVE CHARGES (including local field effects)" in l :
                lstart = True
            if lstart and c < stop :
                c += 1
                lines.append(l)
    lines = lines[2:]
    out = []
    for n in range(natom) :
        out.append([lines[n*4+1].split()[1:4],
                    lines[n*4+2].split()[1:4],
                    lines[n*4+3].split()[1:4]])
    return np.array(out,dtype=np.float64)


def EFG_reader(path,natom):
    lines = []
    lstart = False
    c = 0
    stop = natom+4
    with open(path) as f :
        for l in f :
            if " Electric field gradients (V/A^2)" in l :
                lstart = True
            if lstart and c < stop :
                c += 1
                lines.append(l)
    lines = lines[4:]
    out = [l.split()[1:] for l in lines]
    return np.array(out,dtype=np.float64)

def EFG_preprocess(settings,data):
    #          m  -2,-1, 0           , 1, 2
    T = np.array([[0, 0,-1/np.sqrt(3), 0, 1], #xx
                  [0, 0,-1/np.sqrt(3), 0,-1], #yy
                  [0, 0, 2/np.sqrt(3), 0, 0], #zz
                  [2, 0, 0,            0, 0], #xy
                  [0, 0, 0,            2, 0], #xz
                  [0, 2, 0,            0, 0]],#yz
                 dtype=np.float64)

    trace = [(ten[:,:3].sum(-1)/3) for ten in data.tensors]
    data.tensors = [(ten - np.hstack([tr[:,None]*np.ones((len(ten),3)),np.zeros((len(ten),3))])) @ T for ten, tr in zip(data.tensors,trace)]
    return (None,), 1.

def EFG_postprocess(ten,conf,args):
    #                  xx          , yy          , zz          , xy , xz , yz        m
    invT = np.array([[ 0           , 0           , 0           , 0.5, 0  , 0  ],  # -2
                     [ 0           , 0           , 0           , 0  , 0  , 0.5],  # -1
                     [-np.sqrt(3)/6,-np.sqrt(3)/6, np.sqrt(3)/3, 0  , 0  , 0  ],  #  0
                     [ 0           , 0           , 0           , 0  , 0.5, 0  ],  #  1
                     [ 0.5         ,-0.5         , 0           , 0  , 0  , 0  ]], #  2
                    dtype=np.float64) 
    return ten.T @ invT

@dataclass
class Tensor_Type:
    
    Type : str = 'Born effective Charges'
    unit : str = '|e|'
    lamb : int = 1
    deriv: bool= True
    summ : bool= True
    reader       : Callable[[str,int],np.ndarray]=BEC_reader
    preprocess   : Callable[[type,type],tuple]=BEC_preprocess
    postprocess  : Callable[[np.ndarray,type,tuple],np.ndarray]=BEC_postprocess
    d_postprocess: Callable[[np.ndarray,type,tuple],np.ndarray]=BEC_d_postprocess

Type_list = { 'BEC' : Tensor_Type(Type='Born effective Charges',
                                  unit = '|e|',
                                  lamb = 1,
                                  deriv= True,
                                  summ = True,
                                  reader       = BEC_reader,
                                  preprocess   = BEC_preprocess,
                                  postprocess  = BEC_postprocess,
                                  d_postprocess= BEC_d_postprocess),
              'EFG' : Tensor_Type(Type='Electric field gradients',
                                  unit = 'V/Å²',
                                  lamb = 2,
                                  deriv= False,
                                  summ = False,
                                  reader       = EFG_reader,
                                  preprocess   = EFG_preprocess,
                                  postprocess  = EFG_postprocess,
                                  d_postprocess= EFG_postprocess)
             }
