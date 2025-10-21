#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
import numpy as np
import pickle
from copy import deepcopy

from .POSCAR_reader import read_POSCAR
from .ML_AB_reader import read_ML_AB
from .tensor_classes import Tensor_Type, Type_list


@dataclass
class Ten_Training_Data:
    r"""
    Dataclass containing all vital information needed for training tensors.
    
    Args:
        nconf (int) : Number of configurations
        atomname (list) : List of strings containing the atom names
        maxtype (int) : Number of different atom species
        configurations (list) : List of :meth:`Configuration <polipy4vasp.ML_AB_reader.Configuration>`
        tensors (list) : List of training tensors
        Type (str) : Type of tensor
    """
    nconf : int
    maxtype : int
    atomname : list
    configurations: list
    tensors : list
    Type : Tensor_Type
    _ten : bool = True
    
    def save(self,filename="TEN_ML_AB"):
        r'''
        Wirtes the training data to a file.

        Parameters
        ----------
        filename : str, optional
            Filename where to write the TEN_ML_AB file. The default is 'TEN_ML_AB'.

        Returns
        -------
        None

        '''
        with open(filename,'wb') as file:
            pickle.dump(self, file)
            
    def reduce(self,confs,copy=False):
        r"""
        Reduces the training data set. Local refference configurations are discarded.

        Parameters
        ----------
        confs : list of ndarray of int
            Index of selected training configurations.
        copy : bool, optional
            Returnes a copy if ``True``. The default is False.

        Returns
        -------
        ML_AB : Training_Data, optional
            Returns reduced training data set if ``copy=True``

        """
        if copy :
            ML_AB = deepcopy(self)
        else :
            ML_AB = self
        ML_AB.nconf = len(confs)
        ML_AB.configurations = [ML_AB.configurations[i] for i in confs]
        ML_AB.tensors = [ML_AB.tensors[i] for i in confs]
        if copy :
            return ML_AB


def create_ten_TrainingsFile(Type='BEC',OUTCAR_path='OUT/',POSCAR_path='POS/',sep='.',start=1):
    r'''
    Wirtes the training data to a file.
    
    Parameters
    ----------
    Type : str, optional
        The default is 'BEC'.
    OUTCAR_path : str, optional
        The default is 'OUT/'.
    POSCAR_path : str, optional
        The default is 'POS/'.
    sep : str, optional
        The default is '.'.
    start : int, optional
        The default is 1.

    Returns
    -------
    Ten_ML_AB : Ten_Training_Data
        Tensor training data.

    Notes
    -----
    Types to select:

    BEC   Born effective Charges

    EFG   Electric field gradients
    
    '''
        
    Type_class = Type_list[Type]
        
    POS, TEN = [], []
    i = start
    maxtype = 0
    atomname = []
    found = False
    while os.path.isfile(OUTCAR_path+"OUTCAR"+sep+str(i)) and os.path.isfile(POSCAR_path+"POSCAR"+sep+str(i)):
        conf = read_POSCAR(POSCAR_path+"POSCAR"+sep+str(i))
        maxtype = max([maxtype,conf.maxtype])
        if len(conf.atomname) > len(atomname): atomname = conf.atomname
        POS.append(conf)
        TEN.append(Type_class.reader(OUTCAR_path+"OUTCAR"+sep+str(i),conf.natom))
        i += 1
        found = True
    if found :
        data = Ten_Training_Data(nconf = i-1,
                                 maxtype = maxtype,
                                 atomname = atomname,
                                 configurations = POS,
                                 tensors = TEN,
                                 Type = Type_class)
        return data
    else :
        print('No data found!')   
        
def read_TEN_ML_AB(filename="TEN_ML_AB"):
    with open(filename,'rb') as file:
        buf = file.read()
    data = pickle.loads(buf)
    return data
    
def read_data(filename):
    if type(filename) == str :
        try :
            data = read_TEN_ML_AB(filename)
        except pickle.UnpicklingError :
            data = read_ML_AB(filename)
    else : 
        data = deepcopy(filename)
    print('Read input file. Identified %d training configurations.' % data.nconf)
    return data
    
