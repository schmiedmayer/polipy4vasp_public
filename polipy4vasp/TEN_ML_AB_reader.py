#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
import numpy as np
import pickle

from .POSCAR_reader import read_POSCAR
from .ML_AB_reader import read_ML_AB


@dataclass
class Ten_Training_Data:
    
    nconf : int
    maxtype : int
    atomtypes : list
    configurations: list
    tensors : list

def create_ten_TrainingsFile(OUTCAR_path="OUT/",POSCAR_path="POS/",sep=".",filename="TEN_ML_AB",start=1):
    POS, TEN = [], []
    i = start
    maxtype = 0
    atomtypes = []
    while os.path.isfile(OUTCAR_path+"OUTCAR"+sep+str(i)) and os.path.isfile(POSCAR_path+"POSCAR"+sep+str(i)):
        conf = read_POSCAR(POSCAR_path+"POSCAR"+sep+str(i))
        maxtype = max([maxtype,conf.maxtype])
        if len(conf.atomname) > len(atomtypes): atomtypes = conf.atomname
        POS.append(conf)
        lines = []
        born_chagre = False
        c = 0
        stop = conf.natom*4+2
        with open(OUTCAR_path+"OUTCAR"+sep+str(i)) as f :
            t = 0
            for l in f :
                t += 1
                if " BORN EFFECTIVE CHARGES (including local field effects)" in l :
                    born_chagre = True
                if born_chagre and c < stop :
                    c += 1
                    lines.append(l)
        lines = lines[2:]
        out = []
        for n in range(conf.natom) :
            out.append([lines[n*4+1].split()[1:4],
                        lines[n*4+2].split()[1:4],
                        lines[n*4+3].split()[1:4]])
        TEN.append(np.array(out,dtype=np.float64))
        i += 1
    if i > 1 :
        data = Ten_Training_Data(nconf = i-1,
                                 maxtype = maxtype,
                                 atomtypes = atomtypes,
                                 configurations = POS,
                                 tensors = TEN)
        with open(filename,'wb') as file:
            pickle.dump(data, file)
    else :
        print('No data found!')   
        
def read_TEN_ML_AB(filename="TEN_ML_AB"):
    with open(filename,'rb') as file:
        buf = file.read()
    data = pickle.loads(buf)
    return data
    
def read_data(settings,filename):
    try :
        data = read_TEN_ML_AB(filename)
        if settings.lamb == None: raise AssertionError('ERROR: You need to set "lamb" in settings!')
    except pickle.UnpicklingError :
        data = read_ML_AB(filename)
        if settings.lamb != None and settings.lamb != 0: raise AssertionError('ERROR: "lamb" need to be None or 0!')
    return data
