#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import polipy4vasp as pp

settings = pp.Setup(Rcut = 5.5,
                    Nmax = 8,
                    Wene = 50,
                    Wforc = 1,
                    Lmax = 4,
                    Beta = [.2,.8],
                    SigmaAtom = 0.4,
                    AlgoLRC = 1,
                    NLRC= [30],
                    Validation=0.2)
    
mlff = settings.train('Si_cd_100K-300K/ML_AB')
mlff.save('ML_FF')
