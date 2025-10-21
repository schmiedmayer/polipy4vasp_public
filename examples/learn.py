#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import polipy4vasp as pp

settings = pp.Setup(Rcut2 = 9,
                    Rcut3 = 5.5,
                    Nmax2 = 12,
                    Nmax3 = 8,
                    Lmax = 4,
                    Beta = 0.8,
                    Waderiv=1,
                    SigmaAtom = 0.4,
                    AlgoLRC = 0,
                    Validation=0,
                    ncore=8,
                    Scatter_Plot=False)
    
mlff = settings.train('Si_cd_100K-300K/ML_AB')
mlff.save('ML_FF')
