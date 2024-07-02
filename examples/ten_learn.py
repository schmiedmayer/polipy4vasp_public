#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import polipy4vasp as pp

settings = pp.Setup(Rcut = 5.5,
                    Nmax = 6,
                    Lmax = 3,
                    lamb = 1,
                    SigmaAtom = 0.5,
                    AlgoLRC = 1,
                    Zeta= 0,
                    NLRC = [20,40],
                    Charges=[4,-2],
                    deriv = True,
                    ncore=8,
                    Beta=[0.6,0.4])

mlff = settings.train('ZrO2_Ten/TEN_ML_AB')
mlff.save('TEN_ML_FF')
