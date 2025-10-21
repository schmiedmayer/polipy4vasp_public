#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import polipy4vasp as pp

settings = pp.Setup(Rcut = 6.,
                    Nmax = 8,
                    Lmax = 4,
                    SigmaAtom = 0.4,
                    AlgoLRC = 3,
                    Zeta = 0,
                    Kernel='linear',
                    NLRC = [30,60],
                    TenArgs=([4,-2],))

mlff = settings.train('ZrO2_Ten/TEN_ML_AB')
mlff.save()
