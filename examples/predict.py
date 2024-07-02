#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import polipy4vasp as pp

mlff = pp.load_mlff('ML_FF')

E = []
F = []
for i in range(1,51):
    conf = pp.read_POSCAR('POS/POSCAR'+str(i))
    out = mlff.predict(conf)
    E.append(out[0])
    F.append(out[1])
