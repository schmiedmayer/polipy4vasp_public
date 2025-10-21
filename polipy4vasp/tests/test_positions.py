#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pytest import approx
import numpy as np
from copy import deepcopy

from polipy4vasp.positions import get_nn_confs, get_nn_conf, check_mult_per_img, conf2nn_conf, conf2nn_conf_fast

from test_utils import generate_Settings, generate_Symmetric_Settings, generate_Globals, generate_Configurations, generate_Splines, make_num_diff, lm_to_l_and_m

Settings = generate_Settings()
Configurations = generate_Configurations()

def test_reduce_nnconf():
    def compare_nnconf(a,b):
        if a.mult_per_img:
            sort_l = np.array(a.natom*[np.arange(a.maxnn)])
        else:
            buf = np.copy(a.nnl)
            mask = buf < 0
            buf[mask] = 10000
            sort_l = np.argsort(buf,axis=-1)
        assert np.array_equal(a.nnn, b.nnn, equal_nan=True), 'reduce: "nnn" in nearest neighbor configuration not consistent!'
        assert a.maxnn == b.maxnn, 'reduce: "maxnn" in nearest neighbor configuration not consistent!'
        assert a.natom == b.natom, 'reduce: "natom" in nearest neighbor configuration not consistent!'
        assert a.maxtype == b.maxtype, 'reduce: "maxtype" in nearest neighbor configuration not consistent!'
        assert np.array_equal(a.centraltype, b.centraltype, equal_nan=True), 'reduce: "centraltype" in nearest neighbor configuration not consistent!'
        for i, sort in enumerate(sort_l):
            assert np.array_equal(a.nnl[i][sort], b.nnl[i], equal_nan=True), 'reduce: "nnl" in nearest neighbor configuration not consistent!'
            assert np.allclose(a.nnr[i][sort], b.nnr[i], equal_nan=True), 'reduce: "nnr" in nearest neighbor configuration not consistent!'
            assert np.allclose(a.nnvecr[i][sort], b.nnvecr[i], equal_nan=True), 'reduce: "nnvecr" in nearest neighbor configuration not consistent!'
            assert np.array_equal(a.nntype[i][sort], b.nntype[i], equal_nan=True), 'reduce: "nntype" in nearest neighbor configuration not consistent!'
        assert a.mult_per_img == b.mult_per_img, 'reduce: "mult_per_img" in nearest neighbor configuration not consistent!'
        assert a.volume == b.volume, 'reduce: "volume" in nearest neighbor configuration not consistent!'
        
    for sett in Settings:
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            ref_nnconf2 = get_nn_conf(conf,sett.Rcut2)
            ref_nnconf3 = get_nn_conf(conf,sett.Rcut3)
            compare_nnconf(nnconf2,ref_nnconf2)
            compare_nnconf(nnconf3,ref_nnconf3)
            
def test_nnconf_fast():
    def compare_nnconf(a,b):
        if a.mult_per_img:
            sort_l = np.array(a.natom*[np.arange(a.maxnn)])
        else:
            buf = np.copy(a.nnl)
            mask = buf < 0
            buf[mask] = 10000
            sort_l = np.argsort(buf,axis=-1)
        assert np.array_equal(a.nnn, b.nnn, equal_nan=True), 'fast: "nnn" in nearest neighbor configuration not consistent!'
        assert a.maxnn == b.maxnn, 'fast: "maxnn" in nearest neighbor configuration not consistent!'
        assert a.natom == b.natom, 'fast: "natom" in nearest neighbor configuration not consistent!'
        assert a.maxtype == b.maxtype, 'fast: "maxtype" in nearest neighbor configuration not consistent!'
        assert np.array_equal(a.centraltype, b.centraltype, equal_nan=True), 'fast: "centraltype" in nearest neighbor configuration not consistent!'
        for i, sort in enumerate(sort_l):
            assert np.array_equal(a.nnl[i][sort], b.nnl[i], equal_nan=True), 'fast: "nnl" in nearest neighbor configuration not consistent!'
            assert np.allclose(a.nnr[i][sort], b.nnr[i], equal_nan=True), 'fast: "nnr" in nearest neighbor configuration not consistent!'
            assert np.allclose(a.nnvecr[i][sort], b.nnvecr[i], equal_nan=True), 'fast: "nnvecr" in nearest neighbor configuration not consistent!'
            assert np.array_equal(a.nntype[i][sort], b.nntype[i], equal_nan=True), 'fast: "nntype" in nearest neighbor configuration not consistent!'
        assert a.mult_per_img == b.mult_per_img, 'fast: "mult_per_img" in nearest neighbor configuration not consistent!'
        assert a.volume == b.volume, 'fast: "volume" in nearest neighbor configuration not consistent!'
        
    for sett in Settings:
        for conf in Configurations:
            if not check_mult_per_img(conf.lattice,sett.Rcut2):
                ref_nnconf = conf2nn_conf(conf,sett.Rcut2)
                nnconf = conf2nn_conf_fast(conf,sett.Rcut2)
                compare_nnconf(ref_nnconf,nnconf)
            if not check_mult_per_img(conf.lattice,sett.Rcut2):
                ref_nnconf = conf2nn_conf(conf,sett.Rcut2)
                nnconf = conf2nn_conf_fast(conf,sett.Rcut2)
                compare_nnconf(ref_nnconf,nnconf)
    
            
if __name__ == '__main__':
    test_reduce_nnconf()
    test_nnconf_fast()