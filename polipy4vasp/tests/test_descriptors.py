#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pytest import approx
import numpy as np
from copy import deepcopy

from polipy4vasp.descriptors import get_cTilde_grad, get_cTilde, get_cTilde_grad_2b, get_cTilde_2b, get_cTilde_ten_grad_2b, get_cTilde_ten_2b
from polipy4vasp.descriptors import get_c_grad, get_c, get_c_grad_2b, get_c_2b, get_c_ten_grad_2b, get_c_ten_2b
from polipy4vasp.descriptors import get_p_grad, get_p, get_p_ten_grad, get_p_ten
from polipy4vasp.descriptors import get_Descriptor, get_TenDescriptor_grad, get_TenDescriptor

from polipy4vasp.positions import get_nn_confs
from polipy4vasp.sph import setup_asa

from test_utils import dx, generate_Settings, generate_Symmetric_Settings, generate_Globals, generate_Configurations, generate_Splines, make_num_diff, lm_to_l_and_m

setup_asa(6)
Settings = generate_Settings()
Globals = generate_Globals(Settings)
Splines = generate_Splines(Settings)
SymSettings = generate_Symmetric_Settings()
SymGlobals = generate_Globals(SymSettings)
SymSplines = generate_Splines(SymSettings)
Configurations = generate_Configurations()
lamb = [0,1,2,3] # not working for lamb > 3 
TenGlobals = [generate_Globals(Settings,lamb=l,ten=True) for l in lamb]
TenSplines = [generate_Splines(Settings,lamb=l) for l in lamb]
TenSymGlobals = [generate_Globals(SymSettings,lamb=l,ten=True) for l in lamb]
TenSymSplines = [generate_Splines(SymSettings,lamb=l) for l in lamb]



def test_cTilde():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            cTilde, dcTilde = get_cTilde_grad(sett, nnconf3, h[1])
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            ref_cTilde = get_cTilde(sett, nnconf3, h[1])
            assert np.allclose(np.nan_to_num(cTilde), np.nan_to_num(ref_cTilde), equal_nan=True), '"get_cTilde_grad" and "get_cTilde" do not produce same data for cTilde!'
            
            pconf = deepcopy(conf)
            mconf = deepcopy(conf)
            indx, ppos, mpos = make_num_diff(conf.atompos,dx)
            pconf.atompos = ppos
            mconf.atompos = mpos
            pnnconf2, pnnconf3 = get_nn_confs(sett, pconf)
            mnnconf2, mnnconf3 = get_nn_confs(sett, mconf)
            
            pcTilde = get_cTilde(sett, pnnconf3, h[1])
            mcTilde = get_cTilde(sett, mnnconf3, h[1])

            num_dcTilde = (pcTilde - mcTilde) / (dx * 2)
            if nnconf3.mult_per_img:
                
                nn_num_dcTilde = np.zeros((sett.Nmax3,(sett.Lmax + 1)**2,pnnconf3.natom))
                nn_dcTilde = np.zeros((sett.Nmax3,(sett.Lmax + 1)**2,nnconf3.natom))
                for i in range(pnnconf3.natom):
                    nn_num_dcTilde[:,:,i] = num_dcTilde[:,:,i,pnnconf3.nnl[i] == indx[0]].sum(-1)
                    nn_dcTilde[:,:,i] = dcTilde[:,:,i,nnconf3.nnl[i] == indx[0],indx[1]].sum(-1)
                
                mask = np.arange(nnconf3.natom) != indx[0]
                assert np.allclose(nn_num_dcTilde[:,:,mask],nn_dcTilde[:,:,mask], equal_nan=True), '"get_cTilde_grad": Derivative for nearest neighbors diverges from numeric differences!'
                assert np.allclose(num_dcTilde[:,:,indx[0],nnconf3.nnl[indx[0]] != indx[0]], - dcTilde[:,:,indx[0],nnconf3.nnl[indx[0]] != indx[0],indx[1]], equal_nan=True), '"get_cTilde_grad": Derivative for central atom diverges from numeric differences!'
            else :
                central_diff = -np.nan_to_num(dcTilde[:,:,indx[0],:,indx[1]]) - np.nan_to_num(num_dcTilde[:,:,indx[0]])
                nn = np.sort(nnconf3.nnl[indx[0],:nnconf3.nnn[indx[0]]])
                nn_diff = np.nan_to_num(dcTilde[:,:,nnconf3.nnl == indx[0],indx[1]]) - np.sum(np.nan_to_num(num_dcTilde[:,:,nn,:]),axis=-1)
                assert nn_diff == approx(0,abs=10**-8,rel=10**-6) , '"get_cTilde_grad": Derivative for nearest neighbors diverges from numeric differences!'
                assert central_diff == approx(0,abs=10**-8,rel=10**-6) , '"get_cTilde_grad": Derivative for central atom diverges from numeric differences!'
            
def test_c():
    for sett, glob, h in zip(Settings, Globals, Splines):
        sym_fac = np.empty((sett.Lmax+1)*(sett.Lmax+1))
        for l in range(sett.Lmax+1):
            sym_fac[l*l:(l+1)*(l+1)] = -(-1)**l
        sym_fac = sym_fac[np.newaxis,:,np.newaxis]
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            ref_c = get_c(sett, nnconf3, h[1])
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            c, dc, self_dc, _ = get_c_grad(sett, nnconf3, h[1])
            assert np.allclose(np.nan_to_num(c), np.nan_to_num(ref_c), equal_nan=True), '"get_c_grad" and "get_c" do not produce same data for c!'
            
            pconf = deepcopy(conf)
            mconf = deepcopy(conf)
            indx, ppos, mpos = make_num_diff(conf.atompos,dx)
            pconf.atompos = ppos
            mconf.atompos = mpos
            pnnconf2, pnnconf3 = get_nn_confs(sett, pconf)
            mnnconf2, mnnconf3 = get_nn_confs(sett, mconf)
            
            pc = get_c(sett, pnnconf3, h[1])
            mc = get_c(sett, mnnconf3, h[1])

            num_dc = (pc - mc) / (dx * 2)
            assert np.allclose(dc[:,:,indx[0],:nnconf3.nnn[indx[0]],indx[1]], sym_fac*num_dc[:,:,nnconf3.centraltype[indx[0]],nnconf3.nnl[indx[0],:nnconf3.nnn[indx[0]]]], equal_nan=True), '"get_c_grad": Derivative diverges from numeric differences!'
            assert np.allclose(self_dc[:,:,:,indx[0],indx[1]], num_dc[:,:,:,indx[0]], equal_nan=True) , '"get_c_grad": Derivative for self term diverges from numeric differences!'


def test_consistency_between_cTilde_and_cTilde_2b():
    for sett, glob, h in zip(SymSettings, SymGlobals, SymSplines):
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            cTilde, dcTilde = get_cTilde_grad_2b(sett, nnconf2, h[0])
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            cTilde_3b, dcTilde_3b = get_cTilde_grad(sett, nnconf3, h[1])
            assert np.allclose(cTilde, cTilde_3b[:,0], equal_nan=True), '"get_cTilde_grad_2b" and "get_cTilde_grad" do not produce same data for cTilde for l=0!'
            assert np.allclose(dcTilde, dcTilde_3b[:,0], equal_nan=True), '"get_cTilde_grad_2b" and "get_cTilde_grad" do not produce same data for dcTilde for l=0!'
            
def test_cTilde_2b():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            cTilde, dcTilde = get_cTilde_grad_2b(sett, nnconf2, h[0])
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            ref_cTilde = get_cTilde_2b(sett, nnconf2, h[0])
            assert np.allclose(cTilde, ref_cTilde, equal_nan=True), '"get_cTilde_grad_2b" and "get_cTilde_2b" do not produce same data for cTilde!'
            
            pconf = deepcopy(conf)
            mconf = deepcopy(conf)
            indx, ppos, mpos = make_num_diff(conf.atompos,dx)
            pconf.atompos = ppos
            mconf.atompos = mpos
            pnnconf2, pnnconf3 = get_nn_confs(sett, pconf)
            mnnconf2, mnnconf3 = get_nn_confs(sett, mconf)
            
            pcTilde = get_cTilde_2b(sett, pnnconf2, h[0])
            mcTilde = get_cTilde_2b(sett, mnnconf2, h[0])

            num_dcTilde = (pcTilde - mcTilde) / (dx * 2)
            if nnconf2.mult_per_img:
                nn_num_dcTilde = np.zeros((sett.Nmax2,pnnconf2.natom))
                nn_dcTilde = np.zeros((sett.Nmax2,nnconf2.natom))
                for i in range(pnnconf2.natom):
                    nn_num_dcTilde[:,i] = num_dcTilde[:,i,pnnconf2.nnl[i] == indx[0]].sum(-1)
                    nn_dcTilde[:,i] = dcTilde[:,i,nnconf2.nnl[i] == indx[0],indx[1]].sum(-1)
                
                mask = np.arange(nnconf2.natom) != indx[0]
                assert np.allclose(nn_num_dcTilde[:,mask],nn_dcTilde[:,mask], equal_nan=True), '"get_cTilde_grad_2b": Derivative for nearest neighbors diverges from numeric differences!'
                assert np.allclose(num_dcTilde[:,indx[0],nnconf2.nnl[indx[0]] != indx[0]], - dcTilde[:,indx[0],nnconf2.nnl[indx[0]] != indx[0],indx[1]], equal_nan=True), '"get_cTilde_grad_2b": Derivative for central atom diverges from numeric differences!'
            else :
                central_diff = -np.nan_to_num(dcTilde[:,indx[0],:,indx[1]]) - np.nan_to_num(num_dcTilde[:,indx[0]])
                nn = np.sort(nnconf2.nnl[indx[0],:nnconf2.nnn[indx[0]]])
                nn_diff = np.nan_to_num(dcTilde[:,nnconf2.nnl == indx[0],indx[1]]) - np.sum(np.nan_to_num(num_dcTilde[:,nn,:]),axis=-1)
                assert nn_diff == approx(0,abs=10**-8,rel=10**-6) , '"get_cTilde_grad_2b": Derivative for nearest neighbors diverges from numeric differences!'
                assert central_diff == approx(0,abs=10**-8,rel=10**-6) , '"get_cTilde_grad_2b": Derivative for central atom diverges from numeric differences!'

def test_consistency_between_c_and_c_2b():
    for sett, glob, h in zip(SymSettings, SymGlobals, SymSplines):
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            c, dc = get_c_grad_2b(sett, nnconf2, h[0])
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            c_3b, dc_3b = get_c_grad(sett, nnconf3, h[1])
            assert np.allclose(c, c_3b[:,0], equal_nan=True), '"get_c_grad_2b" and "get_c_grad" do not produce same data for cTilde for l=0!'
            assert np.allclose(dc, dc_3b[:,0], equal_nan=True), '"get_c_grad_2b" and "get_c_grad" do not produce same data for dcTilde for l=0!'

def test_c_2b():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            ref_c = get_c_2b(sett, nnconf2, h[0])
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            c, dc, self_dc, _ = get_c_grad_2b(sett, nnconf2, h[0])
            assert np.allclose(c, ref_c, equal_nan=True), '"get_c_grad_2b" and "get_c_2b" do not produce same data for c!'
            
            pconf = deepcopy(conf)
            mconf = deepcopy(conf)
            indx, ppos, mpos = make_num_diff(conf.atompos,dx)
            pconf.atompos = ppos
            mconf.atompos = mpos
            pnnconf2, pnnconf3 = get_nn_confs(sett, pconf)
            mnnconf2, mnnconf3 = get_nn_confs(sett, mconf)
            
            pc = get_c_2b(sett, pnnconf2, h[0])
            mc = get_c_2b(sett, mnnconf2, h[0])

            num_dc = (pc - mc) / (dx * 2)
            assert np.allclose(dc[:,indx[0],:nnconf2.nnn[indx[0]],indx[1]], - num_dc[:,nnconf2.centraltype[indx[0]],nnconf2.nnl[indx[0],:nnconf2.nnn[indx[0]]]], equal_nan=True), '"get_c_grad_2b": Derivative diverges from numeric differences!'
            assert np.allclose(self_dc[:,:,indx[0],indx[1]], num_dc[:,:,indx[0]], equal_nan=True) , '"get_c_grad_2b": Derivative for self term diverges from numeric differences!'

def test_consistency_between_cTilde_and_cTilde_ten_2b():
    for l, ten_glob, ten_h in zip(lamb, TenSymGlobals, TenSymSplines):
        for sett, glob, h in zip(SymSettings, ten_glob, ten_h):
            for conf in Configurations:
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                cTilde, dcTilde = get_cTilde_ten_grad_2b(sett, nnconf2, h[0], l)
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                cTilde_3b, dcTilde_3b = get_cTilde_grad(sett, nnconf3, h[1])
                assert np.allclose(cTilde, cTilde_3b[:,l**2:(l+1)**2], equal_nan=True), '"get_cTilde_ten_grad_2b" and "get_cTilde_grad" do not produce same data for cTilde for l=0!'
                assert np.allclose(dcTilde, dcTilde_3b[:,l**2:(l+1)**2], equal_nan=True), '"get_cTilde_ten_grad_2b" and "get_cTilde_grad" do not produce same data for dcTilde for l=0!'

def test_cTilde_ten_2b():
    for l, ten_glob, ten_h in zip(lamb, TenGlobals, TenSplines):
        for sett, glob, h in zip(Settings, ten_glob, ten_h):
            for conf in Configurations:
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                cTilde, dcTilde = get_cTilde_ten_grad_2b(sett, nnconf2, h[0], l)
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                ref_cTilde = get_cTilde_ten_2b(sett, nnconf2, h[0], l)
                assert np.allclose(cTilde, ref_cTilde, equal_nan=True), '"get_cTilde_ten_grad_2b" and "get_cTilde_ten_2b" do not produce same data for cTilde!'
                
                pconf = deepcopy(conf)
                mconf = deepcopy(conf)
                indx, ppos, mpos = make_num_diff(conf.atompos,dx)
                pconf.atompos = ppos
                mconf.atompos = mpos
                pnnconf2, pnnconf3 = get_nn_confs(sett, pconf)
                mnnconf2, mnnconf3 = get_nn_confs(sett, mconf)
                
                pcTilde = get_cTilde_ten_2b(sett, pnnconf2, h[0], l)
                mcTilde = get_cTilde_ten_2b(sett, mnnconf2, h[0], l)
    
                num_dcTilde = (pcTilde - mcTilde) / (dx * 2)
                if nnconf2.mult_per_img:
                    nn_num_dcTilde = np.zeros((sett.Nmax2,2*l+1,pnnconf2.natom))
                    nn_dcTilde = np.zeros((sett.Nmax2,2*l+1,nnconf2.natom))
                    for i in range(pnnconf2.natom):
                        nn_num_dcTilde[:,:,i] = num_dcTilde[:,:,i,pnnconf2.nnl[i] == indx[0]].sum(-1)
                        nn_dcTilde[:,:,i] = dcTilde[:,:,i,nnconf2.nnl[i] == indx[0],indx[1]].sum(-1)
                    
                    mask = np.arange(nnconf2.natom) != indx[0]
                    assert np.allclose(nn_num_dcTilde[:,:,mask],nn_dcTilde[:,:,mask], equal_nan=True), '"get_cTilde_ten_grad_2b": Derivative for nearest neighbors diverges from numeric differences!'
                    assert np.allclose(num_dcTilde[:,:,indx[0],nnconf2.nnl[indx[0]] != indx[0]], - dcTilde[:,:,indx[0],nnconf2.nnl[indx[0]] != indx[0],indx[1]], equal_nan=True), '"get_cTilde_ten_grad_2b": Derivative for central atom diverges from numeric differences!'
                else :
                    central_diff = -np.nan_to_num(dcTilde[:,:,indx[0],:,indx[1]]) - np.nan_to_num(num_dcTilde[:,:,indx[0]])
                    nn = np.sort(nnconf2.nnl[indx[0],:nnconf2.nnn[indx[0]]])
                    nn_diff = np.nan_to_num(dcTilde[:,:,nnconf2.nnl == indx[0],indx[1]]) - np.sum(np.nan_to_num(num_dcTilde[:,:,nn,:]),axis=-1)
                    assert nn_diff == approx(0,abs=10**-8,rel=10**-6) , '"get_cTilde_ten_grad_2b": Derivative for nearest neighbors diverges from numeric differences!'
                    assert central_diff == approx(0,abs=10**-8,rel=10**-6) , '"get_cTilde_ten_grad_2b": Derivative for central atom diverges from numeric differences!'

def test_consistency_between_c_and_c_ten_2b():
    for l, ten_glob, ten_h in zip(lamb, TenSymGlobals, TenSymSplines):
        for sett, glob, h in zip(SymSettings, ten_glob, ten_h):
            for conf in Configurations:
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                c, dc, _, _ = get_c_ten_grad_2b(sett, nnconf2, h[0], l)
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                c_3b, dc_3b, _, _ = get_c_grad(sett, nnconf3, h[1])
                assert np.allclose(c, c_3b[:,l**2:(l+1)**2], equal_nan=True), '"get_c_ten_grad_2b" and "get_c_grad" do not produce same data for c for l=0!'
                assert np.allclose(dc, dc_3b[:,l**2:(l+1)**2], equal_nan=True), '"get_c_ten_grad_2b" and "get_c_grad" do not produce same data for dc for l=0!'

def test_c_ten_2b():
    for l, ten_glob, ten_h in zip(lamb, TenGlobals, TenSplines):
        for sett, glob, h in zip(Settings, ten_glob, ten_h):
            for conf in Configurations:
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                ref_c = get_c_ten_2b(sett, nnconf2, h[0], l)
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                c, dc, self_dc, _ = get_c_ten_grad_2b(sett, nnconf2, h[0], l)
                assert np.allclose(c, ref_c, equal_nan=True), '"get_c_ten_grad_2b" and "get_c_ten_2b" do not produce same data for c!'
                
                pconf = deepcopy(conf)
                mconf = deepcopy(conf)
                indx, ppos, mpos = make_num_diff(conf.atompos,dx)
                pconf.atompos = ppos
                mconf.atompos = mpos
                pnnconf2, pnnconf3 = get_nn_confs(sett, pconf)
                mnnconf2, mnnconf3 = get_nn_confs(sett, mconf)
                
                pc = get_c_ten_2b(sett, pnnconf2, h[0], l)
                mc = get_c_ten_2b(sett, mnnconf2, h[0], l)
    
                num_dc = (pc - mc) / (dx * 2)
                # (-1)**l !!!!!!!!!???????????!!!!!!!!!!!
                assert np.allclose(dc[:,:,indx[0],:nnconf2.nnn[indx[0]],indx[1]], - (-1)**l *  num_dc[:,:,nnconf2.centraltype[indx[0]],nnconf2.nnl[indx[0],:nnconf2.nnn[indx[0]]]], equal_nan=True), '"get_c_ten_grad_2b": Derivative diverges from numeric differences!'
                assert np.allclose(self_dc[:,:,:,indx[0],indx[1]], num_dc[:,:,:,indx[0]], equal_nan=True) , '"get_c_ten_grad_2b": Derivative for self term diverges from numeric differences!'

def test_p():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            ref_c = get_c(sett, nnconf3, h[1])
            ref_p = get_p(sett,glob,nnconf3,ref_c)
            nnconf2, nnconf3 = get_nn_confs(sett, conf)
            c = get_c(sett, nnconf3, h[1])
            p, dp = get_p_grad(sett,glob,nnconf3,c)
            assert np.allclose(np.nan_to_num(p), np.nan_to_num(ref_p), equal_nan=True), '"get_p_grad" and "get_p" do not produce same data for p!'
            
            indx, pc, mc = make_num_diff(c,dx)
            
            pp = get_p(sett,glob,nnconf3,pc)
            mp = get_p(sett,glob,nnconf3,mc)

            num_dp = (pp - mp) / (dx * 2)
            
            l, m = lm_to_l_and_m(indx[1])
            dp[indx] *= 2
            
            assert np.allclose(dp[:,indx[1],:,indx[3]], num_dp[indx[0],:,l,indx[2],:,indx[3]], equal_nan=True), '"get_p_grad": Derivative diverges from numeric differences!'
            assert np.allclose(dp[:,indx[1],:,indx[3]], num_dp[:,indx[0],l,:,indx[2],indx[3]], equal_nan=True), '"get_p_grad": Derivative diverges from numeric differences!'
            
def test_p_ten():
    for l, ten_glob, ten_h in zip(lamb, TenGlobals, TenSplines):
        for sett, glob, h in zip(Settings, ten_glob, ten_h):
            for conf in Configurations:
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                ref_c = get_c(sett, nnconf3, h[1])
                ref_p = get_p_ten(sett,glob,nnconf3,ref_c, l)
                nnconf2, nnconf3 = get_nn_confs(sett, conf)
                c = get_c(sett, nnconf3, h[1])
                p, dp = get_p_ten_grad(sett,glob,nnconf3,c, l)
                assert np.allclose(np.nan_to_num(p), np.nan_to_num(ref_p), equal_nan=True), '"get_p_ten_grad" and "get_p" do not produce same data for p!'
                
                indx, pc, mc = make_num_diff(c,dx)
                
                pp = get_p_ten(sett,glob,nnconf3,pc,l)
                mp = get_p_ten(sett,glob,nnconf3,mc,l)
    
                num_dp = (pp - mp) / (dx * 2)
                
                ll, m = lm_to_l_and_m(indx[1])
                
                for i, len_c in enumerate(glob.len_cleb):
                    buf_dp = dp[:,i,:,:len_c,:,indx[3]]
                    for llp, lm_sum in zip(glob.ten_lm_to_l[indx[1]],glob.ten_lm_sum[i][indx[1]]):
                        if llp[0] == llp[1]:
                            buf_dp[0,indx[0],lm_sum[0],indx[2]] *= 2
                            buf_dp[1,indx[0],lm_sum[1],indx[2]] *= 2
                            
                        assert np.allclose(buf_dp[0,:,lm_sum[0]].sum(0), num_dp[i,indx[0],:,llp[0],indx[2],:,indx[3]], equal_nan=True), '"get_p_ten_grad": Derivative diverges from numeric differences!'
                        assert np.allclose(buf_dp[1,:,lm_sum[1]].sum(0), num_dp[i,:,indx[0],llp[1],:,indx[2],indx[3]], equal_nan=True), '"get_p_ten_grad": Derivative diverges from numeric differences!'

def test_Descriptor():
    for sett, glob, h in zip(Settings, Globals, Splines):
        for conf in Configurations:
            desc = get_Descriptor(sett,glob,conf,h)
            
            
                            
if __name__ == '__main__':
    test_cTilde()
    test_c()
    test_consistency_between_cTilde_and_cTilde_2b()
    test_cTilde_2b()
    test_consistency_between_c_and_c_2b
    test_c_2b()
    test_consistency_between_cTilde_and_cTilde_ten_2b()
    test_cTilde_ten_2b()
    test_consistency_between_c_and_c_ten_2b()
    test_c_ten_2b()
    test_p()
    test_p_ten()
    test_Descriptor()