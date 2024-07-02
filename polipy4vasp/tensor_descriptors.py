#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from joblib import Parallel, delayed
from polipy4vasp.preprocess import Ang_to_AU, calc_ionic_borncharges, calc_ionic_polarisation, polarisation_to_minimgcon, fix_polarisation_timeseries
from tqdm import tqdm

def get_ten_p(settings,glob,nnconf,c):
    p = np.zeros([2*settings.lamb+1,settings.Nmax,settings.Nmax,glob.len_m_index_sum,nnconf.maxtype,nnconf.maxtype,nnconf.natom])
    dp = np.zeros([2,2*settings.lamb+1,settings.Nmax,glob.max_len_cleb,nnconf.maxtype,nnconf.natom])
    for k, ind in enumerate(glob.c_index):
        buf = np.multiply(c[:,np.newaxis,ind[0],:,np.newaxis,:],c[np.newaxis,:,ind[1],np.newaxis,:,:])*glob.cleb[k]
        dp[0,k,:,:glob.len_cleb[k]]= c[:,ind[1]]*glob.cleb[k][:,0,:,:,0,:]
        dp[1,k,:,:glob.len_cleb[k]]= c[:,ind[0]]*glob.cleb[k][:,0,:,:,0,:]
        for l, m in enumerate(glob.m_index_sum[k]):
            p[k,:,:,l] = np.sum(buf[:,:,m],axis=2)
    return p, dp
    
def get_ten_norm(lc):
    abslc = np.sqrt(np.linalg.norm(np.einsum('lim,kim->ilk',lc,lc),ord='fro',axis=(1,2)))
    hatlc = lc / abslc[None,:,None]
    return hatlc, abslc 

def get_ten_dK(desc,lrc,J):
    K = np.moveaxis(np.tensordot(desc.lc[J],lrc,axes=([-1],[-1])),[0,1,2,3],[2,0,3,1])
    dK = lrc
    return K, dK

def get_ten_K(desc,lrc,J):
    K = np.moveaxis(np.tensordot(desc.lc[J],lrc,axes=([-1],[-1])),[0,1,2,3],[2,0,3,1])
    return K

def get_ten_normK(desc,lrc,J):
    hatlc, abslc = get_ten_norm(desc.lc[J])
    hatlrc, _ = get_ten_norm(lrc)
    K = np.moveaxis(np.tensordot(hatlc,hatlrc,axes=([-1],[-1])),[0,1,2,3],[2,0,3,1])
    return K

def get_ten_dnormK(settings,desc,lrc,J):
    def XXTX(X):
        out = np.empty_like(X)
        for i, x in enumerate(np.swapaxes(X,0,1)):
            out[:,i,:] = x.dot(x.T).dot(x)
        return out
    hatlc, abslc = get_ten_norm(desc.lc[J])
    hatlrc, _ = get_ten_norm(lrc)
    dK  = -np.tensordot(hatlc,hatlrc,axes=([-1],[-1]))[None,:,:,:,:,None] * XXTX(hatlc)[:,None,:,None,None,:]
    dK += (np.eye(2 * settings.lamb + 1)[:,:,None,None,None]*hatlrc[None,None])[:,:,None]
    dK /= abslc[None,None,:,None,None,None]
    return np.moveaxis(dK,(0,1,2,3,4,5),(0,2,1,4,3,5))

def ten_dnormKxdp(settings,glob,desc,dK,J):
    def contract1(a,b):
        sa = a.shape
        aa = np.prod(sa[1:5]) * sa[6]
        a  = np.transpose(a,(0,1,2,3,4,6,5,7)).reshape(sa[0],aa,-1)
        sb = b.shape
        b  = b.reshape(-1,sb[-1]).T
        out = np.empty((sa[0],aa))
        for i, [x,y] in enumerate(zip(a,b)):
            out[i] = np.dot(x,y)
        return np.transpose(out.reshape(*sa[:5],sa[6]),(4,5,0,2,1,3))
    def contract2(a,b):
        sa = a.shape
        aa = np.prod(sa[1:5]) * sa[6]
        a  = np.transpose(a,(0,1,2,3,5,7,4,6)).reshape(sa[0],aa,-1)
        sb = b.shape
        b  = b.reshape(-1,sb[-1]).T
        out = np.empty((sa[0],aa))
        for i, [x,y] in enumerate(zip(a,b)):
            out[i] = np.dot(x,y)
        return np.transpose(out.reshape(*sa[:5],sa[6]),(4,5,0,2,1,3))
    size = dK.shape[:-1]
    dK   = dK.reshape(*size,settings.Nmax,settings.Nmax,glob.len_m_index_sum,desc.maxtype,desc.maxtype)
    out  = np.zeros((settings.Nmax,(settings.Lmax+1)*(settings.Lmax+1),desc.maxtype,size[1],size[3],2*settings.lamb+1,2*settings.lamb+1))
    for i, len_c in enumerate(glob.len_cleb):
        buf_dp = desc.dp[J][:,i,:,:len_c]
        for lm, lm_sum in enumerate(glob.ten_lm_sum[i]):
            for l, llp in enumerate(glob.ten_lm_to_l[lm]):
                if np.sum(lm_sum[l,0]) > 0 :
                    out[:,lm] += contract1(dK[i,:,:,:,:,:,:,llp[0]],np.sum(buf_dp[0,:,lm_sum[l,0]],axis=0))
                if np.sum(lm_sum[l,1]) > 0 :
                    out[:,lm] += contract2(dK[i,:,:,:,:,:,:,llp[1]],np.sum(buf_dp[1,:,lm_sum[l,1]],axis=0))
    return out

def ten_dnormKxdpxdc(settings,glob,desc,dK,J):
    def contract1(a,b):
        sa = a.shape
        sb = b.shape
        a  = a.reshape(-1,np.prod(sa[2:]))
        b  = b.reshape(-1,np.prod(sb[2:]))
        return np.dot(b.T,a).reshape(*sb[2:],*sa[2:])
    def contract2(a,b):
        sa = a.shape
        sb = b.shape
        a  = np.swapaxes(a.reshape(-1,sa[2],np.prod(sa[3:])),0,1)
        b  = np.swapaxes(b.reshape(-1,*sb[2:]),0,1)
        out = np.empty((sa[2],sb[3],np.prod(sa[3:])))
        for i, [x, y] in enumerate(zip(a,b)):
            out[i] = np.dot(y.T,x)
        return out.reshape(sa[2],sb[3],*sa[3:])
    buf = ten_dnormKxdp(settings,glob,desc,dK,J)
    out = np.zeros((desc.natom,3,*buf.shape[4:]))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += contract1(buf[:,:,Jp,i],desc.dc[J][:,:,i,do])
        out[desc.centraltype[J]] += contract2(buf[:,:,Jp],desc.self_dc[J][:,:,Jp])
    return out

def ten_dKxdp(settings,glob,desc,dK,J):
    def contract1(a,b):
        sa = a.shape
        aa = sa[0] * sa[1] * sa[2] * sa[4]
        a  = np.transpose(a,(0,1,2,4,3,5)).reshape(aa,-1)
        sb = b.shape
        b  = b.reshape(-1,sb[-1])
        return np.transpose(np.dot(a,b).reshape(*sa[:3],sa[4],sb[-1]),(2,3,4,1,0))
    def contract2(a,b):
        sa = a.shape
        aa = sa[0] * sa[1] * sa[2] * sa[4]
        a  = np.transpose(a,(0,1,3,5,2,4)).reshape(aa,-1)
        sb = b.shape
        b  = b.reshape(-1,sb[-1])
        return np.transpose(np.dot(a,b).reshape(*sa[:3],sa[4],sb[-1]),(2,3,4,1,0))
    size = dK.shape[:2]
    dK = dK.reshape(*size,settings.Nmax,settings.Nmax,glob.len_m_index_sum,desc.maxtype,desc.maxtype)
    out = np.zeros((settings.Nmax,(settings.Lmax+1)*(settings.Lmax+1),desc.maxtype,desc.dp[J].shape[-1],size[1],2*settings.lamb+1,2*settings.lamb+1))
    for i, len_c in enumerate(glob.len_cleb):
        buf_dp = desc.dp[J][:,i,:,:len_c]
        for lm, lm_sum in enumerate(glob.ten_lm_sum[i]):
            for l, llp in enumerate(glob.ten_lm_to_l[lm]):
                if np.sum(lm_sum[l,0]) > 0 :
                    out[:,lm,:,:,:,i] += contract1(dK[:,:,:,:,llp[0]],np.sum(buf_dp[0,:,lm_sum[l,0]],axis=0))
                if np.sum(lm_sum[l,1]) > 0 :
                    out[:,lm,:,:,:,i]+= contract2(dK[:,:,:,:,llp[1]],np.sum(buf_dp[1,:,lm_sum[l,1]],axis=0))
    return out

def ten_dKxdpxdc(settings,glob,desc,dK,J):
    def contract1(a,b):
        sa = a.shape
        sb = b.shape
        a  = a.reshape(-1,np.prod(sa[2:]))
        b  = b.reshape(-1,np.prod(sb[2:]))
        return np.dot(b.T,a).reshape(*sb[2:],*sa[2:])
    def contract2(a,b):
        sa = a.shape
        sb = b.shape
        a  = np.swapaxes(a.reshape(-1,sa[2],np.prod(sa[3:])),0,1)
        b  = np.swapaxes(b.reshape(-1,*sb[2:]),0,1)
        out = np.empty((sa[2],sb[3],np.prod(sa[3:])))
        for i, [x, y] in enumerate(zip(a,b)):
            out[i] = np.dot(y.T,x)
        return out.reshape(sa[2],sb[3],*sa[3:])
    buf = ten_dKxdp(settings,glob,desc,dK,J)
    out = np.zeros((desc.natom,3,*buf.shape[4:]))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            if np.sum(do) > 0: out[dl[i,do]] += contract1(buf[:,:,Jp,i],desc.dc[J][:,:,i,do])
        out[desc.centraltype[J]] += contract2(buf[:,:,Jp],desc.self_dc[J][:,:,Jp])
    return out

def ten_3body_dKxdc(settings,glob,desc,dK,J):
    out = np.zeros((desc.natom,3,dK.shape[1],2*settings.lamb+1,2*settings.lamb+1))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('iln,njda->dalji',dK[:,:,Jp::desc.maxtype],desc.dc[J][:,glob.two_body_mask,i][:,:,do], optimize = True)
        out[desc.centraltype[J]] += np.einsum('jln,nkia->ialkj',dK[:,:,Jp::desc.maxtype],desc.self_dc[J][:,glob.two_body_mask,Jp,:], optimize = True)
    return out
            

def get_ten_phi(settings,desc,lrc):
    phi = []
    for J in range(desc.maxtype):
        if settings.Zeta == 0 : K = get_ten_K(desc,lrc[J],J)
        else: K = get_ten_normK(desc,lrc[J],J)
        phi.append(np.sum(K,axis=0))
    return np.swapaxes(np.concatenate(phi,0),0,1).reshape(2*settings.lamb+1,-1) #why swap axes?

    
def get_ten_dphi(settings,glob,desc,lrc):
    dphi = []
    if settings.Zeta == 0 :
        for J in range(desc.maxtype):
            dK = lrc[J]
            dK = ten_dKxdpxdc(settings,glob,desc,dK,J)
            #dK = settings.Beta[1]*ten_dKxdpxdc(settings,glob,desc,dK[:,:,settings.Nmax*desc.maxtype:],J) + settings.Beta[0]*ten_3body_dKxdc(settings,glob,desc,dK[:,:,:settings.Nmax*desc.maxtype],J)
            dphi.append(dK)
    else : 
        for J in range(desc.maxtype):
            dK = get_ten_dnormK(settings,desc,lrc[J],J)
            dK = ten_dnormKxdpxdc(settings,glob,desc,dK,J)
            dphi.append(dK)
            
    return np.moveaxis(np.concatenate(dphi,2),2,3).reshape(desc.natom,3,2*settings.lamb+1,-1) #why swap axes?

def get_ten_PHI(settings,glob,descriptors,lrc):
    if settings.deriv :
        dPHI = np.concatenate(Parallel(n_jobs=settings.ncore,backend='threading')(delayed(get_ten_dphi)(settings,glob,desc,lrc) for desc in tqdm(descriptors, desc='Bulding PHI')),0)
        dPHI = dPHI.reshape(-1,dPHI.shape[-1])
        if settings.lamb == 0 :
            PHI = np.concatenate(Parallel(n_jobs=settings.ncore,require='sharedmem')(delayed(get_ten_phi)(settings,desc,lrc) for desc in tqdm(descriptors, desc='Bulding PHI')),0)
            PHI = np.vstack([PHI*settings.Wene/descriptors[0].natom,-dPHI*settings.Wforc])
        else : PHI = dPHI
    else :
        PHI = np.concatenate(Parallel(n_jobs=settings.ncore,require='sharedmem')(delayed(get_ten_phi)(settings,desc,lrc) for desc in tqdm(descriptors, desc='Bulding PHI')),0)        
    return PHI
    
def get_ten_Y(settings,data):
    if settings.deriv :
        # acoustic sum rule
        Y = np.concatenate([np.roll(ten-np.mean(ten,axis=0)-calc_ionic_borncharges(conf,settings.Charges),shift=-1,axis=2) for ten, conf in zip(data.tensors, data.configurations)],0).reshape(-1)
        
        #Z = np.array(data.tensors).trace(axis1=-2,axis2=-1).mean(axis=0)
        #Z = np.array([np.diag([z]*3) for z in Z])
        #Y = np.concatenate([np.roll(ten-np.mean(ten,axis=0)-Z,shift=-1,axis=2) for ten, conf in zip(data.tensors, data.configurations)],0).reshape(-1)
    else :
        Y = [polarisation_to_minimgcon(Ang_to_AU(ten)-calc_ionic_polarisation(conf,settings.Charges),conf) for ten, conf in zip(data.tensors, data.configurations)]
        Y = np.concatenate(np.roll(Y,shift=-1,axis=1),0).reshape(-1)
        
    sig = 1.0 # np.std(Y)
    Y /= sig
    return Y, sig
        
def ten_solver(Y,X):
    if X.shape[0] <= X.shape[1] : print('WARNING: Desing Matrix over defined reduce LRC!!!')
    w, _, _, singular = np.linalg.lstsq(X,Y,rcond=-1)
    out = [np.dot(X,w),Y]
    return w, singular, out

def get_ten_dK_predict(desc,lrc,J,w):
    dK = np.dot(np.moveaxis(lrc,(0,1,2),(2,1,0)).reshape(lrc.shape[2],-1),w[J])
    K = np.rollaxis(np.dot(desc.lc[J],dK), 1)
    return K, dK

def ten_dKxdp_predict(settings,glob,desc,dK,J):
    dK = dK.reshape(settings.Nmax,settings.Nmax,glob.len_m_index_sum,desc.maxtype,desc.maxtype)
    out = np.zeros((settings.Nmax,(settings.Lmax+1)*(settings.Lmax+1),desc.maxtype,desc.dp[J].shape[-1],2*settings.lamb+1))
    for i, len_c in enumerate(glob.len_cleb):
        buf_dp = desc.dp[J][:,i,:,:len_c]
        for lm, lm_sum in enumerate(glob.ten_lm_sum[i]):
            for l, llp in enumerate(glob.ten_lm_to_l[lm]):
                if np.sum(lm_sum[l,0]) > 0 :
                    out[:,lm,:,:,i] += np.einsum('pnqJ,nJi->pqi',dK[:,:,llp[0]],np.sum(buf_dp[0,:,lm_sum[l,0]],axis=0))
                if np.sum(lm_sum[l,1]) > 0 :
                    out[:,lm,:,:,i]+= np.einsum('npJq,nJi->pqi',dK[:,:,llp[1]],np.sum(buf_dp[1,:,lm_sum[l,1]],axis=0))
    return out

def ten_dKxdpxdc_predict(settings,glob,desc,dK,J):
    buf = ten_dKxdp_predict(settings,glob,desc,dK,J)
    out = np.zeros((desc.natom,3,*buf.shape[4:]))
    dl = desc.derivl[desc.centraltype[J]]
    for Jp in range(desc.maxtype):
        for i, do in enumerate(desc.derivtype[Jp][desc.centraltype[J]]):
            out[dl[i,do]] += np.einsum('nml,nmda->dal',buf[:,:,Jp,i],desc.dc[J][:,:,i,do], optimize = True)
        out[desc.centraltype[J]] += np.einsum('nmil,nmia->ial',buf[:,:,Jp],desc.self_dc[J][:,:,Jp], optimize = True)
    return out
    
def get_ten_phi_predict(settings,desc,lrc,w):
    phi = np.zeros(2*settings.lamb + 1)
    for J in range(desc.maxtype):
        K, _ = get_ten_dK_predict(desc,lrc[J],J,w)
        phi += np.sum(K,axis=0)
    return phi
    
def get_ten_dphi_predict(settings,glob,desc,lrc,w):
    dphi = np.zeros((desc.natom,3,2*settings.lamb+1))
    for J in range(desc.maxtype):
        _, dK = get_ten_dK_predict(desc,lrc[J],J,w)
        dK = ten_dKxdpxdc_predict(settings,glob,desc,dK,J)
        dphi += dK
    return dphi 
