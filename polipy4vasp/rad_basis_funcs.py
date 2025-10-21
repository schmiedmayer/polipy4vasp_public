#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:49:07 2022
@author: carolin
"""
import numpy as np
from scipy.special import spherical_jn
#from quadpy import quad
#from scipy.integrate import quad
from .sph import setup_asa, sph_harm
from .splines import get_qnl, radial_basis_function
from .splines import get_splines
from. preprocess import AU_to_Ang

def sph_bessel_analytic(Rcut, Nmax, Lmax, normG):
    r'''
    Calculates the radial basis functions
    
    .. math:: r_{nl} = 4 \pi a i^l  \frac{R_{\text{cut}}^2}{|\textbf{G}|^2 - q_{nl}^2} j_l(|\textbf{G}| R_{\text{cut}}) j'_l(q_{nl} R_{\text{cut}}) q_{nl}
    
    or 
    
    .. math:: 4 \pi i^l \sqrt{\frac{R_{\text{cut}}^3}{2}} j'_l(q_{nl} R_{\text{cut}})
    
    if :math:`|\textbf{G}| \approx q_{nl}`.
    
    Parameters
    ----------
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`
    Nmax : int, scalar
        Maximal main quantum number :math:`n_\text{max}`
    Lmax : int, scalar
        Maximal angular quantum number :math:`l_\text{max}`
    normG : ndarray
        Array containing all the norms of the wave vectors.
    Returns
    -------
    radial_term : ndarray
        Radial part of the basis functions.
    '''
    qnl = get_qnl(Rcut, Nmax, Lmax).transpose()
    #derivative of spherical bessel function at qnl * Rcut for all l
    djq = np.asarray([spherical_jn(l, qnl[:, l] * Rcut, derivative = True) for l in range(Lmax + 1)]).T
    #spherical bessel function at G * Rcut for all l
    jg = np.asarray([spherical_jn(l, normG * Rcut) for l in range(Lmax + 1)])
    #find difference between normG and qnl
    compare_q_g = normG[np.newaxis, np.newaxis, :] - qnl[:, :, np.newaxis]
    #approximation for the radial part, used when normG ~ qnl
    a_approx = (Rcut**3 / 2 * djq**2)
    #no approximation
    a = Rcut**2 / (normG[np.newaxis, np.newaxis, :]**2 - qnl[:, :, np.newaxis]**2) * (jg[np.newaxis, :] 
            * djq[:, :, np.newaxis] * qnl[:, :, np.newaxis])
    
    #decide when approximation to use and multiply by 4pi and normalization factor for the spherical bessel functions
    j_term = np.where(np.abs(compare_q_g) < 1e-5, a_approx[:, :, np.newaxis], a) * np.pi * 4 / np.sqrt(a_approx)[:, :, np.newaxis]
    #expand to combined index and multiply by i^l
    radial_term = np.zeros((Nmax, (Lmax + 1)**2, len(normG)), dtype = 'complex128')
    for l in range(Lmax + 1):
        radial_term[:, l**2 : (l + 1)**2, :] = j_term[:, l, :][:, np.newaxis] * 1j**l #* (-1)**l
    return radial_term

def pref( ):
   """ 
      returns fourier prefactors for STOs basis functions
   """
   return np.array( [4.,
                     9.2376043070340122321463804880313,
                     20.238577025077627724792918684369,
                     43.271897842100048768492458361136,
                     91.225170572774189972327588140289,
                     190.56316628960683366394689895492,
                     395.51348702025120058422281727376,
                     816.96914582925666139910973918249,
                     1681.308631377617906137546146679,
                     3449.9725179347084494263033600983,
                     7062.3188993431834434762271846826,
                     14428.428204880373153441280385513,
                     29428.334786863015160449151394487,
                     59936.698549911957110390308499087,
                     121922.66344265294032099734246268,
                     247747.09986269451110058275249598,
                     502945.65845056913678927505716584,
                     1.0201599915281611306711051022182e6,
                     2.0677080539379837444042294150502e6,
                     4.1880986917179073017796800146519e6,
                     8.4777307557745483034357631666226e6,
                     1.7151484919824371884299810518606e7,
                     3.4682019687348455175500152497639e7,
                     7.0098070814801598128389850511917e7,
                     1.4161948920354579080570768306053e8,
                     2.8600235095128735684217853084494e8,
                     5.7737575590603205916015662574103e8,
                     1.1652019647473204446286320825001e9,
                     2.3507571886784548795943807789291e9,
                     4.7411903075815770303511128328124e9,
                     9.5597890882411784958646459314989e9,
                     1.9270723439021001990018510964791e10,
                     3.8836787954779645127913532377437e10] )

def overlap(z, n, m, l):
   """ 
   computes the overlap (w.r.t. r^2 dr) of two Slater type orbitals 
   """
   return  (2*z[m]/(z[m]+z[n]))**(l+1.5) * (2*z[n]/(z[m]+z[n]))**(l+1.5) 


def slater_like_analytic_1(Lmax, Nmax, normG, rcut):    
    """
      Slater 1, 2 and 3 basis functions are defined as : 
      R_nl( r) = N_nl r^l Exp( -z_nl r ) 
      N_nl = (2z_nl)^(3/2 +l ) / sqrt( (2l+2)!)
      This implies that <R_nl|R_nl> = int_0^infty dr r^2 R_nl(r)^2 = 1 
      The Fourier transformation is obtained via 
      R_nl(q) = 4*pi* i^l  int_0^infty dr r^2 j_l( q r ) R_nl( r ) 
      it has essentially a form that looks like
      R_nl(q) = 4*pi* i^l  pf(l) * sqrt(1/z_nl^3) * (q/z_nl)^l / (1+(q/z_nl)^2)^(l+2)
      here pf(l) is a number 
    """
    # sanity check
    if Nmax>4 :
       raise ValueError( " Slater1 basis functions implemented only for Nmax<=4, requested Nmax={n:d}".format(n=Nmax))

    # initialize radial basis function in reciprocal space 
    radial_term = np.zeros((Nmax, (Lmax + 1)**2, len(normG)), dtype = 'complex128')
    u = np.zeros((Nmax, (Lmax + 1), len(normG)))
    
    zetanl = np.loadtxt('./BasisFunctions/STOs/phi_N_4.dat')[:4 * (Lmax + 1), :]

    zetanl = np.reshape(zetanl[:, 0], (Nmax, Lmax + 1), order = 'F') #(4, Lmax + 1), order = 'F') 

    #scaling for better results and faster decay 
    if Lmax > 0:
        zetanl[:, 1] *= 1 / 3
    if Lmax > 1:
        zetanl[:, 2] *= 0.2
        
    zetanl = zetanl[:Nmax]
    zetanl = np.flip(zetanl, axis =  0)

    # loop over all angular quantum numbers 
    for l in range(Lmax + 1):
        zetan = zetanl[:, l]
    
        # determine overlap
        S = np.zeros((len(zetan), len(zetan)), dtype='float64')
        
        # Compute Löwdin orthogonalization matrix 
        for n in range( Nmax ):
            for m in range( Nmax ):
                S[n, m] = overlap(zetan, n, m, l)       
        L = np.linalg.cholesky(S[:, :])
        S[:, :] = np.linalg.inv(L).T
       # print('S:\n', S)
        #S = np.eye(len(zetan))
        
        for n in range( Nmax ): 
           z=zetan[n]

           # denominator of each individual function
           den = (1+(normG[:]/z)**2)
           arg = normG[:]/z
           prefactor = np.sqrt( 1/z**3 )

           # basis functions: sqrt( (2z)^3 / (2l+2)! ) int_r r^2 (2zr)^(l) exp(-z r) j_l( q r)
           # the fourier transform yields a prefactor that is 
           pf = pref() 
          
           # the fourier transform has following general form
           u[n,l,:] = pf[l] * prefactor * ( (arg[:] / den[:] ) ** l  / den[:]**2 )

        radial_term[:, l**2 : (l + 1)**2, :] = (1j**l * np.matmul(S.T[:, :], u[:, l, :]) * 4 * np.pi)[:, np.newaxis, :]
    
    return np.flip(radial_term, axis = 0)

def slater_like_analytic_2(Lmax, Nmax, normG, Rcut): 
    """
      Slater 1, 2 and 3 basis functions are defined as : 
      R_nl( r) = N_nl r^l Exp( -z_nl r ) 
      N_nl = (2z_nl)^(3/2 +l ) / sqrt( (2l+2)!)
      This implies that <R_nl|R_nl> = int_0^infty dr r^2 R_nl(r)^2 = 1 
      The Fourier transformation is obtained via 
      R_nl(q) = 4*pi* i^l  int_0^infty dr r^2 j_l( q r ) R_nl( r ) 
      it has essentially a form that looks like
      R_nl(q) = 4*pi* i^l  pf(l) * sqrt(1/z_nl^3) * (q/z_nl)^l / (1+(q/z_nl)^2)^(l+2)
      here pf(l) is a number 
    """
    # initialize radial basis function in reciprocal space 
    radial_term = np.zeros((Nmax, (Lmax + 1)**2, len(normG)), dtype = 'complex128')
    u = np.zeros((Nmax, (Lmax + 1), len(normG)))

    # loop over all angular quantum numbers 
    for l in range(Lmax + 1):
        # read in proper basis function
        zetanl = np.loadtxt('./BasisFunctions/STO1/phi_l_{l:d}_N_{n:d}.dat'.format( n=Nmax, l=l))[:Nmax, :]

        wn = np.reshape(zetanl[:, 1], (Nmax, 1), order = 'F')
        zetan = np.reshape(zetanl[:, 0], (Nmax, 1), order = 'F')
        zetan = np.flip(zetan)
    
        # determine overlap
        S = np.zeros((len(zetan), len(zetan)), dtype='float64')
        # Compute Löwdin orthogonalization matrix 
        
        for n in range( Nmax ):
            for m in range( Nmax ):
                S[n, m] = overlap(zetan, n, m, l)       
        #print( S )
        #raise TypeError('STOP')
        L = np.linalg.cholesky(S[:, :])
        S[:, :] = np.linalg.inv(L).T

        
        for n in range( Nmax ): 
           z=zetan[n]
           #w=wn[n]

           # denominator of each individual function
           den = (1+(normG[:]/z)**2)
           arg = normG[:]/z
           #prefactor = np.sqrt( w/z**3 )
           prefactor = np.sqrt( 1/z**3 )

           # basis functions: int_r (2zr)^(n-1) exp(-z r) j_0( q r)
           pf = pref() 
          
           u[n,l,:] = pf[l] * prefactor * arg[:]**l / den[:] ** (l+2) 

        radial_term[:, l**2 : (l + 1)**2, :] = (1j**l * np.matmul(S.T[:, :], u[:, l, :]) * 4 * np.pi)[:, np.newaxis, :]

    return np.flip(radial_term, axis = 0)


def slater_like_analytic_3(Lmax, Nmax, normG, rcut, scale, zeta_max):    
    """
      Slater 1, 2 and 3 basis functions are defined as : 
      R_nl( r) = N_nl r^l Exp( -z_nl r ) 
      N_nl = (2z_nl)^(3/2 +l ) / sqrt( (2l+2)!)
      This implies that <R_nl|R_nl> = int_0^infty dr r^2 R_nl(r)^2 = 1 
      The Fourier transformation is obtained via 
      R_nl(q) = 4*pi* i^l  int_0^infty dr r^2 j_l( q r ) R_nl( r ) 
      it has essentially a form that looks like
      R_nl(q) = 4*pi* i^l  pf(l) * sqrt(1/z_nl^3) * (q/z_nl)^l / (1+(q/z_nl)^2)^(l+2)
      here pf(l) is a number 
    """
    
    # initialize radial basis function in reciprocal space 
    radial_term = np.zeros((Nmax, (Lmax + 1)**2, len(normG)), dtype = 'complex128')
    u = np.zeros((Nmax, (Lmax + 1), len(normG)))
            
    if zeta_max == 0:        
        zetanl =  np.zeros((5, Lmax + 1))
        zetanl[:, 0] = np.loadtxt('optimized_exponents_l0_Al.dat')#[:Nmax]
        zetanl = zetanl[:Nmax, :]
        zetanl = np.flip(zetanl, axis =  0)
    else:
        zetanl =  np.zeros((Nmax, Lmax + 1))
        zetanl[0, 0] = zeta_max
        for i in range(1, Nmax):
            zetanl[i, 0] = zeta_max / scale**i
        zetanl = np.flip(zetanl, axis =  0)
    #print('zeta', zetanl)
        

    #loop over all angular quantum numbers 
    for l in range(Lmax + 1):
        zetan = zetanl[:, 0] * (0.3)**l

        #here it works better without orthonormalization
        S = np.eye(len(zetan))

        for n in range( Nmax ): 
           z=zetan[n]

           # denominator of each individual function
           den = (1+(normG[:]/z)**2)
           arg = normG[:]/z
           prefactor = np.sqrt( 1/z**3 )

           # basis functions: sqrt( (2z)^3 / (2l+2)! ) int_r r^2 (2zr)^(l) exp(-z r) j_l( q r)
           # the fourier transform yields a prefactor that is 
           pf = pref() 
          
           # the fourier transform has following general form
           # pf[l] /sqrt( z^3 ) * ( (q/z) / ( 1+(q/z)^2 ) )^l / ( 1+(q/z)^2 ) ^ 2
           u[n,l,:] = pf[l] * prefactor * ( (arg[:] / den[:] ) ** l  / den[:]**2 ) #* norms[n, l][np.newaxis]

        radial_term[:, l**2 : (l + 1)**2, :] = (1j**l * np.matmul(S.T[:, :], u[:, l, :]) * 4 * np.pi)[:, np.newaxis, :] 
    return np.flip(radial_term, axis = 0)
