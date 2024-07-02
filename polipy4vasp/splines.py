#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
===========================================================================
Splines, Cutoff Functions and, Basis Functions (:mod:`polipy4vasp.splines`)
===========================================================================

.. currentmodule:: polipy4vasp.splines
    
"""
import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import spherical_jn 
from scipy.special import spherical_in

def cutoff_function(Rcut,r):
    r"""
    This is the cutoff function for :math:`r\leq R_\text{cut}` defined by Behler and Parrinello [#]_ as,
        
    .. math:: f_\text{cut}=\frac{1}{2}\left(\cos\left(\pi\frac{r}{R_\text{cut}}\right)+1\right).
        
    Arguments
    ---------
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`
    r    : array_like
        Argument of the cutoff function :math:`r`
        
    Returns
    -------
    fcut : ndarray
        :math:`f_\text{cut}`
    
    References
    ----------
    .. [#] https://doi.org/10.1103/PhysRevLett.98.146401
    """
    return 0.5*(np.cos(np.pi*r/Rcut)+1)

def get_qnl(Rcut,Nmax,Lmax):
    r"""
    This routine is used to compute the parameters :math:`q_{nl}`.
    
    Arguments
    ---------
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`
    Nmax : int, scalar
        Maximal main quantum number :math:`n_\text{max}`
    Lmax : int, scalar
        Maximal angular quantum number :math:`l_\text{max}`
        
    Returns
    -------
    qnl  : ndarray
        Parameters :math:`q_{nl}`
    
    Notes
    -----
    The parameters :math:`q_{nl}` are defined such that the spherical Bessel function
    of the first kind is zero at :math:`j_n(q_{nl}R_\text{cut})`. The parameters 
    are computed by finding the roots of :math:`j_n(r)` via the classic Brent’s method [#]_.
    The roots of a lower order spherical Bessel function are used to calculate the roots
    of a higher one by using its roots as bracketing interval.
    
    .. warning::
        Difference to VASP. In VASP :math:`l` is truncated for large :math:`n`. Not all 
        :math:`l` are computed in order to save compute time this is not done in this
        code.
        
    
    References
    ----------
    .. [#]
       Brent, R. P.,
       *Algorithms for Minimization Without Derivatives*.
       Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.
    
    """
    # hard coded roots for n=0 same as for sin(x)
    x = np.arange(1,Nmax+Lmax+1)*np.pi
    qnl = [x[:Nmax]]
    # calculating roots for higer orders up to Lmax
    for l in range(1,Lmax+1):
        f = lambda r : spherical_jn(l,r)
        buf = [brentq(f, x[j], x[j+1]) for j in range(Nmax+Lmax-l)]
        x = np.array(buf)
        qnl.append(x[:Nmax])
    return np.array(qnl)/Rcut

def radial_basis_function(qnl,Rcut,r):
    r"""
    This is the radial basis function for :math:`r\leq R_\text{cut}` defined by the normalized
    spherical Bessel function of the first kind,
        
    .. math::  \chi_{nl}(r) = \hat{j}_l(q_{nl}r).
    
    Arguments
    ---------
    qnl  : array_like
        Parameters :math:`q_nl`
    Rcut : scalar
        The cutoff radius :math:`R_\text{cut}`
    r    : array_like
        Argument :math:`r` of the radial basis function
    
    Returns
    -------
    chi_nl  : ndarray
        Radial basis function :math:`\chi_{nl}(r)`
    
    Notes
    -----
    The normalization is done by numerical integration using the technique from
    the FORTRAN library QUADPACK.
    
    """
    chi_nl = []
    for l,q in enumerate(qnl):
        # calculating norm
        f = lambda r : spherical_jn(l,q*r)*spherical_jn(l,q*r) *r*r
        norm = np.sqrt(quad_vec(f,0,Rcut)[0])
        
        # calculating chi_nl
        chi_nl.append(spherical_jn(l,q*r)/norm)
    return np.array(chi_nl).T

def exp_times_bessel(Lmax,SigmaAtom,r,rp):
    r"""
    This function returns an array of all orders :math:`l` up to :math:`l_\text{max}` of the modified
    spherical Bessel function of the first kind :math:`\iota_l` multiplied with an exponential function like,
    
    .. math:: f_l(r,r') = \iota_l\left(\frac{rr'}{\sigma^2_\text{atom}}\right)\exp\left(-\frac{r^2+r'^2}{2\sigma^2_\text{atom}}\right).
    
    Arguments
    ---------
    Lmax : int, scalar
        Maximal angular quantum number :math:`l_\text{max}`
    SigmaAtom : scalar
        Width of the Gaussian function :math:`\sigma_\text{atom}`
    r    : array_like
        Argument :math:`r`
    rp   : array_like
        Argument :math:`r'`
        
    Returns
    -------
    f    : ndarray
        Function :math:`f_l(r,r')`
        
    Notes
    -----
    For large :math:`R_\text{cut}` and small :math:`\sigma_\text{atom}` the individual 
    functions :math:`\iota_l` and :math:`\exp` become very large/little respectively.
    In extrem cases the floating point precision of float64 is not sufficent. Therfore 
    the product of these two functions was derived analyticaly before implementing. For
    :math:`l=0` the function is calculated as
    
    .. math:: f_0(r,r') = \frac{\sigma^2_\text{atom}}{2rr'}\left(\exp\left(-\frac{1}{2\sigma_\text{atom}}(r-r')^2\right)-\exp\left(-\frac{1}{2\sigma_\text{atom}}(r+r')^2\right)\right).
    """
    # combinging parameter
    SigmaAtom2 = SigmaAtom*SigmaAtom
    mr2 = 0.5*(r-rp)**2/SigmaAtom2
    pr2 = 0.5*(r+rp)**2/SigmaAtom2
    rrp = r*rp/SigmaAtom2
    # generating mask for differentiating between large and small rrp to avoid devision with small denominator
    mask = rrp > 1
    amask = np.logical_not(mask)
    # pre calculating exponential function
    exprrp = np.exp(-0.5*(r**2+rp**2)[amask]/SigmaAtom2)
    # hard coded for Lmax = 0
    f = np.zeros([Lmax+1,*rrp.shape])
    f[0,mask] = 0.5*(np.exp(-mr2[mask])-np.exp(-pr2[mask]))/rrp[mask]
    f[0,amask] = exprrp * spherical_in(0,rrp[amask])
    # hard coded for Lmax = 1
    if Lmax >= 1 :
        f[1,mask] = 0.5*(rrp[mask]*(np.exp(-mr2[mask])+np.exp(-pr2[mask]))-(np.exp(-mr2[mask])-np.exp(-pr2[mask])))/rrp[mask]**2
        f[1,amask] = exprrp * spherical_in(1,rrp[amask])
    # calculate for Lmax > 1
    if Lmax > 1 :
        for l in range(2,Lmax+1):
            f[l,mask] = f[l-2,mask]-(2*(l-1)+1)*f[l-1,mask]/rrp[mask]
            f[l,amask] = exprrp * spherical_in(l,rrp[amask])
    return f

def get_hnl(Rcut,SigmaAtom,Nmax,Lmax,r):
    r"""
    This function returns an array of all orders :math:`l` up to :math:`l_\text{max}`
    and :math:`n` up to :math:`n_\text{max}` for the function :math:`h_{nl}(r)` defined as
        
    .. math::   h_{nl}(r)=\frac{4\pi}{(2\pi\sigma^2_\text{atom})^{\frac{3}{2}}}f_\text{cut}(r)\int_{0}^{R_\text{cut}}\chi_{nl}(r')\exp \left(-\frac{r'^2+r^2}{2\sigma^2_\text{atom}}\right)\iota_l\left(\frac{rr'}{\sigma^2_\text{atom}}\right)r'^2\mathrm{d} r'.
    
    The integral is solved numericaly using the technique from the FORTRAN
    library QUADPACK.
    
    Arguments
    ---------
    Rcut      : scalar
        The cutoff radius :math:`R_\text{cut}`
    SigmaAtom : scalar
        Width of the gaussian distribution :math:`\sigma_\text{atom}`
    Nmax      : int, scalar
        Maximal main quantum number :math:`n_\text{max}`
    Lmax      : int, scalar
        Maximal angular quantum number :math:`l_\text{max}`
    r         : 
        Argument :math:`r`
        
    Returns
    -------
    hnl    : ndarray
        Function :math:`h_{nl}(r)`    
    """
    qnl = get_qnl(Rcut,Nmax,Lmax)
    # set up integrand
    f = lambda rp : (exp_times_bessel(Lmax,SigmaAtom,r,rp)[np.newaxis,:,:]*
                     radial_basis_function(qnl,Rcut,rp)[:,:,np.newaxis])*rp*rp
    # performe integration
    integral = quad_vec(f,0,Rcut)[0]
    return 4*np.pi*(2*np.pi*SigmaAtom*SigmaAtom)**-1.5*cutoff_function(Rcut,r)*integral
    
def get_splines(settings,splineDensity = 100):
    r"""
    This routine generates cubic splines for the the function :math:`h_{nl}(r)` defined
    as:
        
    .. math::   h_{nl}(r)=\frac{4\pi}{(2\pi\sigma^2_\text{atom})^{\frac{3}{2}}}f_\text{cut}(r)\int_{0}^{R_\text{cut}}\chi_{nl}(r')\exp \left(-\frac{r'^2+r^2}{2\sigma^2_\text{atom}}\right)\iota_l\left(\frac{rr'}{\sigma^2_\text{atom}}\right)r'^2\mathrm{d} r'.
    
    When the spline function is call, it automatically generates an array for all
    orders :math:`l` up to :math:`l_\text{max}` and :math:`n` up to :math:`n_\text{max}`.
    The derivaive is also calculated.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    splineDensity :  int, scalar, optional
        Sets the density of the :math:`r`-grid 
        
    Returns
    -------
    hnl   : CubicSpline [#]_
        Cubic spline of :math:`h_{nl}(r)`
    
    Examples
    --------
    In this example the cubic spline of :math:`h_{nl}(r)` is calculated for :math:`n_\text{max}=8`,
    :math:`l_\text{max}=4`, :math:`\sigma_\text{atom}=0.4`, and :math:`R_\text{cut}=5`.
    The cubic spline of :math:`h_{nl}(r)` and its first derivative is plotted betwen 0 and :math:`R_\text{cut}`
    for :math:`n=2` and :math:`l=1`
    
    >>> import polipy4vasp as pp
    >>> from polipy4vasp.splines import get_splines
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> settings = pp.Setup(Rcut = 5,
    ...                     Nmax = 8,
    ...                     Lmax = 4,
    ...                     SigmaAtom = 0.4)
    >>> hnl = get_splines(settings)
    >>> r = np.linspace(0, 5, 50)
    >>> fig, ax = plt.subplots(figsize=(6.5, 4))
    >>> ax.plot(r,hnl(r)[2, 1],label="h")
    >>> ax.plot(r,hnl(r, 1)[2, 1],label="h'")
    >>> ax.set_xlim(0, 5)
    >>> ax.legend(loc='upper right')
    >>> plt.show()
    
    .. image:: /figures/Splines.png
    
    
    References
    ----------
    .. [#] `Cubic Spline Interpolation
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_
            on docs.scipy.
    """

    rMesh = np.linspace(0,settings.Rcut,splineDensity)
    hMesh = get_hnl(settings.Rcut,settings.SigmaAtom,settings.Nmax,settings.Lmax,rMesh)
    return CubicSpline(rMesh,hMesh,axis=2)
