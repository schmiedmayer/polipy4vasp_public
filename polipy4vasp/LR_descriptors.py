#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import floor, sqrt
from scipy.special import spherical_jn
from .sph import setup_asa, sph_harm
from .splines import get_qnl, radial_basis_function
from .positions import get_rlattice_and_v, NN_Configuration
from .splines import get_splines


# prefactors for STOs basis functions

pf = np.array( [4.,
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
            
    zetanl =  np.zeros((Nmax, Lmax + 1))
    zetanl[0, 0] = zeta_max
    for i in range(1, Nmax):
        zetanl[i, 0] = zeta_max / scale**i
    zetanl = np.flip(zetanl, axis =  0)
        

    #loop over all angular quantum numbers 
    for l in range(Lmax + 1):
        zetan = zetanl[:, 0]# * (0.3)**l

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
           # the fourier transform has following general form
           # pf[l] /sqrt( z^3 ) * ( (q/z) / ( 1+(q/z)^2 ) )^l / ( 1+(q/z)^2 ) ^ 2
           u[n,l,:] = pf[l] * prefactor * ( (arg[:] / den[:] ) ** l  / den[:]**2 ) #* norms[n, l][np.newaxis]

        radial_term[:, l**2 : (l + 1)**2, :] = (1j**l * np.matmul(S.T[:, :], u[:, l, :]) * 4 * np.pi)[:, np.newaxis, :] 
    return np.flip(radial_term, axis = 0)

def get_rec_lattice_points(mx, my, mz):
    r'''
    Calculates the integers :math:`m_i` in each direction for later calculating the wave vectors
    
    .. math:: \textbf{G} = m_{x} * \textbf{b}_{1} + m_{y} * \textbf{b}_2 + m_{x} * \textbf{b}_3
    
    where the :math:`\textbf{b}_i` are the reciprocal lattice vectors.
    This is done by creating the arrays
    
    .. math:: [0, 1, ...,  m_i / 2 - 1, -m_i / 2, ..., -1] \qquad \text{if } m_i \text{ is even}
    
    .. math:: [0, 1, ...,  (m_i - 1)/ 2, -(m_i - 1) / 2, ..., -1] \qquad \text{if } m_i \text{ is odd}
    
    with the help of numpy.fft.fftfreq.

    Parameters
    ----------
    mx : integer, scalar
        Number of reciprocal grid points in x-direction
    my : integer, scalar
        Number of reciprocal grid points in y-direction
    mz : integer, scalar
        Number of reciprocal grid points in z-direction

    Returns
    -------
    ndarray
        Array of multipliers for :math:`\textbf{b}_{1}`
    ndarray
        Array of multipliers for :math:`\textbf{b}_{2}`
    ndarray
        Array of multipliers for :math:`\textbf{b}_{3}`

    '''
    return np.fft.fftfreq(mx) * mx, np.fft.fftfreq(my) * my, np.fft.fftfreq(mz) * mz

def get_wave_vectors(lattice, freq1, freq2, freq3):
    r'''
    Calculates the array containing the wave vectors
    
    .. math:: \textbf{G} = m_{x} * \textbf{b}_{1} + m_{y} * \textbf{b}_{2} + m_{x} * \textbf{b}_{3}
    
    where the :math:`\textbf{b}_{i}` are the reciprocal lattice vectors.

    Parameters
    ----------
    lattice : ndarray
        Matrix containing the lattice vectors as rows.
    freq1 : ndarray
        Array of integers for the reciprocal latice points in x-direction (obtained from get_rec_lattice_points).
    freq2 : ndarray
        Array of integers for the reciprocal latice points in y-direction (obtained from get_rec_lattice_points).
    freq3 : ndarray
        Array of integers for the reciprocal latice points in x-direction (obtained from get_rec_lattice_points).

    Returns
    -------
    G : ndarray
        Array with the wave vectors of size :math:`(m_{x} \cdot m_{y} \cdot m_{z}) \times 3`.
    normG : ndarray
        Array containing all the norms of the wave vectors.

    '''
    b, _ = get_rlattice_and_v(lattice) #b = reciprocal lattice matrix
    b = b.T
    #multiplication by 2pi here just for simplicity, is later needed for the structure factor
    G =np.asarray(
            [np.ndarray.flatten(np.add.outer(b[0, 0] * freq1, np.add.outer(b[0, 1] * freq2, b[0, 2] * freq3))),
             np.ndarray.flatten(np.add.outer(b[1, 0] * freq1, np.add.outer(b[1, 1] * freq2, b[1, 2] * freq3))),
             np.ndarray.flatten(np.add.outer(b[2, 0] * freq1, np.add.outer(b[2, 1] * freq2, b[2, 2] * freq3)))
        ]).T * 2 * np.pi
    normG = np.linalg.norm(G, axis = 1)
    return G, normG

def get_G2(lattice, freq1, freq2, freq3):
    r'''
    Calculates an array containing the squared norm of all wave vectors. 

    Parameters
    ----------
    lattice : ndarray
        Matrix containing the lattice vectors as rows.
    freq1 : ndarray
        Array of integers for the reciprocal latice points in x-direction (obtained from get_rec_lattice_points).
    freq2 : ndarray
        Array of integers for the reciprocal latice points in y-direction (obtained from get_rec_lattice_points).
    freq3 : ndarray
        Array of integers for the reciprocal latice points in x-direction (obtained from get_rec_lattice_points).

    Returns
    -------
    G2 : ndarray
        Array containing the squared norms of all wave vectors.

    '''
    b, v = get_rlattice_and_v(lattice)
    b = b.T
    lat_const = np.linalg.norm(b, axis = 0)
    G2 = np.ndarray.flatten(np.add.outer(lat_const[0]**2 * freq1**2, np.add.outer(lat_const[1]**2 * freq2**2, lat_const[2]**2 * freq3**2)))
    return G2.T * v

def calculate_atom_density(G, normG, positions, maxtypes, types, sigma):
    r'''
    Calculate the atom density and apply a Gaussian brodening to it such that
    
    .. math:: \tilde{\rho}(\textbf{G}) = \rho(\textbf{G}) \exp\left( - \frac{|\textbf{G}|^2 \sigma^2}{2} \right)
    
    is the final result for the broadened density and the exponetina term is the broadening with the brodening parameter :math:`\sigma`.
    
    .. math:: \rho(\textbf{G}) = \sum_{\textbf{r}} \exp(i\textbf{G} \textbf{r})
    
    is the atom density where :math:`\textbf{r}` is a vector containing the position of an atom in real space.

    Parameters
    ----------
    G : ndarray
        Array with the wave vectors of size :math:`(m_x \cdot m_y \cdot m_z) \times 3`.
    normG : ndarray
        Array containing all the norms of the wave vectors.
    positions : ndarray
        Array with the positions of each atom in real space and size :math:`N_{\text{atom}} \times 3`.
    maxtypes : integer, scalar
        Number of different atom types in the system.
    types : ndarray
        Array containing the atom type of each atom.
    sigma : scalar
        Broadening parameter.

    Returns
    -------
    rho : ndarray
        Atom density.

    '''
    rho = np.zeros((np.shape(G)[0], maxtypes), dtype = 'complex128')
    #calculate the atom density for each atom type and consider only the atoms of the respective type
    for t in range(maxtypes):
        mask = t  == types
        pos = positions[mask].T
        rho[:, t] = np.sum(np.exp(1j * np.dot(G, pos)), axis = 1)
    rho *= np.exp(-normG**2 * sigma**2 / 2)[:, np.newaxis]
    return rho

def calculate_basis_functions(settings, normalized_G, normG):
    r'''
    Calculates the basis functions in reciprocal space
    
    .. math:: f_{lmn}(\textbf{G}) = 4 \pi c i^l Y_{lm}(-\hat{\textbf{G}}) \frac{R_{\text{cut}}^2}{|\textbf{G}|^2 - q_{nl}^2} j_l(|\textbf{G}| R_{\text{cut}}) j'_l(q_{nl} R_{\text{cut}}) q_{nl}
                      
    where 
    
    .. math:: c = \left( \frac{R_{\text{cut}}^3}{2} j'_{l}(q_{nl}R_{\text{cut}})^2 \right)^{-\frac{1}{2}}
                        
    is the approximation of the normalization factor for the spherical Bessel functions :math:`j_l`.
                      
    Parameters
    ----------
    Rcut : scalar
        Cutoff radius for calulating the radial part of the basis functions.
    Nmax : integer, scalar
        Maximal main quantum number :math:`n_\text{max}`.
    Lmax : integer, scalar
        Maximal angular quantum number :math:`l_\text{max}`.
    normalized_G : ndarray
        Array of normalized wave vectors.
    normG : ndarray
        Array containing all the norms of the wave vectors.

    Returns
    -------
    flmn : ndarray
        Array of basis functions of size :math:`( \text{len}(\textbf{G}) \times (l_\text{max} + 1)^2 \times n_\text{max}`).
    '''

    # -normalizedG to avoid multiplication by -1 for odd values of l
    y = sph_harm(settings.Lmax, -normalized_G) 
    y = y[:, :, 0]
    
    radial_term = slater_like_analytic_3(settings.Lmax,
                                         settings.Nmax3,
                                         normG,
                                         settings.Rcut2,
                                         settings.scale,
                                         settings.zeta_max)
    
    #multiply radial and angular term
    flmn = np.asarray((radial_term * y[np.newaxis, :, :]).transpose(), dtype = 'complex128')
    return flmn


def get_c_LR_grad(settings, configuration):
    r'''
    
    Parameters
    ----------
    Lmax : integer, scalar
        Maximal angular quantum number :math:`l_\text{max}`.
    Nmax : integer, scalar
        Maximal main quantum number :math:`n_\text{max}`.
    lattice : ndarray
        Matrix containing the lattice vectors as rows.
    positions : ndarray
        Array with the positions of each atom in real space and size :math:`N_{atoms} \times 3`.
    mx : integer, scalar
        Number of reciprocal grid points in x-direction
    my : integer, scalar
        Number of reciprocal grid points in y-direction
    mz : integer, scalar
        Number of reciprocal grid points in z-direction
    Rcut : scalar
        Cutoff radius for calulating the radial part of the basis functions.
    types : ndarray
        Array containing the atom type of each atom.
    maxtypes : integer, scalar
        Number of different atom types in the system.
    sigma : scalar
        Broadening parameter.

    Returns
    -------
    c : ndarray
        Array containing all expansion coefficients with size (:math:`n_\text{max} \times (l_\text{max} + 1)^2 \times N_{\text{type}} \times N_{\text{atom}}`)
    dc : ndarray
        Array containing all the derivatives of the expansion coefficients with respect to the atom positions. The derivatives of the respective central
        atoms are not included.
        Has size: (:math:`n_\text{max} \times (l_\text{max} + 1)^2 \times N_{\text{atom}} \times (N_{\text{atom}} - 1) \times 3`).
    self_dc: ndarray
        Array containing the derivatives of the expansion coefficients of the respective central atom.
        Has size: (:math:`n_\text{max} \times (l_\text{max} + 1)^2 \times N_{\text{atom}} \times (N_{\text{atom}} - 1) \times 3`)

    '''

    def prep(settings, lattice, positions, sigma):
        [mx, my, mz] = settings.ReciprocalPoints
        freq1, freq2, freq3 = get_rec_lattice_points(mx, my, mz) 
        G, normG = get_wave_vectors(lattice, freq1, freq2, freq3)
        normG = np.where(abs(normG) < 1e-10, 1e-10, normG) #avoid 0 entries 
        normalized_G = np.where(normG[:, np.newaxis, np.newaxis] == 0, G[:, np.newaxis, :], G[:, np.newaxis, :] / normG[:, np.newaxis, np.newaxis])
        flmn = calculate_basis_functions(settings, normalized_G, normG)
        rho = calculate_atom_density(G, normG, positions, maxtypes, types, sigma)
        return flmn, rho, G, normG
    
    Nmax = settings.Nmax3
    Lmax = settings.Lmax
    sigma = settings.SigmaAtom

    lattice, positions, types, maxtypes = configuration.lattice, configuration.atompos, configuration.atomtype, configuration.maxtype

    flmn, rho, G, normG = prep(settings, lattice, positions, sigma)

    exp = np.exp(-1j * G @ positions.T)

    c =  np.einsum('Mln,MJi->nlJi', flmn, rho[:, :, np.newaxis] * exp[:, np.newaxis, :], optimize = True)

    mask_diag = np.eye(len(positions),dtype= bool)
    rho_term = np.zeros((len(positions),len(positions),len(G),maxtypes), dtype = 'complex128')
    rho_term[np.eye(len(positions),dtype= bool)] = rho
    
    k_term = np.random.rand(settings.ReciprocalPoints[0]**3, len(positions)) + 1j * np.random.rand(settings.ReciprocalPoints[0]**3, len(positions))
    k_term[:, :] = (np.exp(1j * G @ positions.T) * np.exp(-normG**2 * sigma**2 / 2)[:,np.newaxis])
    k_term = np.swapaxes(k_term, 0, 1)
    
    dc = np.zeros((Nmax, (Lmax + 1)**2, len(positions), len(positions), 3))
    self_dc = np.random.rand(Nmax, (Lmax+1)**2, maxtypes, len(positions), 3)
    
    exp = G[:, np.newaxis, :] * exp[:, :, np.newaxis]
    
    for J in range(maxtypes):
        mask_k = types == J
        type_term = np.where(mask_k[:, np.newaxis], k_term, np.zeros((len(positions), rho_term.shape[2]))) 
        dc_buf = (type_term - rho_term[:, :, :, J]) #expression in the brakets
        dc_buf = np.swapaxes(dc_buf, 1, 2)
        dc_buf = np.swapaxes(dc_buf, 0, 1)
        dc_buf = dc_buf[:, :, :,  np.newaxis] * exp[:, :, np.newaxis, :]

        helper_dc = np.real(1j * np.einsum('Mln,Mija->nlija', flmn, dc_buf, optimize = True))
        dc += helper_dc
        self_dc[:, :, J] = helper_dc[:, :, mask_diag]
        
    dc = dc[:,:, ~mask_diag,:  ].reshape(Nmax, (Lmax + 1)**2, len(positions), len(positions) - 1, 3) #comment out for try out   
    
    c_self = np.sum(flmn * np.exp(- normG**2 * sigma**2 / 2)[:, np.newaxis, np.newaxis], axis = 0)[0]
    c[:, 0, :, :] -= c_self[:, np.newaxis, np.newaxis]

    _, v = get_rlattice_and_v(lattice)
    c = np.real(c) / v
    dc = dc / v
    self_dc = self_dc / v
        
    return c, dc, self_dc

def get_c_LR(settings, configuration):
    r'''
    
    Parameters
    ----------
    Lmax : integer, scalar
        Maximal angular quantum number :math:`l_\text{max}`.
    Nmax : integer, scalar
        Maximal main quantum number :math:`n_\text{max}`.
    lattice : ndarray
        Matrix containing the lattice vectors as rows.
    positions : ndarray
        Array with the positions of each atom in real space and size :math:`N_{atoms} \times 3`.
    mx : integer, scalar
        Number of reciprocal grid points in x-direction
    my : integer, scalar
        Number of reciprocal grid points in y-direction
    mz : integer, scalar
        Number of reciprocal grid points in z-direction
    Rcut : scalar
        Cutoff radius for calulating the radial part of the basis functions.
    types : ndarray
        Array containing the atom type of each atom.
    maxtypes : integer, scalar
        Number of different atom types in the system.
    sigma : scalar
        Broadening parameter.

    Returns
    -------
    c : ndarray
        Array containing all expansion coefficients with size (:math:`n_\text{max} \times (l_\text{max} + 1)^2 \times N_{\text{type}} \times N_{\text{atom}}`)
   

    '''

    def prep(settings, lattice, positions, sigma):
        [mx, my, mz] = settings.ReciprocalPoints
        freq1, freq2, freq3 = get_rec_lattice_points(mx, my, mz) 
        G, normG = get_wave_vectors(lattice, freq1, freq2, freq3)
        normG = np.where(abs(normG) < 1e-10, 1e-10, normG) #avoid 0 entries 
        normalized_G = np.where(normG[:, np.newaxis, np.newaxis] == 0, G[:, np.newaxis, :], G[:, np.newaxis, :] / normG[:, np.newaxis, np.newaxis])
        flmn = calculate_basis_functions(settings, normalized_G, normG)
        rho = calculate_atom_density(G, normG, positions, maxtypes, types, sigma)
        return flmn, rho, G, normG
    
    Nmax = settings.Nmax3
    Lmax = settings.Lmax
    sigma = settings.SigmaAtom

    lattice, positions, types, maxtypes = configuration.lattice, configuration.atompos, configuration.atomtype, configuration.maxtype

    flmn, rho, G, normG = prep(settings, lattice, positions, sigma)

    exp = np.exp(-1j * G @ positions.T)

    c =  np.einsum('Mln,MJi->nlJi', flmn, rho[:, :, np.newaxis] * exp[:, np.newaxis, :], optimize = True)

    _, v = get_rlattice_and_v(lattice)
    c = np.real(c) / v
        
    return c
