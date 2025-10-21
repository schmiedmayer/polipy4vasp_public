# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate._cubic import CubicSpline
from dataclasses import dataclass, field
from time import time
import pickle

from .sph import setup_asa
from .TEN_ML_AB_reader import read_data
from .splines import get_splines
from .preprocess import pre_process, ten_pre_process, split_vali_Data
from .descriptors import get_AllDescriptors, get_Descriptor, get_AllTenDescriptors, get_TenDescriptor, get_TenDescriptor_grad
from .reference import get_LRC
from .train import get_Y, get_w
from .desing import get_PHI, get_phi
from .globals import setup_globals, Globals
from .console_output import print_output, print_time, print_lrc
from .tensor_descriptors import get_ten_Y, get_ten_PHI, get_ten_phi_predict, get_ten_dphi_predict
from .make_linear_fast import make_fast, fast_prediction, fast_prediction_grad
from .tensor_classes import Tensor_Type


@dataclass
class Setup:
    r'''
    Class for defining the input parameters used for training a MLFF.
    
    Arguments
    ---------
        Rcut2 : scalar, optional
            The cutoff radius :math:`R_\text{cut}` for the two-body descriptor. Default 8 angstrom.
        Rcut3 : scalar, optional
            The cutoff radius :math:`R_\text{cut}` for the three-body descriptor. Default 8 angstrom.
        SigmaAtom : scalar, optional
            Width of the gaussian atomic brodening function :math:`\sigma_\text{atom}`. Default 0.4 angstrom.
        Beta : float, optional
            The weighting parameter for the radial descriptors, :math:`\beta^{(2)}`. The weighting parameter for the angular descriptors is computed via :math:`\beta^{(3)} = 1 - \beta^{(2)}`. :math:`\beta^{(2)}` has to fulfill :math:`1 \geq \beta^{(2)} \geq 0`. Default 0.9
        EpsCur : scalar, optional
            Threshold :math:`\epsilon_\text{cur}` for the CUR algorithm. Descriptors that produce eigenvalues larger than :math:`\epsilon_\text{cur}` are used as Local refferenc structures. Default 1e-10.
        Nmax2 : int, optional
            Maximal main quantum number :math:`n_\text{max}` for the two-body descriptor. Default 12.
        Nmax3 : int, optional
            Maximal main quantum number :math:`n_\text{max}` for the three-body descriptor. Default 8.
        Lmax : int, optional
            Maximal angular quantum number :math:`l_\text{max}`. Default 4.
        Kernel : str, optional
            Type of kernel used. Default "poli"
        Zeta : scalar, optional
            Power of the polinomial kernel :math:`\zeta`. Default 4.
        SigmaExp : scalar, optional
            :math:`\sigma` used for the gaussian kernel.
        Waderiv : scalar, optional
            Weight parameter for scaling the anti derivative for fitting. Default 1.
        AlgoLRC : int, optional
            Integer for specifing the algorythm used for selecting local refferenc configuration. Default = 1.
        NLRC : list, optional
            Number of local refferenc configurations for each atomic species. Default = [15]
        ncore : int, optional
            Number of cores for paralelisation. Defailt = -1 (all cores)
        Scatter_Plot: bool, optional
            If ``True`` a scatter plot will be displaied. Default = ``True``
    
    
    Examples
    --------
    In this example it is shown how to specifi the parameters before training.
    We will change :math:`R_\text{cut}` to 4 , :math:`N_\text{max}` to 6, and 
    :math:`\zeta` to 3. For the remaning parametes the default vallues are used.
    
    >>> import polipy4vasp as pp
    >>> settings = pp.Setup(Rcut3 = 4,
    ...                     Nmax3 = 6,
    ...                     Zeta  = 3)
    
    '''
    Rcut2: float = 8.
    Rcut3: float = 5.
    SigmaAtom: float = 0.5
    Beta: float = 0.9
    EpsCur: float = 1e-10
    Nmax2: int = 12
    Nmax3: int = 8
    Lmax: int = 4
    Kernel: str = "poli"
    Zeta: int = 4
    SigmaExp: float = 0.4
    Waderiv: float = 1
    AlgoLRC: int = 1
    NLRC : list = field(default_factory=lambda:[15])
    ncore: int = -1
    Scatter_Plot: bool = True
    Validation: float = 0
    TenArgs: tuple = None
    #LR
    LR: bool = False
    zeta_max: float = 2.0
    scale: float = 1.5
    ReciprocalPoints: list = field(default_factory=lambda:[5, 5, 5])

    def _check_type(self):
        r'''
        Used to check user input and give feedback if an error occures.
        '''
        if not isinstance(self.Rcut2,float) and not isinstance(self.Rcut2,int):
            raise TypeError('Rcut2 has wrong Type! Must be float (or int).')
        if not isinstance(self.Rcut3,float) and not isinstance(self.Rcut2,int):
            raise TypeError('Rcut3 has wrong Type! Must be float (or int).')
        if not isinstance(self.SigmaAtom,float) and not isinstance(self.SigmaAtom,int):
            raise TypeError('SigmaAtom has wrong Type! Must be float (or int).')
        if not isinstance(self.Beta,float) and not isinstance(self.Beta,int):
            raise TypeError('Beta has wrong Type! Must be float (or int).')
        if self.Beta < 0 or self.Beta > 1 :
            raise TypeError('Beta must be between [0,1]')
        if not isinstance(self.EpsCur,float) and not isinstance(self.EpsCur,int):
            raise TypeError('EpsCur has wrong Type! Must be float (or int).')
        if not isinstance(self.Nmax2,int) or not isinstance(self.Nmax3,int):
            raise TypeError('Nmax has wrong Type! Must be int!')
        if not isinstance(self.Lmax,int) :
            raise TypeError('Nmax has wrong Type! Must be int!')
        if not isinstance(self.Zeta,float) and not isinstance(self.Zeta,int):
            raise TypeError('Zeta has wrong Type! Must be int (or float).')
        if not isinstance(self.Waderiv,float) and not isinstance(self.Waderiv,int):
            raise TypeError('Waderiv has wrong Type! Must be float (or int).')

            
            
    def train(self,filename='ML_AB'):
        r'''
        Trains a MLFF using the traning data from a given file.
        
        Arguments
        ---------
        filename : str, optional
            Name and path of the ML_AB file, default = "ML_AB"
    
        Returns
        -------
        MLFF : ML_ForceField
            A machine leard force field
        
        Examples
        --------
        After defining the settings, done in the previous example, one can train a machine learned 
        force field as long as a ML_AB file from VASP is provided.
        
        >>> mlff = settings.train('ML_AB')
            Read input file. Dedected  134 training Configurations.
            Selected  188 local refferenc configurations for atom type Si.
            Condition number of PHI :  3.6E+11
            Done gennerating MLFF in  32.1 sec.
            Abs mean error in total energy = 0.156 meV/atom
            Abs mean error in forces       = 0.009 eV/Angstrom
        
        Output may vary.
        '''
        self._check_type()
        Data = read_data(filename)
        begin = time()
        setup_asa(self.Lmax)
        glob = setup_globals(self,Data)
        if self.Validation > 0: Data, vali_Data = split_vali_Data(Data,self.Validation)
        if Data._ten :
            h = get_splines(self,Data.Type.lamb)
            post_process_args, pre_process_args = ten_pre_process(self,Data)
            l_descriptors = get_AllTenDescriptors(self,glob,Data.configurations,h,Data.Type)
        else :
            h = get_splines(self)
            post_process_args, pre_process_args = pre_process(self,Data)
            l_descriptors = get_AllDescriptors(self,glob,Data.configurations,h)
        lrc, nlrc, _ = get_LRC(self,l_descriptors,Data)
        print_lrc(Data.atomname,nlrc)
        if not Data._ten :
            Y = get_Y(Data,pre_process_args)
            PHI = get_PHI(self,glob,l_descriptors,lrc,Data.maxtype,pre_process_args)
        else :
            Y = get_ten_Y(Data)
            PHI = get_ten_PHI(self,glob,Data.Type,l_descriptors,lrc)
        w, out = get_w(self,Data,PHI,Y,nlrc)
        print_time(time() - begin)
        err = print_output(self,Data,out,pre_process_args)
        if self.Validation > 0:
            print('Validation:')
            if Data._ten :
                vali_descriptors = get_AllTenDescriptors(self,glob,vali_Data.configurations,h,vali_Data.Type)
                _, pre_process_args = ten_pre_process(self,vali_Data)
                vali_PHI = get_ten_PHI(self,glob,vali_Data.Type,vali_descriptors,lrc)
                if vali_Data.Type.summ :
                    vali_out = [vali_PHI @ np.hstack(w), get_ten_Y(vali_Data)]
                else :
                    vali_out = [[phi @ ww, y] for phi, ww, y in zip(vali_PHI,w,get_ten_Y(vali_Data))]
                    
            else:
                vali_descriptors = get_AllDescriptors(self,glob,vali_Data.configurations,h)
                pre_process_args[0] = np.array([conf.natom for conf in vali_Data.configurations])
                vali_PHI = get_PHI(self,glob,vali_descriptors,lrc,vali_Data.maxtype,pre_process_args)
                vali_out = [vali_PHI @ np.hstack(w), get_Y(vali_Data,pre_process_args)]
            err = {'train' : err, 'test' : print_output(self,vali_Data,vali_out,pre_process_args)}
        lrc,w = make_fast(self,Data,lrc,w)
        if Data._ten :
            mlff = ML_TenField(settings = self,
                               errors = err,
                               glob = glob,
                               lrc =lrc,
                               w = w,
                               post_process_args = post_process_args,
                               h = h,
                               TenType = Data.Type)
        else:
            mlff = ML_ForceField(settings = self,
                                 errors = err,
                                 glob = glob,
                                 lrc =lrc,
                                 w = w,
                                 post_process_args = post_process_args,
                                 h = h)

        if mlff.w == 0: mlff._fast = True
        return mlff

        
@dataclass
class ML_ForceField:
    r'''
    The machine learned force field.
    
    Arguments
    ---------
    settings : Setup
        Class containing all settings for predicting.
    errors : dict
        Information ont training and testing errors.
    glob : Globals
        Class containing all global variables.
    lrc : list
        The local refferenc configurations
    w : ndarray
        Vector containing the optimal weights :math:`\textbf{w}`
    post_process_args : tuple
        Arguments for post processing.
    h : CubicSpline
        Cubic spline of :math:`h_{nl}(r)`
    TenType : Tensor_Type
        
    '''
    settings : Setup
    errors : dict
    glob : Globals
    lrc : list
    w : np.ndarray
    post_process_args : tuple
    h : CubicSpline
    _fast : bool = False
    _check : str = '*polipy4vasp~mlff*'
    
    
    def save(self,filename='ML_FF'):
        r'''
        Saves the machine learnd force field to a binary file.
        
        Arguments
        ---------
        filename : str, optional
            Name of the file where the machine learnd force field is stored
        '''
        with open(filename,'wb') as file:
            pickle.dump(self, file)
    
    def predict(self,conf,write_output=True):
        r'''
        Predicts the energy and forces of a given configutation.
        
        Arguments
        ---------
        conf : Configuration
            An atomic configuration
        write_output : logical, optional
            Logical operator for writing out energies, default = ``Ture``
        Returns
        -------
        E : scalar
            The energy of the configuration
        F : ndarray
            Array containing the forces
        
        Examples
        --------
        Before predicting one needs to load in a configuration. This can be done using
        ``polipy4vasp.read_POSCAR()`` to read in a VASP POSCAR file.
        
        >>> conf = pp.read_POSCAR('POSCAR')
        >>> mlff.predict(conf)
            Total energy of the configuration  -345.974  eV.
            
        Output may vary
        '''
        desc = get_Descriptor(self.settings,self.glob,conf,self.h)
        if self._fast:
            E, F = get_phi(self.settings,self.glob,desc,self.lrc,conf.maxtype,np.ones((desc.maxtype,1)),True)
        else :
            E, F = get_phi(self.settings,self.glob,desc,self.lrc,conf.maxtype,self.w,True)
        E = E + self.post_process_args[0]*conf.natom
        if write_output :
            print('Total energy of the configuration %9.3f eV.' % E)
        return E, F
    
    def next_TimeStep(self,conf,dt,mass):
        r'''
        Computes the next time step using the Verlet algoritm.

        Arguments
        ---------
        conf : Configuration
            An atomic configuration
        dt : float
            Timestep in femto seconds.
        mass : list
            Mass of the atomic species

        Returns
        -------
        None.

        '''
        mass = np.array(mass)
        if conf.atomvelocities == None :
            conf.atomvelocities = np.zeros((conf.natom,3))
        desc = get_Descriptor(self.settings,self.glob,conf,self.h)
        E, F = get_phi(self.settings,self.glob,desc,self.lrc,conf.maxtype,self.w,True)
        conf.atomvelocities += (dt/mass(conf.atomtype))[:,np.newaxis]*F
        conf.atompos += dt*conf.atomvelocities

@dataclass
class ML_TenField:
    r'''
    The machine learned force field.
    
    Arguments
    ---------
    errors : dict
        Information ont training and testing errors.
    settings : Setup
        Class containing all settings for predicting.
    glob : Globals
        Class containing all global variables.
    lrc : list
        The local refferenc configurations
    w : ndarray
        Vector containing the optimal weights :math:`\textbf{w}`
    post_process_args : tuple
        Arguments for post processing.
    h : CubicSpline
        Cubic spline of :math:`h_{nl}(r)`
    TenType : Tensor_Type
        
    '''
    errors : dict
    settings : Setup
    glob : Globals
    lrc : list
    w : np.ndarray
    post_process_args : tuple
    h : CubicSpline
    TenType : Tensor_Type = None
    _fast : bool = False
    _check : str = '*polipy4vasp~mlff*'
    
    
    def save(self,filename='ML_TEN'):
        r'''
        Saves the machine learnd tensor framework to a binary file.
        
        Arguments
        ---------
        filename : str, optional
            Name of the file where the machine learnd force field is stored
        '''
        with open(filename,'wb') as file:
            pickle.dump(self, file)
    
    def predict(self,conf,deriv=False):
        r'''
        Predicts the energy, forces, and stress tensor of a given configutation.
        
        Arguments
        ---------
        conf : Configuration
            An atomic configuration
        deriv : logical, optional
            Logical operator for derivatives, default = ``False``
        Returns
        -------
        (d)T : scalar
            Tenosor or its derivative
        '''
        
        if deriv:
            desc = get_TenDescriptor_grad(self.settings,self.glob,conf,self.h,self.TenType.lamb)
            if self._fast: dT = fast_prediction_grad(self.settings,self.glob,desc,self.lrc)
            else: dT = get_ten_dphi_predict(self.settings, self.glob, desc, self.lrc,self.w,self.TenType.lamb)
            return self.TenType.d_postprocess(dT,conf,self.post_process_args)
        else :
            desc = get_TenDescriptor(self.settings,self.glob,conf,self.h,self.TenType.lamb)
            if self._fast: T = fast_prediction(desc,self.lrc,self.TenType.summ)
            else: T = get_ten_phi_predict(self.settings, desc, self.lrc,self.w,self.TenType.lamb)
            return self.TenType.postprocess(T,conf,self.post_process_args)
       
        
def load_mlff(filename='ML_FF'):
    r'''
    Loads a machine learnd from a binary file generated by ``ML_ForceField.save()``.
    
    Arguments
    ---------
    filename : str, optional
        Name of the file where the machine learnd force field is stored
    
    Returns
    -------
    MLFF : ML_ForceField
        A machine learnd force field
    '''
    with open(filename,'rb') as file:
        buf = file.read()
    mlff = pickle.loads(buf)
    try:
        buf = mlff._check == '*polipy4vasp~mlff*'
    except:
        raise TypeError(filename + ' is no polipy mlff!')
    setup_asa(mlff.settings.Lmax)
    return mlff
