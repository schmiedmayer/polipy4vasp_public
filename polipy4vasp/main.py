# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate._cubic import CubicSpline
from dataclasses import dataclass, field
from time import time
import pickle
from copy import deepcopy

from .sph import setup_asa
from .TEN_ML_AB_reader import read_data
from .splines import get_splines
from .preprocess import pre_process, ten_pre_process, conver_set, AU_to_eV, AU_to_eVperAng, AU_to_Ang, calc_ionic_borncharges, calc_ionic_polarisation, polarisation_to_minimgcon, split_vali_Data
from .descriptors import get_AllDescriptors, get_Descriptor
from .reference import get_LRC
from .train import get_Y, get_w
from .desing import get_PHI, get_phi
from .globals import setup_globals, Globals
from .console_output import print_output
from .tensor_descriptors import get_ten_Y, get_ten_PHI, get_ten_phi_predict, get_ten_dphi_predict
from .make_linear_fast import make_fast, fast_prediction_P, fast_prediction_Z


@dataclass
class Setup:
    r'''
    Class for defining the input parameters used for training a MLFF.
    
    Arguments
    ---------
        Rcut : scalar, optional
            The cutoff radius :math:`R_\text{cut}`. Default 5 angstrom.
        SigmaAtom : scalar, optional
            Width of the gaussian atomic brodening function :math:`\sigma_\text{atom}`. Default 0.4 angstrom.
        Beta : list, optional
            List of length 2 containing the weighting parameters for the radial and angular descriptors respectively, like :math:`\left[\beta^{(1)},\beta^{(2)}\right]`. Default :math:`[0.2,0.8].`
        EpsCur : scalar, optional
            Threshold :math:`\epsilon_\text{cur}` for the CUR algorithm. Descriptors that produce eigenvalues larger than :math:`\epsilon_\text{cur}` are used as Local refferenc structures. Default 1e-10.
        Nmax : int, optional
            Maximal main quantum number :math:`n_\text{max}`. Default 8.
        Lmax : int, optional
            Maximal angular quantum number :math:`l_\text{max}`. Default 4.
        Kernel : str, optional
            Type of kernel used. Default "poli"
        Zeta : scalar, optional
            Power of the polinomial kernel :math:`\zeta`. Default 4.
        SigmaExp : scalar, optional
            :math:`\sigma` used for the gaussian kernel.
        Wene : scalar, optional
            Weight parameter for scaling the total energies for fitting. Default 1.
        Wforc : scalar, optional
            Weight parameter for scaling the forces for fitting. Default 1.
        Wstress : scalar, optional
            Weight parameter for scaling the stress tensors for fitting. Default 1.
        AlgoLRC : int, optional
            Integer for specifing the algorythm used for selecting local refferenc configuration. Default = 1.
        NLRC : list, optional
            Number of local refferenc configurations for each atomic species. Default = [15]
        lamb : int, optional
            Integer specifing the Tensorial kernal dimenstion :math:`(2\lambda+1)`.Default = ``None``
        Charges : list, optional
            Ionic charges in :math:`e` for each atomic species. Default = ``None``
        deriv : bool, optional
            ``True`` when learning Born effective charges.
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
    >>> settings = pp.Setup(Rcut = 4,
    ...                     Nmax = 6,
    ...                     Zeta = 3)
    
    '''
    Rcut: float = 5.
    SigmaAtom: float = 0.5
    Beta: list = field(default_factory=lambda:[0.2,0.8])
    EpsCur: float = 1e-10
    Nmax: int = 8
    Lmax: int = 4
    Kernel: str = "poli"
    Zeta: int = 4
    SigmaExp: float = 0.4
    Wene: float = 1
    Wforc: float = 1
    Wstress: float = 1
    AlgoLRC: int = 1
    NLRC : list = field(default_factory=lambda:[15])
    lamb: int = None
    Charges: list = None
    deriv: bool = True
    ncore: int = -1
    Scatter_Plot: bool = True
    Validation: float = 0

    def _check_type(self):
        r'''
        Used to check user input and give feedback if an error occures.
        '''
        if not isinstance(self.Rcut,float) and not isinstance(self.Rcut,int):
            raise TypeError('Rcut has wrong Type! Must be float (or int).')
        if not isinstance(self.SigmaAtom,float) and not isinstance(self.SigmaAtom,int):
            raise TypeError('SigmaAtom has wrong Type! Must be float (or int).')
        if not isinstance(self.Beta,list) :
            raise TypeError('Beta must be a list!')
        if len(self.Beta) != 2 :
            raise TypeError('Beta must be a list of len(2)!')
#        if sum(self.Beta) != 1 :
#            raise TypeError('sum(Beta) musst be 1!')
        for b in self.Beta:
            if not isinstance(b,float) and not isinstance(b,int):
                raise TypeError('Entries of Beta have wrong Type! Must be float (or int).')
        if not isinstance(self.EpsCur,float) and not isinstance(self.EpsCur,int):
            raise TypeError('EpsCur has wrong Type! Must be float (or int).')
        if not isinstance(self.Nmax,int) :
            raise TypeError('Nmax has wrong Type! Must be int!')
        if not isinstance(self.Lmax,int) :
            raise TypeError('Nmax has wrong Type! Must be int!')
        if not isinstance(self.Zeta,float) and not isinstance(self.Zeta,int):
            raise TypeError('Zeta has wrong Type! Must be int (or float).')
        if not isinstance(self.Wene,float) and not isinstance(self.Wene,int):
            raise TypeError('Wene has wrong Type! Must be float (or int).')
        if not isinstance(self.Wforc,float) and not isinstance(self.Wforc,int):
            raise TypeError('Wfroc has wrong Type! Must be float (or int).')
        if not isinstance(self.Wstress,float) and not isinstance(self.Wstress,int):
            raise TypeError('Wstress has wrong Type! Must be float (or int).')
            
            
    def train(self,filename='ML_AB',f2='M'): #remove f2
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
        do_both = False #remove
        self._check_type()
        conset = conver_set(self)
        if do_both : conset.NLRC = [int(i//2) for i in conset.NLRC] #remove
        begin = time()
        setup_asa(conset.Lmax)
        glob = setup_globals(conset)
        if type(filename) == str : Data = read_data(conset,filename)
        else : Data = deepcopy(filename)
        if do_both : Data2 = read_data(conset,f2) #remove
        print('Read input file. Dedected %4d training Configurations.' % Data.nconf)
        if do_both : print('Read input file2. Dedected %4d training Configurations.' % Data2.nconf) #remove
        if conset.lamb == None : shift_ene, shift_stress = pre_process(conset,Data)
        else : ten_pre_process(conset,Data)
        if conset.Validation > 0: Data, vali_Data = split_vali_Data(Data,conset.Validation)
        h = get_splines(conset)
        l_descriptors = get_AllDescriptors(conset,glob,Data.configurations,h)
        lrc, nlrc, _ = get_LRC(conset,l_descriptors,Data)
        if do_both : l_descriptors2 = get_AllDescriptors(conset,glob,Data2.configurations,h) #remove
        if do_both : lrc2, nlrc2 = get_LRC(conset,l_descriptors2,Data2) #remove
        if do_both : nlrc = [a + b for a, b in zip(nlrc,nlrc2)] #remove
        if do_both : lrc = [np.concatenate([a,b],1) for a, b in zip(lrc,lrc2)] #remove
        for J, typ in enumerate(Data.atomtypes):
            print(('Selected %4d local refferenc configurations for atom type '+typ+'.') % nlrc[J])
        if conset.lamb == None :
            Y = get_Y(conset,Data)
            PHI = get_PHI(conset,glob,l_descriptors,lrc,Data.maxtype)
        else :
            shift_ene, shift_stress = None, None
            if conset.lamb == 0 : 
                shift_ene, shift_stress = pre_process(conset,Data)
                Y = get_Y(conset,Data)
            else : Y, sig = get_ten_Y(conset,Data)
            PHI = get_ten_PHI(conset,glob,l_descriptors,lrc)
            if do_both : conset.deriv = False #remove
            if do_both : Y2, sig = get_ten_Y(conset,Data2) #remove
            if do_both : PHI2 = get_ten_PHI(conset,glob,l_descriptors2,lrc) #remove
            if do_both : Y = np.concatenate([10*Y2,Y],0) #remove
            if do_both : PHI = np.concatenate([10*PHI2,PHI],0) #remove

        w, singular, out = get_w(conset,PHI,Y,nlrc)
        if do_both : out = np.array(out)[:,len(Y2):] #remove
        print("Condition number of PHI : %8.1E" % (np.max(singular)/np.min(singular)))
        end = time()
        duration = end - begin
        if duration > 60 :
            print('Done gennerating MLFF in %2d min %4.1f sec.' % (int(duration // 60),duration % 60))
        else :
            print('Done generating MLFF in %4.1f sec.' % duration)
        print_output(conset,Data,out)
        if conset.Validation > 0:
            print('Validating...')
            vali_descriptors = get_AllDescriptors(conset,glob,vali_Data.configurations,h)
            vali_PHI = get_PHI(conset,glob,vali_descriptors,lrc,vali_Data.maxtype)
            vali_out = [vali_PHI @ np.hstack(w), get_Y(conset,vali_Data)]
            print_output(conset,vali_Data,vali_out)
        lrc,w = make_fast(conset,lrc,w)
        return ML_ForceField(info = self,
                             settings = conset,
                             glob = glob,
                             lrc =lrc,
                             w = w,
                             shift_ene = shift_ene,
                             shift_stress = shift_stress,
                             h = h)

        
@dataclass
class ML_ForceField:
    r'''
    The machine learned force field.
    
    Arguments
    ---------
    settings : Setup
        Class containing all the user defined settings for training the MLFF
    lrc : list
        The local refferenc configurations
    w : ndarray
        Vector containing the optimal weights :math:`\textbf{w}`
    shift_ene : scalar
        Shift of the total energy traning data
    shift_stress : scalar
        Shift of the stress tensor traning data
    h : CubicSpline
        Cubic spline of :math:`h_{nl}(r)`
    '''
    info : Setup
    settings : Setup
    glob : Globals
    lrc : list
    w : np.ndarray
    shift_ene : float
    shift_stress : float
    h : CubicSpline
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
    
    def predict(self,conf,deriv=False,write_output=True):
        r'''
        Predicts the energy, forces, and stress tensor of a given configutation.
        
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
            Total energy of configuration  -345.974  eV.
            
        Output may vary
        '''
        desc = get_Descriptor(self.settings,self.glob,conf,self.h)
        if self.settings.lamb == None :
            if self.w == 0:
                E, F = get_phi(self.settings,self.glob,desc,self.lrc,conf.maxtype,np.ones((desc.maxtype,1)),True)
            else :
                E, F = get_phi(self.settings,self.glob,desc,self.lrc,conf.maxtype,self.w,True)
            E = AU_to_eV((E + self.shift_ene)*conf.natom)
            if write_output :
                print('Total energy of configuration %9.3f eV.' % E)
            return E, AU_to_eVperAng(F)
        else :
            if self.w == 0:
                P = fast_prediction_P(desc,self.lrc)
            else:
                P = get_ten_phi_predict(self.settings, desc, self.lrc,self.w)
            P = AU_to_Ang(polarisation_to_minimgcon(np.roll(P,1) + calc_ionic_polarisation(conf,self.settings.Charges),conf))
#            if write_output :
#                print('Polarization of configuration (%9.3f, %9.3f, %9.3f) eÅ.' % P)
            if  deriv :
                if self.w == 0:
                    Z = np.roll(fast_prediction_Z(self.settings,self.glob,desc,self.lrc),shift=1,axis=2) + calc_ionic_borncharges(conf,self.settings.Charges)
                else :
                    Z = np.roll(get_ten_dphi_predict(self.settings, self.glob, desc, self.lrc,self.w),shift=1,axis=2) + calc_ionic_borncharges(conf,self.settings.Charges)
                return P, Z
            else :
                return P 
    
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
