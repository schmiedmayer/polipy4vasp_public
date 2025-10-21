#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
=======================================================
Reading "ML_AB" files (:mod:`polipy4vasp.ML_AB_reader`)
=======================================================

.. currentmodule:: polipy4vasp.ML_AB_reader
    
"""
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

from .sph import setup_asa
from .globals import setup_globals
from .splines import get_splines
from .descriptors import get_AllDescriptors
from .reference import get_LRC

@dataclass
class Configuration:
    r"""
    Dataclass containing all information to describe an atomic configuration.
    
    Args:
        natom (int) : Number of atoms
        lattice (ndarray) : Array containing the lattice vectors
        atompos (ndarray) : Array containing the coordinates for each atom
        atomtype(ndarray) : Array containing information on the atom type
        atomname (list) : List of strings containing the atom names
        maxtype (int) : Number of different atom species in this configuration
        sysname(str, optional): Name of the configuration
        atomvelocities(ndarray, optional) : Array containing the atomic velocities
    """
    natom   : int
    lattice : np.ndarray
    atompos : np.ndarray
    atomtype: np.ndarray
    atomname: list
    maxtype : int
    sysname : str = None
    atomvelocities: np.ndarray = None

    def info(self,write=True,si=True):
        r"""
        Prints the lattice infomation to screen. (Optional) the information is
        Returned.

        Arguments
        --------_
        write : bool, optional
            If False the information is returned instead of written to screen.
            The default is True.
        si : bool, optional
            Us of SI units or au. The default is True.

        Returns
        -------
        abs : tuple
            Absolute value of the lattice vectors (|a|, |b|, |c|).
        deg : tuple
            The lattice angles (α, β, γ).
        vol : float
            The Volume of the cell.

        """
        if si :
            a = self.lattice[0]
            b = self.lattice[1]
            c = self.lattice[2]
            unit = "Å"
        else:
            a = self.lattice[0] * 1.889726125
            b = self.lattice[1] * 1.889726125
            c = self.lattice[2] * 1.889726125
            unit = "a_0"
        
        absa = np.linalg.norm(a)
        absb = np.linalg.norm(b)
        absc = np.linalg.norm(c)
        
        alph = np.arccos(np.dot(b,c)/(absb*absc))*180/np.pi
        beta = np.arccos(np.dot(a,c)/(absa*absc))*180/np.pi
        gamm = np.arccos(np.dot(a,b)/(absa*absb))*180/np.pi
        
        vol  = np.dot(np.cross(a,b),c)
        
        if write :
            print(('a=(%10.3e, %10.3e, %10.3e) '+unit) % (a[0], a[1], a[2]))
            print(('b=(%10.3e, %10.3e, %10.3e) '+unit) % (b[0], b[1], b[2]))
            print(('c=(%10.3e, %10.3e, %10.3e) '+unit) % (c[0], c[1], c[2]))
            print('')
            print(('|a|=%7.3f '+unit+'   α=%5.1f°') % (absa, alph))
            print(('|b|=%7.3f '+unit+'   β=%5.1f°') % (absb, beta))
            print(('|c|=%7.3f '+unit+'   γ=%5.1f°') % (absc, gamm))
            print('')
            print(('V=%8.2f '+unit+'³') % vol)
        else :
            return (absa, absb, absc), (alph, beta, gamm), vol
        
        
        
    
    def write_to_file(self,filename="POSCAR"):
        r"""
        Write the atomc configuration to a file in the POSCAR format.

        Argument
        --------
        filename : str, optional
            File name to which the atomic possitions are written. The default is "POSCAR".
        """
        
        def form(value):
            return "  %12.8f"*3 % (value[0],value[1],value[2])
        
        latt  = [form(l) for l in self.lattice.tolist()]
        pos   = [form(p) for p in self.atompos.tolist()]
        atypeN = [np.sum(self.atomtype == J) for J in range(self.maxtype)] 
        aname = ""
        anum  = ""
        for n, an in zip(atypeN,self.atomname):
            n  = str(n)
            a  = 2
            if len(an) == 1 : a+=1
            if len(n)  >  2 : a+=len(n) -2
            if len(n)  == 1 : n=" "+n
            aname += " "*a + an
            anum  += "  " + n
        
        lines = [
            self.sysname,
            "     1.0",
            *latt,
            aname,
            anum,
            "Cartesian",
            *pos]
        with open(filename,'w') as out:
            out.write('\n'.join(lines))
        

@dataclass
class Training_Data:
    r"""
    Dataclass containing all vital information needed for training.
    
    Args:
        configurations (list) : List of :meth:`Configuration <polipy4vasp.ML_AB_reader.Configuration>`
        nconf (int) : Number of configurations
        matom (list): List of atom masses for different species
        maxtype (int) : Number of different atom species
        energies (ndarray) : Array containing energies for each configuration
        forces (ndarray) : Array containing forces for each configuration
        stresstensors (ndarray) : Array containing the stress tensors for each configuration
        atomname (list) : List of strings containing the atom names
        lrc (list) : Local refferenc configurations selected by vasp
    """
    configurations : list
    nconf : int
    matom : list
    maxtype : int
    energies: np.ndarray
    forces: np.ndarray
    stresstensors: np.ndarray
    atomname: list
    lrc : list
    _ten : bool = False
    
    def reduce(self,confs,copy=False):
        r"""
        Reduces the training data set. Local refference configurations are discarded.

        Parameters
        ----------
        confs : list of ndarray of int
            Index of selected training configurations.
        copy : bool, optional
            Returnes a copy if ``True``. The default is False.

        Returns
        -------
        ML_AB : Training_Data, optional
            Returns reduced training data set if ``copy=True``

        """
        if copy :
            ML_AB = deepcopy(self)
        else :
            ML_AB = self
        ML_AB.nconf = len(confs)
        ML_AB.configurations = [ML_AB.configurations[i] for i in confs]
        ML_AB.energies = np.array([ML_AB.energies[i] for i in confs])
        ML_AB.forces = [ML_AB.forces[i] for i in confs]
        ML_AB.stresstensors = np.array([ML_AB.stresstensors[i] for i in confs])
        ML_AB.lrc = [np.array([],dtype=np.int32) for _ in ML_AB.atomname]
        if copy :
            return ML_AB
    
    def select(self,settings):
        r'''
        Selects new local refference configurations as defined in Setup

        Parameters
        ----------
        settings : Setup
            Settings for selecting new local refference configurations.

        Returns
        -------
        None.

        '''
        conset = settings
        setup_asa(conset.Lmax)
        glob = setup_globals(conset)
        h = get_splines(conset)
        l_descriptors = get_AllDescriptors(conset,glob,self.configurations,h)
        _, nlrc, lrc = get_LRC(conset,l_descriptors,self)
        for J, typ in enumerate(self.atomname):
                print(('Selected %4d local refferenc configurations for atom type '+typ+'.') % nlrc[J])
        self.lrc = lrc
    
    def write(self,path='ML_AB_new'):
        r'''
        Wirtes :meth:`Training_Data <polipy4vasp.ML_AB_reader.Training_Data>` to ML_AB file.

        Parameters
        ----------
        path : str, optional
            Path where to write the ML_AB file. The default is 'ML_AB_new'.

        Returns
        -------
        None

        '''
        def w_3(l,fmt,sep=" "):
            a = (len(l) - 1) // 3
            b = len(l) % 3
            out = ""
            for _ in range(a):
                out += "     "+fmt+sep+fmt+sep+fmt+"\n"
            out += "     "+fmt
            for _ in range(b-1):
                out += sep+fmt
            out += "\n"
            return out % (*l,)
            
        def lrc_pp_to_vasp(ML_AB):
            lrc_list = [[] for _ in range(ML_AB.maxtype)]
            
            for n, conf in enumerate(ML_AB.configurations,start=1):
                Naprev = 1
                for J, aname in enumerate(conf.atomname):
                    JJ = np.argwhere(ML_AB.atomname == aname)[0,0]
                    ntype = np.sum(conf.atomtype == J)
                    l2 = np.arange(ntype,dtype=np.int32) + Naprev
                    l1 = np.ones(ntype,dtype=np.int32) * n
                    l  = np.vstack([l1,l2]).T
                    lrc_list[JJ].append(l)
                    Naprev += ntype
            
            return [np.vstack(l) for l in lrc_list]
                    
        vasp_lrc_list = lrc_pp_to_vasp(self)
        
        with open(path,'wt') as f:
            f.write(" 1.0 Version\n")
            f.write("**************************************************\n")
            f.write("     The number of configurations\n")
            f.write("--------------------------------------------------\n")
            f.write("     %6i\n" % self.nconf)
            f.write("**************************************************\n")
            f.write("     The maximum number of atom type\n")
            f.write("--------------------------------------------------\n")
            f.write("     %3i\n" % self.maxtype)
            f.write("**************************************************\n")
            f.write("     The atom types in the data file\n")
            f.write("--------------------------------------------------\n")
            f.write(w_3(self.atomname,"%2s"))
            f.write("**************************************************\n")
            f.write("     The maximum number of atoms per system\n")
            f.write("--------------------------------------------------\n")
            f.write("     %10i\n" % max([conf.natom for conf in self.configurations]))
            f.write("**************************************************\n")
            f.write("     The maximum number of atoms per atom type\n")
            f.write("--------------------------------------------------\n")
            f.write("     %10i\n" % max([max([np.sum(conf.atomtype == J) for J in range(conf.maxtype)]) for conf in self.configurations]))
            f.write("**************************************************\n")
            f.write("     Reference atomic energy (eV)\n")
            f.write("--------------------------------------------------\n")
            f.write(w_3(self.maxtype*[0.0],"%4.2f"))
            f.write("**************************************************\n")
            f.write("     Atomic mass\n")
            f.write("--------------------------------------------------\n")
            f.write(w_3(self.matom,"%12.8f"))
            f.write("**************************************************\n")
            f.write("     The numbers of basis sets per atom type\n")
            f.write("--------------------------------------------------\n")
            f.write(w_3([len(lrc) for lrc in self.lrc],"%5i"))
            
            for J, atn in enumerate(self.atomname):
                f.write("**************************************************\n")
                f.write("     Basis set for %2s\n" % atn)
                f.write("--------------------------------------------------\n")
                np.savetxt(f, vasp_lrc_list[J][self.lrc[J]],fmt="%6i")
            
            
            for n, (conf, E, F, S) in enumerate(zip(self.configurations,self.energies,self.forces,self.stresstensors),start=1):
                f.write("**************************************************\n")
                f.write("     Configuration num. %6i\n" % n)
                f.write("==================================================\n")
                f.write("     System name\n")
                f.write("--------------------------------------------------\n")
                f.write("     "+conf.sysname+"\n")
                f.write("==================================================\n")
                f.write("     The number of atom types\n")
                f.write("--------------------------------------------------\n")
                f.write("     %3i\n" % conf.maxtype)
                f.write("==================================================\n")
                f.write("     The number of atoms\n")
                f.write("--------------------------------------------------\n")
                f.write("     %10i\n" % conf.natom)
                f.write("**************************************************\n")
                f.write("     Atom types and atom numbers\n")
                f.write("--------------------------------------------------\n")
                for J in range(conf.maxtype):
                    natomt = np.sum(conf.atomtype == J)
                    f.write("     %2s %6i\n" % (conf.atomname[J],natomt))
                f.write("==================================================\n")
                f.write("     CTIFOR\n")
                f.write("--------------------------------------------------\n")
                f.write("     %12.8f\n" % 3e-2)
                f.write("==================================================\n")
                f.write("     Primitive lattice vectors (ang.)\n")
                f.write("--------------------------------------------------\n")
                np.savetxt(f, conf.lattice, fmt=' %12.8f',delimiter='')
                f.write("==================================================\n")
                f.write("     Atomic positions (ang.)\n")
                f.write("--------------------------------------------------\n")
                np.savetxt(f, conf.atompos, fmt=' %12.8f',delimiter='')
                f.write("==================================================\n")
                f.write("     Total energy (eV)\n")
                f.write("--------------------------------------------------\n")
                f.write("     %12.8f\n" % E)
                f.write("==================================================\n")
                f.write("     Forces (eV ang.^-1)\n")
                f.write("--------------------------------------------------\n")
                np.savetxt(f, F, fmt=' %12.8f',delimiter='')
                f.write("==================================================\n")
                f.write("     Stress (kbar)\n")
                f.write("--------------------------------------------------\n")
                f.write("     XX YY ZZ\n")
                f.write("--------------------------------------------------\n")
                f.write(" %12.8f %12.8f %12.8f\n" % (*S[0],))
                f.write("--------------------------------------------------\n")
                f.write("     XY YZ ZX\n")
                f.write("--------------------------------------------------\n")
                f.write(" %12.8f %12.8f %12.8f\n" % (*S[1],))
        

def read_ML_AB(filename='ML_AB'):
    r"""
    This routine reads the VASP ML_AB file and returns the training data.
    
    Arguments
    ---------
    filename : str, optional
        Name of the ML_AB file. Default = 'ML_AB' 
        
    Returns
    -------
    data : Training_Data
        Contains all vital information from the ML_AB file
    """
    file = open(filename)
    line = []
    for l in file:
        line.append(l.replace('\n',''))
    Version = line[0].replace('Version','').strip()
    if Version != '1.0' :
        raise AttributeError('Wrong Version of the ML_AB file')
    MaxConf = int(line[4])
    MaxType = int(line[8])
    AtomTypes = np.array(line[12].strip().split())
#    MaxAtomPerSys = int(line[16])
    MaxAtomPerType = [int(i) for i in line[20].strip().split()]
    MaxAtomPerTypeSum = [sum(MaxAtomPerType[:i]) for i in range(len(MaxAtomPerType))]
#    RefEne = [float64(f) for f in line[24].strip().split()]
    AtomM = [np.float64(f) for f in line[28].strip().split()]
    NBasis = [int(i) for i in line[32].strip().split()]
    
    
    line = line[33:]
    Basis = []
    for J in range(MaxType):
        line = line[3:]
        Basis.append([[int(i) - 1 for i in line[k].strip().split()] for k in range(NBasis[J])])
        line = line[NBasis[J]:]
    #line = line[33+3*len(NBasis)+sum(NBasis):]
    k=0
    conf = []
    TotEneL = []
    ForcL = []
    StressL = []
    NTypeL = []
    for _ in range(MaxConf):
        SysName = line[k+5].strip()
        ConfMaxType = int(line[k+9])
        Natom = int(line[k+13])
        AtomType = []
        AtomName = []
        buf_NType = [0]*MaxType
        for j in range(ConfMaxType):
            buf = line[k+17+j].strip().split()
            NAtomType = np.squeeze(np.argwhere(AtomTypes == buf[0]))
            NType = int(buf[1])
            AtomType.append(np.ones(NType,dtype=np.int32)*NAtomType)
            AtomName.append(buf[0])
            buf_NType[NAtomType] = NType
        AtomType = np.concatenate(AtomType,axis = 0)
        NTypeL.append(buf_NType)
        if 'CTIFOR' == line[k+18+ConfMaxType].strip().split()[0]:
            k += 4
        Lattice = np.array([[np.float64(f) for f in line[k+20+ConfMaxType+j].strip().split()] for j in range(3)])
        AtomPos = np.array([[np.float64(f) for f in line[k+26+ConfMaxType+j].strip().split()] for j in range(Natom)])
        TotEneL.append(np.float64(line[k+29+ConfMaxType+Natom]))
        Forc = [[np.float64(f) for f in line[k+33+ConfMaxType+Natom+j].strip().split()] for j in range(Natom)]
        ForcL.append(np.array(Forc))
        Stress = [[np.float64(f) for f in line[k+38+ConfMaxType+2*Natom].strip().split()]]
        Stress.append([np.float64(f) for f in line[k+42+ConfMaxType+2*Natom].strip().split()])
        StressL.append(np.array(Stress))
        conf.append(Configuration(natom = Natom,
                                  lattice = Lattice,
                                  atompos = AtomPos,
                                  atomtype = AtomType,
                                  atomname = AtomName,
                                  maxtype = ConfMaxType,
                                  sysname = SysName))
        k+=43+ConfMaxType+2*Natom
    NTypeLsum = np.array([[sum(l[:i]) for i in range(len(l))] for l in NTypeL]).T
    NTypeL = np.array(NTypeL).T
    lrc = [[sum(NTypeL[J,:b[0]])+b[1] - NTypeLsum[J,b[0]] for b in B] for J, B in enumerate(Basis)]
    return Training_Data(
            configurations = conf,
            nconf = MaxConf,
            matom = AtomM,
            maxtype = MaxType,
            energies = np.array(TotEneL),
            forces = [F for F in ForcL],
            stresstensors = np.array(StressL),
            atomname = AtomTypes,
            lrc = lrc)
