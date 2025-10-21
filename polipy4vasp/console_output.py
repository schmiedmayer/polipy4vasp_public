#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
====================================================================
Console output and plot routines (:mod:`polipy4vasp.console_output`)
====================================================================

.. currentmodule:: polipy4vasp.console_output
    
"""

import numpy as np
import matplotlib.pyplot as plt

def print_EandF_output(settings,Data,con,pre_process_args):
    r'''
    Prints out the RMSE and a scatter plot for energie and force training.
    The scatter plot is optional and can be turned on by `Setup(Scatter_Plot=True)`.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    Data : Training_Data
        Contains all vital information from the ML_AB file.
    con : list
        List containing the predicton and the training data.
    pre_process_args : tupel
        Arguments needed for preprocessing

    Returns
    -------
    err : dict
        Dictionary containing Errors.

    '''
    deltaE = con[0][:Data.nconf] - con[1][:Data.nconf]
    deltaF = con[0][Data.nconf:] - con[1][Data.nconf:]
    E_RMSE = 1e3*np.sqrt(np.mean(deltaE**2))/pre_process_args[3]
    E_MAXE = 1e3*np.max(np.abs(deltaE))/pre_process_args[3]
    F_RMSE = np.sqrt(np.mean(deltaF**2))/pre_process_args[4]
    F_MAXE = np.max(np.abs(deltaF))/pre_process_args[4]
    E_sigm = 1e3*np.std(con[1][:Data.nconf])/pre_process_args[3]
    F_sigm = np.std(con[1][Data.nconf:])/pre_process_args[4]
    print(' RMS error in total energy = %5.3f meV/atom' % E_RMSE)
    print(' RMS error in forces       = %5.3f eV/Angstrom' % F_RMSE)
    print(' MAX error in total energy = %5.3f meV/atom' % E_MAXE)
    print(' Max error in forces       = %5.3f eV/Angstrom' % F_MAXE)
    print('RMSP error in total energy = %8.2e' % (E_RMSE / E_sigm))
    print('RMSP error in forces       = %8.2e' % (F_RMSE / F_sigm))
    if settings.Scatter_Plot :
        plt.scatter(con[1][Data.nconf:]/pre_process_args[4],con[0][Data.nconf:]/pre_process_args[4],0.2)
        plt.xlabel('Training Forces [eV/$\AA$]')
        plt.ylabel('Predicted Forces [eV/$\AA$]')
        x = np.linspace(1.05*np.min(con[1][Data.nconf:]),1.05*np.max(con[1][Data.nconf:]))/pre_process_args[4]
        plt.plot(x,x,c='r')
        plt.show()
    return {'E_RMSE' : E_RMSE,
            'E_MAXE' : E_MAXE,
            'E_sigm' : E_sigm,
            'F_RMSE' : F_RMSE,
            'F_MAXE' : F_MAXE,
            'F_sigm' : F_sigm}
    
def print_tenZ_output(settings,Data,con):
    r'''
    Prints out the RMSE and a scatter plot for the born effective charge training.
    The scatter plot is optional and can be turned on by `Setup(Scatter_Plot=True)`.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    Data : Training_Data
        Contains all vital information from the ML_AB file.
    con : list
        List containing the predicton and the training data.

    Returns
    -------
    None.

    '''
    delta = con[0]-con[1]
    error = 1e3*np.sqrt(np.mean(delta**2))
    maxer = 1e3*np.max(np.abs(delta))
    sigma = 1e3*np.std(con[1])
    print(' RMS error = %5.3f m|e|' % error)
    print(' MAX error = %5.3f m|e|' % maxer)
    print('RMSP error = %8.2e' % (error / sigma))
    if settings.Scatter_Plot :
        plt.scatter(con[1],con[0],0.2)
        mask = np.roll(np.diag((True,True,True)),shift=-1,axis=1)
        con1 = con[1].reshape(-1,3,3)[:,mask].reshape(-1)
        con2 = con[0].reshape(-1,3,3)[:,mask].reshape(-1)
        plt.xlabel('Training Z [|e|]')
        plt.ylabel('Predicted Z [|e|]')
        plt.scatter(con1,con2,0.2)
        x = np.linspace(1.05*np.min(con[1]),1.05*np.max(con[1]))
        plt.plot(x,x,c='r')
        plt.show()

def print_ten_output(settings,Data,con):
    r'''
    Prints out the RMSE and a scatter plot for the tensorial training.
    The scatter plot is optional and can be turned on by `Setup(Scatter_Plot=True)`.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    Data : Training_Data
        Contains all vital information from the ML_AB file.
    con : list
        List containing the predicton and the training data.

    Returns
    -------
    err : dict
        Dictionary containing Errors.

    '''
    def unit_prefix(value):
        #symbol = ['Q' ,'R' ,'Y' ,'Z' ,'E' ,'P' ,'T' ,'G','M','k','h','da','','d' ,'c' ,'m' ,'µ' ,'n' ,'p'  ,'f'  ,'a'  ,'z'  ,'y'  ,'r'  ,'q'  ]
        #power  = [1e30,1e27,1e24,1e21,1e18,1e15,1e12,1e9,1e6,1e3,1e2,1e1 ,1 ,1e-1,1e-2,1e-3,1e-6,1e-9,1e-12,1e-15,1e-18,1e-21,1e-24,1e-27,1e-30]
        symbol = ['Q' ,'R' ,'Y' ,'Z' ,'E' ,'P' ,'T' ,'G','M','k','','m' ,'µ' ,'n' ,'p'  ,'f'  ,'a'  ,'z'  ,'y'  ,'r'  ,'q'  ]
        power  = [1e30,1e27,1e24,1e21,1e18,1e15,1e12,1e9,1e6,1e3,1 ,1e-3,1e-6,1e-9,1e-12,1e-15,1e-18,1e-21,1e-24,1e-27,1e-30]
        n = 0
        while value/power[n] < 1 : n += 1
        return value/power[n], symbol[n]
    
    def print_err(c):
        delta = c[0]-c[1]
        rmser = np.sqrt(np.mean(delta**2))
        abser = np.mean(np.abs(delta))
        maxer = np.max(np.abs(delta))
        sigma = np.std(c[1])
        print(' RMS error = %7.3f %s%s'% (*unit_prefix(rmser), Data.Type.unit))
        print('MABS error = %7.3f %s%s'% (*unit_prefix(abser), Data.Type.unit))
        print(' MAX error = %7.3f %s%s'% (*unit_prefix(maxer), Data.Type.unit))
        print('RMSP error = %9.3e' % (rmser / sigma))
        return  {'rmse' : rmser,
                 'mabse': abser,
                 'maxe' : maxer,
                 'sig'  : sigma}
    
    def plot_finish(c):
        plt.xlabel('Training [%s]' % Data.Type.unit)
        plt.ylabel('Predicted [%s]' % Data.Type.unit)
        x = np.linspace(1.05*np.min(c[1]),1.05*np.max(c[1]))
        plt.plot(x,x,c='r')
        plt.show()
        
    def plot(c,atomtype=None):
        if type(atomtype) == str:
            plt.scatter(c[1],c[0],0.2,label=atomtype)
            plt.legend()
            np.save('Scatter_'+str(atomtype)+'.npy',c)
        else:
            plt.scatter(c[1],c[0],0.2)
            np.save('Scatter.npy',c)
    
    if Data.Type.summ :
        err = print_err(con)
        if settings.Scatter_Plot :
            plot(con)
            plot_finish(con)
    else:
        err = []
        for atomname, c in zip(Data.atomname,con):
            print('Errors for atom type %2s:' % atomname)
            print('=========================')
            err.append(print_err(c))
            print('-------------------------')
            if settings.Scatter_Plot : plot(c,atomname)
        
        if settings.Scatter_Plot : plot_finish(np.hstack(con))
        return err
        
    

def print_lrc(atomname,nlrc):
    r'''
    Nice print out of selected local refferenc configurations.

    Parameters
    ----------
    atomname : list
        List of atom names.
    nlrc : list
        Number of local refferenc configurations.

    Returns
    -------
    err : dict
        Dictionary containing Errors.

    '''
    for typ, n in zip(atomname,nlrc):
        print(('Selected %4d local reference configurations for atom type %2s.') % (n,typ))

def print_output(settings,Data,con,pre_process_args):
    r'''
    Prints out the RMSE and a scatter plot for the corrisponding training.
    The scatter plot is optional and can be turned on by `Setup(Scatter_Plot=True)`.

    Parameters
    ----------
    settings : Setup
        Class containing all the user defined settings for training the MLFF.
    Data : Training_Data
        Contains all vital information from the ML_AB file.
    con : list
        List containing the predicton and the training data.
    pre_process_args : tupel
        Arguments needed for preprocessing

    Returns
    -------
    None.

    '''
    if not Data._ten : err = print_EandF_output(settings,Data,con,pre_process_args)
    else : err = print_ten_output(settings,Data,con)
    return err


def print_time(duration):
    '''
    Nice format for time.

    Parameters
    ----------
    duration : scalar
        Time in seconds.

    Returns
    -------
    None.

    '''
    if duration > 3600 :
        h = int(duration // 3600)
        m = int((duration % 3600) // 60)
        s = (duration % 3600) % 60
        print('Finished gennerating MLFF in %2d h %2d min %4.1f sec.' % (h, m, s))
    elif duration > 60 :
        m = int(duration // 60)
        s = duration % 60
        print('Finished gennerating MLFF in %2d min %4.1f sec.' % (m, s))
    else :
        print('Finished generating MLFF in %4.1f sec.' % duration)
