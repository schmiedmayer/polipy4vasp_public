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
from .preprocess import AU_to_eV, AU_to_eVperAng, AU_to_Ang

def print_EandF_output(settings,Data,con):
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

    Returns
    -------
    None.

    '''
    deltaE = con[0][:Data.nconf] - con[1][:Data.nconf]
    deltaF = con[0][Data.nconf:] - con[1][Data.nconf:]
    E_RMSE = AU_to_eV(1e3*np.sqrt(np.mean(deltaE**2))/settings.Wene)
    E_MAXE = AU_to_eV(1e3*np.max(np.abs(deltaE))/settings.Wene)
    F_RMSE = AU_to_eVperAng(np.sqrt(np.mean(deltaF**2))/settings.Wforc)
    F_MAXE = AU_to_eVperAng(np.max(np.abs(deltaF))/settings.Wforc)
    E_sigm = AU_to_eV(1e3*np.std(con[1][:Data.nconf])/settings.Wene)
    F_sigm = AU_to_eVperAng(np.std(con[1][Data.nconf:])/settings.Wforc)
    print(' RMS error in total energy = %5.3f meV/atom' % E_RMSE)
    print(' RMS error in forces       = %5.3f eV/Angstrom' % F_RMSE)
    print(' MAX error in total energy = %5.3f meV/atom' % E_MAXE)
    print(' Max error in forces       = %5.3f eV/Angstrom' % F_MAXE)
    print('RMSP error in total energy = %8.2e' % (E_RMSE / E_sigm))
    print('RMSP error in forces       = %8.2e' % (F_RMSE / F_sigm))
    if settings.Scatter_Plot :
        plt.scatter(AU_to_eVperAng(con[1][Data.nconf:]/settings.Wforc),AU_to_eVperAng(con[0][Data.nconf:]/settings.Wforc),0.2)
        plt.xlabel('Training Forces [eV/$\AA$]')
        plt.ylabel('Predicted Forces [eV/$\AA$]')
        x = np.linspace(1.05*AU_to_eVperAng(np.min(con[1][Data.nconf:])),1.05*AU_to_eVperAng(np.max(con[1][Data.nconf:])))/settings.Wforc
        plt.plot(x,x,c='r')
        plt.show()
    
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

def print_tenP_output(settings,Data,con):
    r'''
    Prints out the RMSE and a scatter plot for the polarisation training.
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
    
    con[0], con[1] = AU_to_Ang(con[0]), AU_to_Ang(con[1])
    delta = con[0]-con[1]
    error = 1e3*np.sqrt(np.mean(delta**2))
    maxer = 1e3*np.max(np.abs(delta))
    sigma = 1e3*np.std(con[1])
    print(' RMS error = %5.3f m|e|Å' % error)
    print(' MAX error = %5.3f m|e|Å' % maxer)
    print('RMSP error = %8.2e' % (error / sigma))
    if settings.Scatter_Plot :
        plt.xlabel('Training P [|e|$\AA$]')
        plt.ylabel('Predicted P [|e|$\AA$]')
        plt.scatter(con[1],con[0],0.2)
        x = np.linspace(1.05*np.min(con[1]),1.05*np.max(con[1]))
        plt.plot(x,x,c='r')
        plt.show()
    
def print_output(settings,Data,con):
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

    Returns
    -------
    None.

    '''
    if settings.lamb == None : print_EandF_output(settings,Data,con)
    elif settings.lamb == 0 : print_EandF_output(settings,Data,con)
    else :
        if settings.deriv : print_tenZ_output(settings,Data,con)
        else : print_tenP_output(settings,Data,con)
