 #SRMColliderMetabo
#Author: Premy Shanthamoorthy
#Feb 2019
#Last updated: Nov 2019

"""
Evaluating complex backgrounds that may cause ambiguities in the measurement
of metabolites. This tool first filters a list of identified metabolites to
remove chiral isomers (using the Inchikey). This filtered list is then used to
profile different methods for unique transitions as follows in the Q1 and Q3
phases with the mentioned filters to identify the number of unique ion
signatures (UIS) per mol_id.

Q1/Q3
MS1 -  0.7 Da / -      ; 20ppm / -
MRM -  0.7 Da / 0.7 Da ; 25 Da / 0.7 Da
SWATH - 25 Da / 20 ppm

This tool will also measure the number of interferences for each transition
(the number of identical transitions within the range of metabolites filtered
as specified above). 

"""
import pandas as pd
import numpy as np
import heapq
import rdkit
import difflib
import re
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pyper as pr
from math import sqrt
import scipy.spatial as sp
from statistics import mode
import ast

"""
function read: for NIST dataset
input: a list of compounds, a list of spectra
output: pandas dataframes (allcomp and spectra)
"""
def read(compounds, spectra):
    allcomp = pd.read_pickle(compounds) #63529
    spectra = pd.read_pickle(spectra) #574826
    cf = allcomp.dropna(subset = ['inchi', 'smiles', 'cas_num']) #reset index? #11928 --> after filtering 5550
    compounds_null = allcomp.loc[allcomp['inchi'].isnull()] #51601
    return cf, spectra

"""
function read2: for Pesticides dataset (assay library)
input: a list of compounds, a list of spectra
output: pandas dataframes (allcomp and spectra)
"""
def read2():
    pesticides = pd.read_csv("huge_assay_library_edit.csv")
    pesticides.columns = ['prec_mz', 'peaks']
    pesticides.peaks = pesticides.peaks.apply(ast.literal_eval)
    return(pesticides)

"""
function filter:
Filter the compound list based on the inchikey to take out chiral isomers but keep structural isomers. 

input: Original list of compounds 
output: Filtered compound list
"""
def filter2(compounds_filt):
    subtract = [] #ones that have chiral isomers that need to be removed #6378
    compounds_filt = compounds_filt.drop_duplicates(subset='inchi')
    
    for index, row in compounds_filt.iterrows():
        sample_set = compounds_filt.loc[compounds_filt['exact_mass'] == row['exact_mass']] # specific isomers of the compound being looked at
        sample_set.drop(index)
        inchi = row['inchi']
        matches = []

        if len(sample_set)>1: #there are isomers
            if '/m' in inchi:
                connectivity = ((re.search(('InChI(.*)/m'), inchi)).group(1))
            else:
                connectivity = ""
                                            
            for i2, isomer in sample_set.iterrows(): #check if stereo connectivity is the same, if so, a chiral isomer so can be removed
                other = isomer['inchi']
                if '/m' in other:
                    check = ((re.search(('InChI(.*)/m'), other)).group(1))
                else:
                    check = other
                if (check == connectivity) and (i2 not in subtract):
                    subtract.append(i2)
                    matches.append(other)

    for x in subtract:
        compounds_filt = compounds_filt.drop(x)
    return compounds_filt

"""
function choose_background_and_query_pesticides:
Per mol_id, choosing a specific query (based on given mol_id and parameters), followed by choosing the background
to compare this query to based on the Q1 and Q3 filters.

In this case, a combined background of the assay library and the NIST library is used. 

Input: spectra_filt (filtered spectra for all compounds), mol_id, Q1 parameters (change, ppm - if ppm is filled, that
        will take priority over change), query parameters (col_energy, col_gas, ion_mode, isnt_type, adduct), Q3
        parameters (if q3 = True, will take into account change_q3 or ppm_q3 parameters, otherwise only Q1), top_n
        (the top % of fragment ions to look at from which the most intense is chosen for the transition), UIS_num
        (n number of transitions chosen for evaluation))

Output: query (row), background_filt (background for query prec_mz), transitions_q1 (background for frag_mz),
        query_frag_mz_value (value of transition frag_mz), query_frag_mz (query fragment m/z with intensity)
"""
def choose_background_and_query_pesticides(spectra_filt, background2, mol_id, change=0.7, ppm=0, change_q3=0.7, ppm_q3 = 0, q3= False, top_n= 0.1, UIS_num = 1, plot_back =False):
    print("start")
    query = spectra_filt.loc[(spectra_filt.index == mol_id)]
    background_filt = spectra_filt.drop(query.index) #drop spectra from same mol_id

    background2 = background2[['prec_mz', 'peaks']]
    background_filt = pd.concat([background_filt, background2])
    
    if len(query) == 1:
        #choosing a transition (highest intensity for that query)
        query_prec_mz = float(query['prec_mz'].item())

        #choosing background
        if ppm != 0:
            change = ppm/1000000
       
        low = query_prec_mz - change/2
        high = query_prec_mz + change/2
        background_filt = background_filt.loc[background_filt['prec_mz'].between(low, high, inclusive = False)]

        #choosing the fragment
        query_frag_mz =  list(query['peaks'])[0]
        query_frag_mz.sort(key = lambda x: x[1], reverse = True)
        if query_frag_mz[0][0] != int(query_prec_mz):
            start = 0
        else:
            start = 1
        query_frag_mz = query_frag_mz[start:UIS_num]
        query_frag_mz_values = [query[0] for query in query_frag_mz]
        if q3 == True:
            for transition in query_frag_mz_values:
                if ppm_q3 != 0:
                    change_q3 = ppm_q3/1000000.0
                low = transition - change_q3/2.0
                high = transition + change_q3/2.0
                transitions_q1 = [[(a,b) for (a,b) in peaklist if a>low and a<high and (b>(1*top_n))] for peaklist in background_filt['peaks']] #do transitions here    
                transitions_q1 = [x for x in transitions_q1 if x!= []]
                transitions_q1 = list(itertools.chain.from_iterable(transitions_q1))
                transitions_q1.sort(key = lambda x: x[1], reverse = True)
                background_filt = background_filt.loc[(background_filt['peaks'].apply(lambda x: any(transition in x for transition in transitions_q1)))]

    else:
        query_frag_mz = 0
        query_frag_mz_values = 0

    if plot_back == True:
        ax = sns.distplot(list(background_filt['prec_mz']),bins = 20,kde = False)
        ax.grid()
        plt.show()

    print("done2")
    return query, background_filt, query_frag_mz_values, query_frag_mz  
    
"""
function choose_background_and_query:
Per mol_id, choosing a specific query (based on given mol_id and parameters), followed by choosing the background
to compare this query to based on the Q1 and Q3 filters.

Input: spectra_filt (filtered spectra for all compounds), mol_id, Q1 parameters (change, ppm - if ppm is filled, that
        will take priority over change), query parameters (col_energy, col_gas, ion_mode, isnt_type, adduct), Q3
        parameters (if q3 = True, will take into account change_q3 or ppm_q3 parameters, otherwise only Q1), top_n
        (the top % of fragment ions to look at from which the most intense is chosen for the transition), UIS_num
        (n number of transitions chosen for evaluation))

Output: query (row), background_filt (background for query prec_mz), transitions_q1 (background for frag_mz),
        query_frag_mz_value (value of transition frag_mz), query_frag_mz (query fragment m/z with intensity)
"""
def choose_background_and_query(spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 0, col_gas = '',
                      ion_mode = 'P',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+'], q3 = False, top_n = 0.1, plot_back = False, UIS_num = 2, choose = True):

    #choosing a query 
    query_opt = spectra_filt.loc[(spectra_filt['mol_id'] == mol_id)]
    if adduct != []:
        adduct = [str(x) for x in adduct]
        query_opt = query_opt.loc[query_opt['prec_type'].isin(adduct)]

    #drop all the same adduct from this list of spectra if true
    background_filt = spectra_filt.drop(query_opt.index) #drop spectra from same mol_id
        
    #analyte chosen is MS2 (keep MS3 and MS4 in background) 
    query_opt = query_opt.loc[query_opt['spec_type'] == 'MS2']
    
    if col_energy != 0:
        query_opt['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9]',value=r'')
        low = int(col_energy)-5
        high = int(col_energy)+5
        query_opt = query_opt.loc[pd.to_numeric(query_opt['col_energy']).between(low, high, inclusive = True)]
        background_filt['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9]',value=r'')
        background_filt = background_filt.loc[pd.to_numeric(background_filt['col_energy']).between(low, high, inclusive = True)] #collision energy filter 
    if col_gas != '':
        query_opt = query_opt.loc[query_opt['col_gas'] == str(col_gas)]
        background_filt = background_filt.loc[background_filt['col_gas'] == str(col_gas)]
    if ion_mode != '':
        query_opt = query_opt.loc[query_opt['ion_mode'] == str(ion_mode)]
        background_filt = background_filt.loc[background_filt['ion_mode'] == str(ion_mode)]
    if inst_type != '':
        query_opt = query_opt.loc[query_opt['inst_type'] == str(inst_type)]
        background_filt = background_filt.loc[background_filt['inst_type'] == str(inst_type)]
    
    if len(query_opt)>1:
        query = query_opt.sample(random_state = 9001)
    else:
        query = query_opt

        
    if len(query) == 1:
        #choosing a transition (highest intensity for that query)
        query_prec_mz = float(query['prec_mz'].item())

        #choosing background
        if ppm != 0:
            change = ppm/1000000
        low = query_prec_mz - change/2
        high = query_prec_mz + change/2
        background_filt = background_filt.loc[background_filt['prec_mz'].between(low, high, inclusive = False)]
            
        #choosing the fragment
        query_frag_mz =  list(query['peaks'])[0]
        query_frag_mz.sort(key = lambda x: x[1], reverse = True)
        if query_frag_mz[0][0] != int(query_prec_mz):
            start = 0
        else:
            start = 1
        query_frag_mz = query_frag_mz[start:UIS_num]
        query_frag_mz_values = [query[0] for query in query_frag_mz]
        
        if q3 == True:
            for transition in query_frag_mz_values:
                if ppm_q3 != 0:
                    change_q3 = ppm_q3/1000000.0
                low = transition - change_q3/2.0
                high = transition + change_q3/2.0
                transitions_q1 = [[(a,b) for (a,b) in peaklist if a>low and a<high and (b>(1000*top_n))] for peaklist in background_filt['peaks']] #do transitions here  
                transitions_q1 = [x for x in transitions_q1 if x!= []]
                transitions_q1 = list(itertools.chain.from_iterable(transitions_q1))
                transitions_q1.sort(key = lambda x: x[1], reverse = True)
                background_filt = background_filt.loc[(background_filt['peaks'].apply(lambda x: any(transition in x for transition in transitions_q1)))]
    else:
        query_frag_mz = 0
        query_frag_mz_values = 0

    if plot_back == True:
        ax = sns.distplot(list(background_filt['prec_mz']),bins = 20,kde = False)
        ax.grid()
        plt.show()

    return query, background_filt, query_frag_mz_values, query_frag_mz  

"""
function profile:
Based on the given parameters calculates the number of UIS and Interferences by mol_id.
Input: parameters for choose_background_and_query
Output: compounds list with added columns of 'UIS' and 'Interference'
"""
def profile(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '',
            ion_mode = '',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+'], q3 = True, top_n = 0.1,
            mol_id = 0, plot_back = False, UIS_num=2):
    UIS = []
    Interferences = []
    copy = compounds_filt

    for i, molecule in copy.iterrows():
        molid = molecule['mol_id']
        query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = molid, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                        col_energy = col_energy, col_gas = col_gas,
                                                        ion_mode = ion_mode, inst_type = inst_type,
                                                        adduct = adduct, q3 = q3, top_n = top_n, spectra_filt = spectra_filt, UIS_num=UIS_num)
        if len(query) != 0:
            interferences = len(np.unique(background.mol_id))
            if interferences == 0:
                unique = 1
            else:
                unique = 0
                
            UIS.append(unique)
            Interferences.append(interferences)
        else:
            UIS.append(-1) #if no query was found
            Interferences.append(-1)
    #Interferences = Interferences.drop_duplicates(['mol_id'])
    copy['UIS'] = UIS
    copy['Interferences'] = Interferences
    return copy

"""
function profile_pesticides:
Based on the given parameters calculates the number of UIS and Interferences by mol_id for the pesticides dataset.
Input: parameters for choose_background_and_query
Output: compounds list with added columns of 'UIS' and 'Interferences'
"""
def profile_pesticides(spectra_filt, background2, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, q3 = True, top_n = 0.1,
            mol_id = 0, plot_back = False, UIS_num=1):
    UIS = []
    Interferences = []
    copy = spectra_filt.copy()
    
    for i, molecule in copy.iterrows():
        molid = i
        query, background, frag_mz, frag_mz_int = choose_background_and_query_pesticides(mol_id = molid, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                                                        q3 = q3, top_n = top_n, spectra_filt = spectra_filt, background2 = background2, UIS_num=UIS_num)
        if len(query) != 0:
            interferences = len(np.unique(background.index))
            if interferences == 0:
                unique = 1
            else:
                unique = 0
                
            UIS.append(unique)
            Interferences.append(interferences)
        else:
            UIS.append(-1) #if no query was found
            Interferences.append(-1)
    #Interferences = Interferences.drop_duplicates(['mol_id'])
    copy['UIS'] = UIS
    copy['Interferences'] = Interferences
    print("done3")
    return copy


"""
function profile_specific:
Based on the given parameters calculates the number of UIS and Interferences for a specific mol_id that is provided as input.
Input: parameters for choose_background_and_query, a specific mol_id (mol_id), plot_back = True (will plot interferences if present as range of their prec_mz)
Output: 'interferences' (number of interferring compounds) and 'background' (compounds interfering for a specific mol_id, based on range of prec_mz) 
"""
def profile_specific(compounds_filt, spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '',
                     ion_mode = 'P',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+'], q3 = False, top_n = 0.1, UIS_num = 0):

    query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = mol_id, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                    col_energy = col_energy, col_gas = col_gas,
                                                    ion_mode = ion_mode, inst_type = inst_type,
                                                    adduct = adduct, q3 = q3, top_n = top_n, spectra_filt = spectra_filt, UIS_num=UIS_num)
    if len(query) != 0:
        interferences = len(np.unique(background.mol_id))
        if interferences == 0:
            unique = 1
        else:
            unique = 0
            
    else:
        interferences = -1
        unique = -1

    return interferences, background 

"""
function MS1profiler:
Profiles datasets according to MS1-specific windows (Q1 only)

Q1/Q3
MS1 -  0.7 Da / -      ; 20ppm / -

Input: parameters for choose_background_and_query (q3 stays False in this case, no q3 window is taken into account) 
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences and USI1 for all mol_id, based on range of prec_mz) 
"""
def MS1profiler(change = 0.7, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '', ion_mode = 'P',inst_type = 'HCD',
               adduct = ['[M+H]+', '[M+Na]+', '[M+H-H20]+'], q3 = False, top_n = 10, mol_id = 0, UIS_num=2):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    start = time.time()
    profiled, interferences = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, col_energy = 35, col_gas = '', ion_mode = '',inst_type = 'HCD',
                       adduct = adduct, q3 = False, top_n = 0, mol_id = mol_id, compounds_filt = compounds_filt,
                       spectra_filt = spectra_filt, UIS_num = UIS_num)
    print(profiled)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    print(profiled_filtered) 
    end = time.time()
    list_mol_ids = list(profiled_filtered.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown for MS1 profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['USI1'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))
    plot_interferences(x = list_mol_ids, y = Interferences['Interferences'], title = " MS1 Interferences vs. mol_id")
    plot_unique_identities(x = list_mol_ids, y = profiled_filtered['USI1'], title = 'MS1 USI1 vs. mol_id')
    return profiled

"""
function MRMprofiler:
Profiles datasets according to MRM-specific windows (Q1 and Q3)

Q1/Q3
MRM -  0.7 Da / 0.7 Da ; 25 Da / 0.7 Da

Input: parameters for choose_background_and_query (q3 stays True in this case, q3 window is taken into account) 
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences and USI1 for all mol_id, based on range of prec_mz) 
"""
def MRMprofiler(change = 0.7, ppm = 0, change_q3 = 0.7, ppm_q3 = 0, col_energy = 35, col_gas = '',
                ion_mode = 'P',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+', '[M+H-H20]+'], q3 = True, top_n = 0.1, mol_id = 0, UIS_num = 2):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    start = time.time()
    profiled, interferences = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, col_energy = 35, col_gas = '',
                       ion_mode = ion_mode, inst_type = 'HCD', adduct = adduct, q3 = True, top_n = 0.1, mol_id = mol_id,
                       compounds_filt = compounds_filt, spectra_filt = spectra_filt, UIS_num = UIS_num)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown for MRM profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['USI1'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))    
    plot_interferences(x = list_mol_ids, y = Interferences['Interferences'], title = " MS1 Interferences vs. mol_id")
    plot_unique_identities(x = list_mol_ids, y = profiled['USI1'], title = 'MRM USI1 vs. mol_id')
    return profiled

"""
function SWATHprofiler:
Profiles datasets according to SWATH-specific windows (Q1 and Q3)

Q1/Q3
SWATH - 25 Da / 20 ppm

Input: parameters for choose_background_and_query (q3 stays True in this case, q3 window is taken into account) 
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences and USI1 for all mol_id, based on range of prec_mz) 
"""
def SWATHprofiler(change = 25, ppm = 0, change_q3 = 0, ppm_q3 = 20, col_energy = 35, col_gas = '',
                ion_mode = 'P',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+', '[M+H-H20]+'], q3 = True, top_n = 0.1, mol_id = 0, UIS_num=2):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    start = time.time()
    
    profiled, interferences = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, col_energy = 35, col_gas = '',
                       ion_mode = ion_mode,inst_type = 'HCD', adduct = adduct, q3 = True, top_n = 0.1, mol_id = mol_id,
                       compounds_filt = compounds_filt, spectra_filt = spectra_filt, UIS_num = UIS_num)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown for SWATH profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['USI1'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))
    plot_interferences(x = list_mol_ids, y = Interferences['Interferences'], title = " MS1 Interferences vs. mol_id")
    plot_unique_identities(x = list_mol_ids, y = profiled['USI1'], title = 'SWATH USI1 vs. mol_id')
    return profiled


"""
function pesticides_profiler:
Profiles pesticides dataset according to varying windows (Q1 and Q3)

Different Q1/Q3 windows were tested (for different traditional metabolomics methods)

Input: parameters for choose_background_and_query (q3 stays True in this case, q3 window is taken into account) 
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences and USI1 for all mol_id, based on range of prec_mz) 
"""
def pesticides_profiler(change=25, ppm=0, change_q3 =0, ppm_q3=0, top_n=0.1, mol_id=0, UIS_num=1, q3=False):
    pesticides = read2()
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt_NIST = filter2(compounds_filt = allcomp)
    spectra_filt_NIST = spectra.loc[spectra['mol_id'].isin(list(compounds_filt_NIST.mol_id))]
    spectra_filt_NIST.peaks = [[(a,b/1000) for (a,b) in peaklist] for peaklist in spectra_filt_NIST['peaks']]
    
    start = time.time()
    profiled = profile_pesticides(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, q3 = q3, top_n = 0.1, mol_id = mol_id, spectra_filt = pesticides, background2 = spectra_filt_NIST, UIS_num = UIS_num)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled_filtered.index)
    print("The unique identities and interferences for all mol_id will now be shown for profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['UIS'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))
    return profiled


#Optimizing Collision Energy -------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
function: similarity_score
input: getting two mol_id and generating a pairwise table of similarity scores based on their collision energies.
output: final_scores (pairwise table of similarity scores), final_coll_diff (pairwise table of collision energy differences)
"""
def similarity_score(spectra_filt, query_molid = 3834, compared_molid = 4021, query_name = '1,8-Diaminonaphthalene', compare_name = '1,5-Diaminonaphthalene',t = 0.25, top_n=10, xlim = (50,1200), xthreshold = 0, print_graphic = True, print_alignment = True):
    scores = []
    coll_diff = []
    
    query_spectra = choose_background_and_query(spectra_filt = spectra_filt, mol_id=query_molid, choose = False, top_n = top_n)[0]
    compared_spectra = choose_background_and_query(spectra_filt = spectra_filt, mol_id=compared_molid, choose = False, top_n = top_n)[0]
    collision_energies_q = list(query_spectra['col_energy'])
    collision_energies_c = list(compared_spectra['col_energy'])

    for query_c in collision_energies_q:
        query_spectra2 = query_spectra.loc[query_spectra['col_energy'] == query_c]
        query = list(query_spectra2['peaks'])[0]
        query.sort(key = lambda  x: x[1], reverse = True)
        query_norm = set((a,((b/query[0][1])*100)) for (a,b) in query)
        query_norm = [(a,round(b)) for (a,b) in query_norm]
        query_df = pd.DataFrame(query_norm, columns = ['m/z', 'int'])
        query_score = []
        query_coll_diff = []
        
        for compare_c in collision_energies_c:          
            compared_spectra2 = compared_spectra.loc[compared_spectra['col_energy'] == compare_c]
            compare = list(compared_spectra2['peaks'])[0]
            compare.sort(key = lambda  x: x[1], reverse = True)
            compare_norm = set((a,((b/compare[0][1])*100)) for (a,b) in compare)
            compare_norm = [(a,round(b)) for (a,b) in compare_norm]
            compare_df = pd.DataFrame(compare_norm, columns = ['m/z', 'int'])

            #alignment
            aligned_df = pd.merge(query_df, compare_df, on='m/z', how = 'outer')
            aligned_df = aligned_df.drop_duplicates(subset = 'm/z', keep = False)
            aligned_df.fillna(0, inplace = True)
                        
            #score
            u = aligned_df['int_x'].values.reshape(1,-1)
            v = aligned_df['int_y'].values.reshape(1,-1)
            score = 1 - sp.distance.cdist(u, v, 'cosine')
            
            query_score.append(score[0][0])
            query_coll_diff.append(abs(int(compare_c)-int(query_c)))
            
        scores.append(query_score)
        coll_diff.append(query_coll_diff)

    final_scores = pd.DataFrame(scores)
    final_scores.columns = collision_energies_q
    final_scores.index = collision_energies_c

    final_coll_diff = pd.DataFrame(coll_diff)
    final_coll_diff.columns = [i for i in range(len(collision_energies_q))] #collision_energies_q
    final_coll_diff.index = [i for i in range(len(collision_energies_c))] #collision_energies_c
    
    return(final_scores, final_coll_diff)


""" --  testing
function: optimized_score

From the similarity score matrix and the CE diff matrix, determining what the optimized score would be by traceback based on:
1) primarily lowest CE diff
2) secondarily best similarity score
input:
output: 
"""
#should the start be from the 4 corners of the matrix ? only using one start right now 
def optimized_score(scored_matrix, diff_matrix):
    length = list(scored_matrix.columns)
    width = list(scored_matrix.index)

    minimum = mode(diff_matrix.min())
    columns_min = [diff_matrix.columns[i].tolist() for i in diff_matrix.values == minimum]
    scores_tot = []
    
    for j in range(len(width)):
        column_starts = columns_min[j] #values of i to start from
        for i in column_starts:
        #find min difference --> those are start points --> this will change i and j
            new_score = [(scored_matrix.loc[str(width[j]), str(length[i])]), str(width[j]), str(length[i])] #start off with first score & keep track of coll energies
            score = 1

            while (i<=len(length)) & (j<= len(width)) & (new_score != score): #for i in range(0, len(length)):
                score = new_score
                column = length[i]
                row = width[j]

                query_score = scored_matrix.loc[str(row), str(column)]
                query_diff = diff_matrix.loc[j, i]

                print(query_score)
                print(query_diff)

                if i <= int(length[len(length)//2]):
                    new_i = i+1
                else:
                    new_i = i-1
                if j <= int(length[len(width)//2]):
                    new_j = j+1
                else:
                    new_j = j-1
                    
                left_right = [scored_matrix.loc[str(row), str(length[new_i])], diff_matrix.loc[j, new_i]]
                diagonal = [scored_matrix.loc[str(width[new_j]), str(length[new_i])], diff_matrix.loc[new_j, new_i]]
                top_bottom = [scored_matrix.loc[str(width[new_j]), str(column)], diff_matrix.loc[new_j, i]]

                compare = [left_right, diagonal, top_bottom]
                print(compare)

                #prioritize 
                compare_f = [[a,b] for [a,b] in compare if (b<=(query_diff))]
                compare_f = [[a,b] for [a,b] in compare_f if (a<=(query_score))] #is diff <= the query diff AND score is smaller (more diff)
                
                if len(compare_f) > 0:
                    final = min(compare_f)
                    if compare.index(final) == 0:
                        i = i+1
                    elif compare.index(final) == 1:
                        i = i+1
                        j = j+1
                    elif compare.index(final) == 2:
                        j = j+1
                    w = width[i]
                    l = length[j]
                    new_score = [final[0], str(w), str(l)]

            scores_tot.append(score)
            print("final")
            print(scores_tot)

    #sets = [set(tuple(x) for x in y) for y in scores_tot]
    #final_score = set.intersection(*sets)

    final_score = [x for x in scores_tot if x[0] == mode([x[0] for x in scores_tot])][0] #issue of some being 28,21 not 25, 23 
    return(final_score)


""" --  testing
input:
output: 
"""
#should the start be from the 4 corners of the matrix ? only using one start right now 
def optimized_score2(scored_matrix, diff_matrix):
    length = list(scored_matrix.columns)
    width = list(scored_matrix.index)

    #find min difference --> those are start points --> this will change i and j
    i=0
    j=0
    
    new_score = [(scored_matrix.loc[str(width[j]), str(length[i])][0]), str(width[j]), str(length[i])]  #start off with first score & keep track of coll energies
    score = 1

    while (i<=len(length)) & (j<= len(width)) & (new_score != score): #for i in range(0, len(length)):
        score = new_score
        column = length[i]
        row = width[j]
        query = scored_matrix.loc[str(row), str(column)]
        query_diff = query[1]
        query_score = query[0]
                           
        left_right = scored_matrix.loc[str(row), str(length[i+1])]
        diagonal = scored_matrix.loc[str(width[j+1]), str(length[i+1])]
        top_bottom = scored_matrix.loc[str(width[j+1]), str(column)]

        compare = [left_right, diagonal, top_bottom]
        compare_f = [[a,b] for [a,b] in compare if (b<=(query[1]))]
        compare_f = [[a,b] for [a,b] in compare_f if (a<=(query[0]))] #is diff <= the query diff AND score is smaller (more diff)
        
        if len(compare_f) > 0:
            final = min(compare_f)
            if compare.index(final) == 0:
                i = i+1
            elif compare.index(final) == 1:
                i = i+1
                j = j+1
            elif compare.index(final) == 2:
                j = j+1
            w = width[i]
            l = length[j]
            new_score = [final[0], str(w), str(l)]
    return score


""" -- Need to test
function: collision_energy_optimizer

For each compound in the NIST library, is there a unique collision energy optimum that will allow for the greatest discrimination between that compound and its interferences
at a specific ppm wide window.

input: ppm width of window in MS1 to determine background 
output: dataframe with all compound with two new columns: "optimal collision energy" - optimized CE for greatest discriminatory power, and "Isotopes" number of interferences at that ppm window
"""
def collision_energy_optimizer(ppm = 20):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)

    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    spectra_filt['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9]',value=r'')
    spectra_filt = spectra_filt.loc[spectra_filt['col_energy'] != '']

    collision_energy = []
    interference_num = []
    copy = compounds_filt
    
    for i, molecule in copy.iterrows():
        molid = molecule['mol_id']
        query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = molid, ppm = ppm, q3 = False, spectra_filt = spectra_filt)

        background = background.append(query)
        collision_opt = []
        interference_num.append(len(background)-1)

        if len(background)>1:
            for j, isotope in background.iterrows():
                query_molid = isotope['mol_id']
                background_compare = background.drop(j)
                for c, compare in background_compare.iterrows():
                    compared_molid = compare['mol_id']
                    score_matrix, diff_matrix = similarity_score(spectra_filt = spectra_filt, query_molid = query_molid, compared_molid = compared_molid, query_name = isotope['name'], compare_name = compare['name'], print_graphic = False, print_alignment = False)
                    score = optimized_score(score_matrix, diff_matrix)                    
                    collision_opt.append(score) 
        
        if len(collision_min) == 0:
            collision_energy.append(-1)
        else:          
            collision_energy.append(mode(collision_opt))

    copy['Optimal Collision Energy'] = collision_energy
    copy['Isotopes'] = interference_num
    return copy


#Optimizing Ion Mobility (CCS Compendium) -------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
function: get_isotopes
Get isotopes based on the molecular weight of a compound. 
input: molecular formula and molecular weight of compound of interest
output: a list of isotopes that will then be input into a similarity_score function
"""
def get_isotopes(molecular_weight, compounds_filt):
    isotopes = compounds_filt.loc[(compounds_filt['exact_mass'] <=molecular_weight)]# & (compounds_filt['formula'] == molecular_formula)]
    return isotopes


    #spectra_filt['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9]',value=r'')
    #spectra_filt = spectra_filt.loc[spectra_filt['col_energy'] != '']


"""
function: Isotopes
Get isotopes based on the molecular weight of a compound for all compounds in NIST library. 
input: ccs=True (if you want to compare ccs values between these isotopes based on the ccs compendium)
output: dataframe of all compounds with 'Isotopes' as number of isotopes and 'CCS_difference' as list of differences for CCS for all isotopes 
"""
def Isotopes(ccs=True):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]

    ccs_lib = pd.read_csv("UnifiedCCSCompendium_FullDataSet_2019-09-05.csv")
    adduct = [str(x) for x in ['[M+H]', '[M+Na]']]
    ccs_lib = ccs_lib.loc[ccs_lib['Ion.Species'].isin(adduct)]

    copy = compounds_filt
    copy2 = compounds_filt
    isotope_num = []
    ccs_diff = []
    no_ccs = []

    for i, molecule in copy.iterrows():
        molid = molecule['mol_id']
        query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = molid, ppm = 20, q3 = False, spectra_filt = spectra_filt)        
        molecular_weight = query['prec_mz']
        
        isotopes = get_isotopes(molecular_weight = molecular_weight, compounds_filt = compounds_filt)          
        isotope_num.append(len(isotopes)-1)
        
        if (ccs == True):
            query_opt = find_ccs(molecule, ccs_lib)
            
            if len(query_opt) >= 1:
                ccs1 = int(query_opt['CCS'])
                isotope = (isotopes.loc[isotopes.index != i])
                ccs_temp = []

                for i2, iso in isotope.iterrows():
                    #print(iso)
                    compare_opt = find_ccs(iso, ccs_lib)
                    ccs2 = int(compare_opt['CCS'])
                    
                    if len(compare_opt == 1):
                        ccs_temp.append(ccs2-ccs1)
                    else:
                        ccs_temp.append(-1)
                ccs_diff.append(ccs_temp)
            else:
                no_ccs.append(i)
                ccs_diff.append(-1)
        else:
            ccs_diff.append(-1)
                    
    for x in no_ccs:
        copy2 = copy2.drop(x)
        
    copy['Isotopes'] = isotope_num
    copy['CCS_difference'] = ccs_diff
    
    return copy

"""
function: find_ccs
Find ccs values in ccs_lib for a specific molecule
input: molecule, ccs_lib (ccs compendium is used as default) 
output: specific query within ccs_lib that is a match
"""
def find_ccs(molecule, ccs_lib):
    low = molecule['exact_mass'] - 1
    high = molecule['exact_mass'] +1
    query_opt = ccs_lib.loc[pd.to_numeric(ccs_lib['mz']).between(low, high, inclusive = True)]
    if len(query_opt)>1:
        query_opt = query_opt.loc[query_opt['Compound'] == molecule['name']]
        query_opt = query_opt.drop_duplicates()
        query_opt = query_opt.loc[query_opt['Charge'] == 1]
    return query_opt


"""
function: ccs
showing the power of ccs values by seeing the differences in ccs for all isotopes for each compound in NIST 
input: -
output: dataframe with ccs values for compounds 
"""
#want to show that even if we only look at a large q1 window ccs will be able to differentiate ?? - power of ccs
def ccs():
    copy = Isotopes()
    copy = copy.loc[copy['Isotopes'] == 1]
    ccs_lib = pd.read_csv("UnifiedCCSCompendium_FullDataSet_2019-09-05.csv")
    #compounds_filt = compounds_filt.round({'exact_mass': 4})
    
    ccs = []
    queries = 0

    for i, molecule in copy.iterrows():
        low = molecule['exact_mass'] - 0.5
        high = molecule['exact_mass'] +0.5
        query_opt = ccs_lib.loc[pd.to_numeric(ccs_lib['mz']).between(low, high, inclusive = True)]
        
        if len(query_opt)>1:
            query_opt = query_opt.loc[query_opt['Neutral.Formula'] == molecule['formula']]
        if len(query_opt) ==1:
            queries += 1
        else:
            ccs.append(i)
    for x in ccs:
        copy = copy.drop(x)
    return copy

#Visualization (graphs&plots) -------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
function compare_ppm_q1:
 For a given list of different  Q1 window sizes (listed in ppm, but can include window sizes in Da converted to ppm), a bar plot of interferences or UIS1 will be given
for a specific mol_id. From this, we can decipher the range within the most optimal Q1 window size falls based on the fewest interferences or greatest UIS1.
Input: mol_id, interferences or USI1 (boolean values, based on what plot you would like to obtain) 
Output: barplot (x = ppm list, y = number of interferences or UIS1 for the mol_id) 
"""
def compare_ppm_q1(mol_id):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    ppm_list = [25, 2.5, 0.7, 0.1, 0.00002]
    y = []
    interferences_iso = []

    for p in ppm_list:
        interferences, background = profile_specific(compounds_filt, spectra_filt, mol_id = mol_id, change = p , ppm = 0, change_q3 = 0, ppm_q3 = 0, q3 = False)
        y.append(interferences)

    ylabel = 'Average Interference'
    sns.set(style= "white")
    y = [int(x) for x in y]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)    
    ax.set_xlabel('ppm', fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.set_title(('Comparison at different ppm for mol_id: ' + str(mol_id)), fontsize = 16)
    sns.set_style("white", {'axes.grid' : False})
    splot = sns.barplot(x =ppm_list, y = y, palette = "rocket")

    for p in splot.patches:
        splot.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    ax.grid()
    plt.show()
    return ppm_list, y, background

"""
function compare_ppm_q3:
For a given list of different  Q3 window sizes (listed in ppm, but can include window sizes in Da converted to ppm), a bar plot of interferences or UIS1 will be given
for a specific mol_id. From this, we can decipher the range within the most optimal Q3 window size falls based on the fewest interferences or greatest UIS1.
Input: mol_id, interferences or USI1 (boolean values, based on what plot you would like to obtain) 
Output: barplot (x = ppm list, y = number of interferences or UIS1 for the mol_id) 
"""
def compare_ppm_q3(mol_id, UIS_num = 3):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    ppm_list = [25, 2.5, 0.7]
    y = []
    
    for p in ppm_list:
        interferences, background = profile_specific(compounds_filt, spectra_filt, mol_id = mol_id, change = 0, ppm = 20, change_q3 = p, ppm_q3 = 0, UIS_num = UIS_num, q3 = True)
        y.append(interferences)

    ylabel = 'Average Interference'
    sns.set(({'font.family': 'Helvetica'}))
    sns.set(style= "white")
    y = [int(x) for x in y]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('ppm', fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.set_title(('Comparison at different ppm for mol_id: ' + str(mol_id)), fontsize = 16)
    splot = sns.barplot(x =ppm_list, y = y, palette = "rocket")

##    for p in splot.patches:
##        splot.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    
    ax.grid()
    plt.show()
    return ppm_list, y, background

"""
function plot_interferences:
Plotting the interferences for all mol_id as a range based on prec_mz
Input: x and y values
Output: count plot (interferences for all mol_id, based on range of prec_mz)  
"""
def plot_interferences(y, xlabel = "Molecular ID", ylabel = "Interferences" , title = "Interferences vs. mol_id"):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.set_title(title, fontsize = 16)
    ax = sns.distplot(list(y),bins = 100,kde = False, hist_kws={'range': (min(list(y)), max(list(y)))})
    ax.grid()
    plt.show()     
 
"""
function plot_unique_identities:
Plotting the UIS1 for all mol_id as a range based on prec_mz
Input: x and y values
Output: count plot (UIS1 for all mol_id, based on range of prec_mz)  
"""
def plot_unique_identities(x, y, xlabel = "Molecular ID", ylabel = "USI1" , title = "USI1 vs. mol_id"):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12) 
    ax.set_title(title, fontsize = 16)
    ax = sns.distplot(list(y),bins = 20,kde = False, hist_kws={'range': (min(list(y)), max(list(y)))})
    ax.grid()
    plt.show()


"""
function: compare_UIS
Compare unique identifications for UIS1, UIS2 and UIS3. 
Input: d (dictionary of UIS1, UIS2 and UIS3)
Output: triple bar plot comparing unique identifications out of 100%
"""
def compare_UIS(d=0):
    d = { 'UIS1': [0,1,209,13,1,220], 'UIS2':[0,1,209,73,20,226], 'UIS3':[0,1,209,118,68,232]}
    df = pd.DataFrame(data=d)
    df = (df/239)*100
    print(df)
    df.index = ['MS1-25Da', 'MS1-0.7Da', 'MS1-20ppm', 'MRM-0.7/0.7Da','DIA-25/0.7Da', 'DIA-20ppm/0.7Da']#'SWATH-25Da/20ppm', 'SWATH-20ppm/20ppm']
    sns.set_palette("Purples", n_colors = 3)
    df.plot.bar()
    plt.show()


"""
function plot_spectrum:
Given a specific spectral id, plot the MS2 spectra
Input: spec_id (spectral id), mz and intensity
Output: line MS2 spectrum 
"""
def plot_spectrum(spec_id =0, mz=0, intens=0):
    if (mz==0 & intens==0):
        allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
        spec = spectra.loc[spectra['spectrum_id'] == spec_id]
        mz_int = list(spec['peaks'])[0]
        mz_int.sort(key = lambda x: x[1], reverse = True)
        mz = [peak[0] for peak in mz_int]
        intens = [peak[1] for peak in mz_int]
    fig, ax = plt.subplots(1,1)
    ax.stem(mz, intens, linefmt = 'purple', markerfmt =' ')
    ax.set_xlabel('m/z')
    ax.set_xlim(0,500)
    ax.set_ylabel('Relative Abundance')
    plt.show()
