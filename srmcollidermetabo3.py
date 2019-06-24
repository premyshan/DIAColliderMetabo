 #SRMColliderMetabo
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


"""
function read:
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
function filter:
Filter the compound list based on the inchikey to take out chiral isomers but keep structural isomers. 

input: Original list of compounds 
output: Filtered compound list
"""
def filter2(compounds_filt):
    subtract = [] #ones that have chiral isomers that need to be removed #6378
    lists = [] #list of chiral isomers per each compound
    matches = []
    
    for index, row in compounds_filt.iterrows():
        sample_set = compounds_filt.loc[compounds_filt['formula'] == row['formula']] # specific isomers of the compound being looked at
        sample_set.drop(index)
        inchi = row['inchi']
        matches = []
        if len(sample_set)>1: #there are isomers
            if '/b' in inchi:
                connectivity = ((re.search(('InChI(.*)/b'), inchi)).group(1))
            elif '/t' in inchi:
                connectivity = ((re.search(('InChI(.*)/t'), inchi)).group(1))
            elif '/f' in inchi:
                connectivity = ((re.search(('InChI(.*)/f'), inchi)).group(1))
            elif '/s' in inchi:
                connectivity = ((re.search(('InChI(.*)/s'), inchi)).group(1))
            else:
                connectivity = inchi
                                            
            for i2, isomer in sample_set.iterrows(): #check if structural connectivity is the same, if so, a chiral isomer so can be removed
                other = isomer['inchi']
                if '/b' in other:
                    check = ((re.search(('InChI(.*)/b'), other)).group(1))
                elif '/t' in other:
                     check = ((re.search(('InChI(.*)/t'), other)).group(1))
                elif '/f' in other:
                    check = ((re.search(('InChI(.*)/f'), other)).group(1))
                elif '/s' in other:
                    check = ((re.search(('InChI(.*)/s'), other)).group(1))
                else:
                    check = other
                              
                if (check == connectivity) and (i2 not in subtract): 
                    subtract.append(i2)
                    matches.append(other)

    for x in subtract:
        compounds_filt = compounds_filt.drop(x)
    return compounds_filt

"""
function choose_background_and_query:
Per mol_id, choosing a specific query (based on given mol_id and parameters), followed by choosing the background
to compare this query to based on the Q1 and Q3 filters.

Input: mol_id, Q1 parameters (change, ppm - if ppm is filled, that will take priority over change), query parameters
        (col_energy, col_gas, ion_mode, isnt_type, adduct), Q3 parameters (if q3 = True, will take into account
        change_q3 or ppm_q3 parameters, otherwise only Q1), top_n (the top % of fragment ions to look at from which
        the most intense is chosen for the transition)

Output: query (row), background_filt (background for query prec_mz), transitions_q1 (background for frag_mz),
        query_frag_mz_value (value of transition frag_mz), query_frag_mz (query fragment m/z with intensity)
"""
#input ppm, or else change of 0.7 is the default 
def choose_background_and_query(spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '',
                      ion_mode = '',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+'], q3 = True, top_n = 0.1, plot_back = False):

    #choosing a query 
    query_opt = spectra_filt.loc[(spectra_filt['mol_id'] == mol_id)]
    if adduct != []:
        adduct = [str(x) for x in adduct]
        query_opt = query_opt.loc[query_opt['prec_type'].isin(adduct)]

    #drop all the same adduct from this list of spectra 
    background_filt = spectra_filt.drop(query_opt.index) #drop spectra from same mol_id

    #analyte chosen is MS2 (keep MS3 and MS4 in background) 
    query_opt = query_opt.loc[query_opt['spec_type'] == 'MS2']
    
    if col_energy != 0:
        query_opt['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9]',value=r'')
        low = int(col_energy)-5
        high = int(col_energy)+5
        query_opt = query_opt.loc[pd.to_numeric(query_opt['col_energy']).between(low, high, inclusive = True)]
    if col_gas != '':
        query_opt = query_opt.loc[query_opt['col_gas'] == str(col_gas)]
    if ion_mode != '':
        query_opt = query_opt.loc[query_opt['ion_mode'] == str(ion_mode)]
    if inst_type != '':
        query_opt = query_opt.loc[query_opt['inst_type'] == str(inst_type)] 

    background_filt['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9]',value=r'')
    background_filt = background_filt.loc[pd.to_numeric(background_filt['col_energy']).between(low, high, inclusive = True)] #collision energy filter 
    
    #print(query_opt)
    if len(query_opt)>1:
        query = query_opt.sample(random_state = 9001)
    else:
        query = query_opt
    
    if len(query) != 0:
        #choosing a transition (highest intensity for that query)
        query_prec_mz = float(query['prec_mz'].item())

        #choosing background
        if ppm != 0:
            change = query_prec_mz*(ppm/1000000)
        low = query_prec_mz - change
        high = query_prec_mz + change
        background_filt = background_filt.loc[background_filt['prec_mz'].between(low, high, inclusive = False)]
        
        #choosing the fragment
        query_frag_mz =  list(query['peaks'])[0]
        query_frag_mz.sort(key = lambda x: x[0], reverse = True)
        query_frag_mz = query_frag_mz[0]
        query_frag_mz_value = float(query_frag_mz[1])

        if q3 == True:
            if ppm_q3 != 0:
                change_q3 = query_frag_mz_value*(ppm_q3/1000000)
            low = query_frag_mz_value - change_q3
            high = query_frag_mz_value + change_q3

            transitions_q1 = [[(a,b) for (a,b) in x if b>low and b<high] for x in background_filt['peaks']]
            transitions_q1 = [x for x in transitions_q1 if x!= []]
            transitions_q1 = list(itertools.chain.from_iterable(transitions_q1))
            transitions_q1.sort(key = lambda x: x[0], reverse = True)

            if len(transitions_q1)!=0: #there are interferences with chosen transition (Q3)
                background_filt = background_filt.loc[(background_filt['peaks'].apply(lambda x: any(transition in x for transition in transitions_q1)))]

                #getting top 10% of the fragment ions 
                if (top_n != 0) & (len(transitions_q1) != 0):
                    greatest = transitions_q1[0][0]
                    transitions_q1 = [(a,b) for (a,b) in transitions_q1 if float(a/greatest) > top_n]
    else:
        query_frag_mz = 0
        query_frag_mz_value = 0
        #background_filt = list(background_filt['prec_mz'])

    if plot_back == True:
        ax = sns.distplot(list(background_filt['prec_mz']),bins = 20,kde = False)
        ax.grid()
        plt.show()

    return query, background_filt, query_frag_mz_value, query_frag_mz     


"""
function profile:
Based on the given parameters calculates the number of USI and Interferences by mol_id.
Input: parameters for choose_background_and_query
Output: compounds list with added columns of 'USI1' and 'Average Interference'
"""
def profile(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '',
            ion_mode = '',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+'], q3 = True, top_n = 0.1,
            mol_id = 0, plot_back = False):
    USI1 = []
    Interferences = []
    copy = compounds_filt

    for i, molecule in copy.iterrows():
        molid = molecule['mol_id']
        query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = molid, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                        col_energy = col_energy, col_gas = col_gas,
                                                        ion_mode = ion_mode, inst_type = inst_type,
                                                        adduct = adduct, q3 = q3, top_n = top_n, spectra_filt = spectra_filt)
        if len(query) != 0:
            interferences = len(background)
            if interferences == 0:
                unique = 1
            else:
                unique = 0
                
            USI1.append(unique)
            Interferences.append(interferences)
        else:
            USI1.append(-1) #if no query was found
            Interferences.append(-1)
            
    copy['USI1'] = USI1
    copy['Interferences'] = Interferences
    return copy

"""
function profile_specific:
Based on the given parameters calculates the number of USI and Interferences for a specific mol_id that is provided as input.
Input: parameters for choose_background_and_query, a specific mol_id (mol_id), plot_back = True (will plot interferences if present as range of their prec_mz)
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences for a specific mol_id, based on range of prec_mz) 
"""
def profile_specific(compounds_filt, mol_id, spectra_filt, change = 0.7, ppm = 0, change_q3 = 0.7, ppm_q3 = 0, col_energy = 35, col_gas = '',
                     ion_mode = '',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+'], q3 = True, top_n = 0.1):
    profiled = profile(compounds_filt = compounds_filt, spectra_filt = spectra_filt, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                       col_energy = col_energy, col_gas = col_gas, ion_mode = ion_mode, inst_type = inst_type, adduct = adduct, q3 = q3, top_n = top_n, plot_back = True)
    
    return profiled

"""
function MS1profiler:
Profiles datasets according to MS1-specific windows (Q1 only)

Q1/Q3
MS1 -  0.7 Da / -      ; 20ppm / -

Input: parameters for choose_background_and_query (q3 stays False in this case, no q3 window is taken into account) 
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences and USI1 for all mol_id, based on range of prec_mz) 
"""
#adduct = ['[M+H]+', '[M+Na]+', '[2M+H]+', '[M+K]+'] - first try
#adduct = [] - second try 

def MS1profiler(change = 0.7, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '', ion_mode = '',inst_type = 'HCD',
               adduct = [], q3 = False, top_n = 10, mol_id = 0):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    start = time.time()
    profiled = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, col_energy = 35, col_gas = '', ion_mode = '',inst_type = 'HCD',
                       adduct = adduct, q3 = False, top_n = 0, mol_id = mol_id, compounds_filt = compounds_filt,
                       spectra_filt = spectra_filt)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled_filtered.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown for MS1 profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['USI1'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))
    plot_interferences(x = list_mol_ids, y = profiled_filtered['Interferences'], title = " MS1 Interferences vs. mol_id")
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
                ion_mode = '',inst_type = 'HCD', adduct = [], q3 = True, top_n = 10, mol_id = 0):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    start = time.time()
    profiled = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, col_energy = 35, col_gas = '',
                       ion_mode = '',inst_type = 'HCD', adduct = adduct, q3 = True, top_n = 0.1, mol_id = mol_id,
                       compounds_filt = compounds_filt, spectra_filt = spectra_filt)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown for MRM profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['USI1'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))    
    plot_interferences(x = list_mol_ids, y = profiled['Interferences'], title = " MRM Interferences vs. mol_id")
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
                ion_mode = '',inst_type = 'HCD', adduct = [], q3 = True, top_n = 0.1, mol_id = 0):
    allcomp, spectra = read(compounds = 'df_comp.pkl', spectra = 'df_spec.pkl')
    compounds_filt = filter2(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    start = time.time()
    profiled = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, col_energy = 35, col_gas = '',
                       ion_mode = '',inst_type = 'HCD', adduct = adduct, q3 = True, top_n = 10, mol_id = mol_id,
                       compounds_filt = compounds_filt, spectra_filt = spectra_filt)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown for MS1 profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['USI1'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))
    plot_interferences(x = list_mol_ids, y = profiled['Interferences'], title = " SWATH Interferences vs. mol_id")
    plot_unique_identities(x = list_mol_ids, y = profiled['USI1'], title = 'SWATH USI1 vs. mol_id')
    return profiled

#Visualization (graphs & plots)

"""
function compare_ppm_q1:
For a given list of different  Q1 window sizes (listed in ppm, but can include window sizes in Da converted to ppm), a bar plot of interferences or UIS1 will be given
for a specific mol_id. From this, we can decipher the range within the most optimal Q1 window size falls based on the fewest interferences or greatest UIS1.
Input: mol_id, interferences or USI1 (boolean values, based on what plot you would like to obtain) 
Output: barplot (x = ppm list, y = number of interferences or UIS1 for the mol_id) 
"""
def compare_ppm_q1(mol_id, interferences = True, USI1 = False):
    ppm_list = [0.1, 0.5, 1, 5, 10, 15, 20, 100, 100000, 700000, 2500000, 25000000]
    y = []
    if interferences == True:
        for p in ppm_list:
            mol = MS1profiler(ppm = p, mol_id = mol_id)
            y.append(mol.loc[mol['mol_id'] == mol_id]['Average Interference'])
        ylabel = 'Average Interference'
    if USI1 == True:
        for p in ppm_list:
            mol = MS1profiler(ppm = p)
            y.append(mol.loc[mol['mol_id'] == mol_id]['USI1'])
        ylabel = 'USI1'
    print(y)
    y = [int(x) for x in y]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('ppm', fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.set_title(('Comparison at different ppm for mol_id: ' + str(mol_id)), fontsize = 16)
    ax = sns.barplot(x =ppm_list, y = y)
    ax.grid()
    plt.show()
    return ppm_list, y

"""
function compare_ppm_q3:
For a given list of different  Q3 window sizes (listed in ppm, but can include window sizes in Da converted to ppm), a bar plot of interferences or UIS1 will be given
for a specific mol_id. From this, we can decipher the range within the most optimal Q3 window size falls based on the fewest interferences or greatest UIS1.
Input: mol_id, interferences or USI1 (boolean values, based on what plot you would like to obtain) 
Output: barplot (x = ppm list, y = number of interferences or UIS1 for the mol_id) 
"""
def compare_ppm_q3(mol_id, interferences = True, USI1 = False):
    ppm_list = [0.1, 0.5, 1, 5, 10, 15, 20, 100, 100000, 700000, 2500000, 25000000]
    y = []
    if interferences == True:
        for p in ppm_list:
            mol = SWATHprofiler(ppm_q3 = p, mol_id = mol_id)
            y.append(mol.loc[mol['mol_id'] == mol_id]['Average Interference'])
        ylabel = 'Average Interference'
    if USI1 == True:
        for p in ppm_list:
            mol = SWATHprofiler(ppm = p)
            y.append(mol.loc[mol['mol_id'] == mol_id]['USI1'])
        ylabel = 'USI1'

    y = [int(x) for x in y]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('ppm', fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.set_title(('Comparison at different ppm for mol_id: ' + str(mol_id)), fontsize = 16)
    ax = sns.barplot(x =ppm_list, y = y)
    ax.grid()
    plt.show()
    return ppm_list, y

"""
function plot_interferences:
Plotting the interferences for all mol_id as a range based on prec_mz
Input: x and y values
Output: count plot (interferences for all mol_id, based on range of prec_mz)  
"""
def plot_interferences(x, y, xlabel = "Molecular ID", ylabel = "Interferences" , title = "Interferences vs. mol_id"):
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
