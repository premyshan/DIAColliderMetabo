#DIAColliderMetabo 
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
from math import sqrt
import scipy.spatial as sp
from statistics import mode
import ast
from numpy import dot
from numpy.linalg import norm
from statistics import mode
import ast


"""
function read:
input: a list of compounds, a list of spectra
output: pandas dataframes (allcomp and spectra)
"""
def read(compounds, spectra):
    allcomp = pd.read_pickle(compounds) 
    spectra = pd.read_pickle(spectra) 

    cf = allcomp.dropna(subset = ['inchi', 'smiles', 'cas_num']) 
    compounds_null = allcomp.loc[allcomp['inchi'].isnull()] 
    return cf, spectra

"""
function filter:
Filter the compound list based on the inchikey to take out chiral isomers but keep structural isomers. 

input: Original list of compounds 
output: Filtered compound list
"""
def filtercomp(compounds_filt):
    subtract = [] #remove chiral isomers
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
def choose_background_and_query(spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 0, col_gas = '',
                      ion_mode = 'P',inst_type = ['IT/ion trap', 'Q-TOF', 'HCD', 'IT-FT/ion trap with FTMS'], adduct = ['[M+H]+', '[M+Na]+'], q3 = False, top_n = 0.1, plot_back = False, UIS_num = 2, choose = True):
    #choosing a query `
    query_opt = spectra_filt.loc[(spectra_filt['mol_id'] == mol_id)]
    if adduct != []:
        adduct = [str(x) for x in adduct]
        query_opt = query_opt.loc[query_opt['prec_type'].isin(adduct)]

    same = spectra_filt.loc[spectra_filt['mol_id']==mol_id]
    background_filt = spectra_filt.drop(same.index) #drop spectra from same mol_id
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
        inst_type = [str(x) for x in inst_type]
        query_opt = query_opt.loc[query_opt['inst_type'].isin(inst_type)]
        background_filt = background_filt.loc[background_filt['inst_type'].isin(inst_type)]
    
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

        #choosing fragments
        query_frag_mz =  list(query['peaks'])[0]
        query_frag_mz.sort(key = lambda x: x[1], reverse = True)
        if query_frag_mz[0][0] < int(query_prec_mz):
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
Based on the given parameters calculates the number of USI and Interferences by mol_id.
Input: parameters for choose_background_and_query
Output: compounds list with added columns of 'USI1' and 'Average Interference'
"""
def profile(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '',
            ion_mode = '',inst_type = ['IT/ion trap', 'Q-TOF', 'HCD', 'IT-FT/ion trap with FTMS'], adduct = ['[M+H]+', '[M+Na]+'], q3 = True, top_n = 0.1,
            mol_id = 0, plot_back = False, UIS_num=2):
    USI1 = []
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
                
            USI1.append(unique)
            Interferences.append(interferences)
        else:
            USI1.append(-1) #if no query was found
            Interferences.append(-1)
    copy['UIS'] = USI1
    copy['Interferences'] = Interferences
    return copy

"""
function profile_specific:
Based on the given parameters calculates the number of USI and Interferences for a specific mol_id that is provided as input.
Input: parameters for choose_background_and_query, a specific mol_id (mol_id), plot_back = True (will plot interferences if present as range of their prec_mz)
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences for a specific mol_id, based on range of prec_mz) 
"""
def profile_specific(compounds_filt, spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, col_energy = 35, col_gas = '',
                     ion_mode = 'P',inst_type = 'HCD', adduct = ['[M+H]+', '[M+Na]+'], q3 = False, top_n = 0.1, UIS_num = 0):

    query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = mol_id, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                    col_energy = col_energy, col_gas = col_gas,
                                                    ion_mode = ion_mode, inst_type = inst_type,
                                                    adduct = adduct, q3 = q3, top_n = top_n, spectra_filt = spectra_filt, UIS_num=UIS_num)
    print(query.index)
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
function MethodProfiler:
Profiles datasets according to MS1/MS2 windows set 

Q1/Q3
MS1 -  0.7 Da / -      ; 20ppm / -
MRM -  0.7 Da / 0.7 Da ; 25 Da / 0.7 Da
SWATH - 25 Da / 20 ppm


Input: filecompounds (csv file with compounds), filespectra(csv file with spectra), parameters for choose_background_and_query (q3 stays False if MS1-only) 
Output: compounds list with added columns of 'UIS' and 'Average Interference', count plot (interferences and USI1 for all mol_id, based on range of prec_mz) 
"""
def MethodProfiler(filecompounds, filespectra, change = 0.7, ppm = 0, change_q3 = 0.7, ppm_q3 = 0, col_energy = 35, col_gas = '',
                ion_mode = 'P',inst_type = ['IT/ion trap', 'Q-TOF', 'HCD', 'IT-FT/ion trap with FTMS'], adduct = ['[M+H]+', '[M+Na]+'], q3 = True, top_n = 0.1, mol_id = 0, UIS_num = 2):
    allcomp, spectra = read(compounds = dfcompounds, spectra = dfspectra)
    compounds_filt = filtercomp(compounds_filt = allcomp)
    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(list(spectra_filt.mol_id))]

    profiled = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, col_energy = col_energy, col_gas = col_gas,
                       ion_mode = ion_mode, inst_type = inst_type, adduct = adduct, q3 = a3, top_n = top_n, mol_id = mol_id, compounds_filt = compounds_filt, spectra_filt = spectra_filt, UIS_num = UIS_num)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    list_mol_ids = list(profiled_filtered.mol_id)

    print("The number of unique mol_id is: " + str(len([x for x in profiled['UIS'] if x == 1])))
    return profiled


#Collision Energy Optimization
"""
Determining the optimal collision energy based on the pairwise comparison of compounds in the dataframe with their associated inteferences.

collision_energy_optimizer():
Input: filecompounds (csv file with compounds), filespectra(csv file with spectra), parameters for choose_background_and_query (q3 stays False if MS1-only) 
Output: CEFinal (dataframe with optimized CE for each compound)

Uses the following:

similarity_score:
Input: getting two mol_id and generating a table of similarity scores based on their collision energies.
Output: table of similarity scores.

optimized_score:
Input: similarity_score matrix
Output: score from the matrix

"""
def similarity_score(query_spec, compared_spec, t = 0.25, top_n=10, xlim = (50,1200), xthreshold = 0, print_graphic = False, print_alignment = False):
    scores = []
    coll_diff = []
    both_all = []

    query_spec[["col_energy"]] = query_spec[["col_energy"]].apply(pd.to_numeric)
    query_spec = query_spec.sort_values(by=['col_energy'])
    compared_spec[["col_energy"]] = compared_spec[["col_energy"]].apply(pd.to_numeric)
    compared_spec = compared_spec.sort_values(by=['col_energy'])

    collision_energies_q = list(query_spec['col_energy'])
    collision_energies_c = list(compared_spec['col_energy'])

    for query_c in collision_energies_q:
        query_spectra2 = query_spec.loc[query_spec['col_energy'] == query_c]
        query = list(query_spectra2['peaks'])[0]
        query.sort(key = lambda  x: x[1], reverse = True)
        query_norm = set((a,((b/query[0][1])*100)) for (a,b) in query)
        query_norm = [(a,round(b)) for (a,b) in query_norm]
        query_df = pd.DataFrame(query_norm, columns = ['m/z', 'int'])
        query_score = []
        query_coll_diff = []
        both = []
        
        for compare_c in collision_energies_c:    
            compared_spectra2 = compared_spec.loc[compared_spec['col_energy'] == compare_c]
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
            score = cosine_vectorized(u,v)

            query_score.append(score[0][0])
            query_coll_diff.append(abs(int(compare_c)-int(query_c)))
            both.append([(abs(int(compare_c)-int(query_c))), score[0][0]])

        scores.append(query_score)
        coll_diff.append(query_coll_diff)
        both_all.append(both)
    
    all_df = pd.DataFrame(both_all)
    all_df.columns = collision_energies_c
    all_df.index = collision_energies_q
    return(all_df)


def cosine_vectorized(array1, array2):
    sumyy = (array2**2).sum(1)
    sumxx = (array1**2).sum(1, keepdims=1)
    sumxy = array1.dot(array2.T)
    return (sumxy/np.sqrt(sumxx))/np.sqrt(sumyy)


def collision_energy_optimizer():
    allcomp, spectra = read(compounds = 'dataframe_comp.pkl', spectra = 'highresdata.pkl')
    compounds_filt = filtercomp(compounds_filt = allcomp)

    spectra_filt = spectra.loc[spectra['mol_id'].isin(list(compounds_filt.mol_id))]
    spectra_filt = spectra_filt.loc[spectra_filt['ion_mode']=='P']
    spectra_filt['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9]',value=r'')
    spectra_filt = spectra_filt.loc[spectra_filt['col_energy'] != '']
    spectra_filt = spectra_filt.loc[spectra_filt['inst_type']=='Q-TOF']
    
    adduct = ['[M+H]+', '[M+Na]+']
    spectra_filt_add = spectra_filt.loc[spectra_filt['prec_type'].isin(adduct)]
    spectra_filt_add = spectra_filt_add.loc[spectra_filt_add['spec_type'] == 'MS2']

    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(list(spectra_filt_add.mol_id))]
    
    collision_energy = []
    isotope_num = []
    CEfinal = compounds_filt
    spectra_filt = spectra_filt.loc[spectra_filt['mol_id'].isin(list(copy.mol_id))]
    collision_all = []

    for i, molecule in copy.iterrows():
        molid = molecule['mol_id']
        query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = molid, col_energy = col_energy, change=change, q3 = q3, spectra_filt = spectra_filt, inst_type = inst_type,choose=False)
        query_molid = query['mol_id']

        collision_opt = []
        isotope_num.append(len(background))

        if len(background)>=1: #if theres more than one isotope
            background_id = list(set(background['mol_id']))

            for molid in background_id: #for each isotope
                compared = background.loc[background['mol_id']==molid]
                score_matrix = similarity_score(query_spec=query, compared_spec = compared)
                score = optimized_score(score_matrix)
                collision_opt.append(score)
                
        collision_opt = (list(itertools.chain.from_iterable(collision_opt)))
        collision_all.append(collision_opt)
        
        if len(collision_opt) == 0:
            collision_energy.append(-1)
        else:          
            try:
                collision_mode = mode(collision_opt)
            except:
                collision_mode = collision_opt
            collision_energy.append(collision_mode)
    
    CEfinal['AllCE'] = collision_all
    CEfinal['Optimal Collision Energy'] = collision_energy
    CEfinal['Isotopes'] = isotope_num
    return CEFinal


def optimized_score(scored_matrix):
    length = list(scored_matrix.columns)
    width = list(scored_matrix.index)

    #if more than one match for pairs, need to reset index (can use this value in length and width to get value)
    scored_matrix2 = scored_matrix.reset_index(drop=True)
    scored_matrix2.columns = range(len(length))
    
    minimum =  min([(a) for (a,b) in list(scored_matrix2.min())]) #find minimum diff of col_eng
    starts = pd.DataFrame(scored_matrix2.min())
    starts.columns = ['Starts']
    
    starts = starts[starts['Starts'].apply(lambda x: (x[0]) == minimum)] #dataframe with index as col eng from columns that have minimum coll diff
    
    CE = []

    for col in starts.index: #columns that have starts
        pair = starts.loc[starts.index==col]
        row = scored_matrix2[col].loc[scored_matrix2[col].astype('str')==str(pair.values[0][0])].index[0]

        down=False
        right=False
        best = False #we've found the best score (score is lower than current score)
        score = pair.values[0][0]
        
        while (col<len(length)) & (row< len(width)) & (col>=0) & (row>=0) & (len(length)>1)& (len(width)>1)&(best==False): 
            query_score = score[1]
            query_diff = score[0]

            if col <= int(scored_matrix2.columns[(len(length)//2)-1]):
                newcol = col+1
                right=True
            else:
                newcol = col-1
                right=False
            if row <= int(scored_matrix2.index[(len(width)//2)-1]):
                newrow = row+1
                down=True
            else:
                newrow = row-1
                down=False

            horizontal = scored_matrix2[newcol][row]
            diagonal = scored_matrix2[newcol][newrow]
            vertical = scored_matrix2[col][newrow]
            
            compare = [horizontal, diagonal, vertical, score] #add current score into comparison

            #prioritize comparisons
            compare_f = [[a,b] for [a,b] in compare if (a<=(query_diff))]
            compare_f = [[a,b] for [a,b] in compare_f if (b<=(query_score))]

            if len(compare_f) > 0:
                final = min(compare_f)

            if final == score:
                best = True
            else:
                best=False
                
                if compare.index(final) == 0:
                    if right==True:
                        col = col+1
                    else:
                        col = col-1
                elif compare.index(final) == 1:
                    if right==True:
                        col = col+1
                    else:
                        col = col-1
                    if down==True:
                        row = row+1
                    else:
                        row = row-1
                elif compare.index(final) == 2:
                    if down==True:
                        row = row+1
                    else:
                        row = row-1

                score = final
                
        final_CE = [score, length[col], width[row]] 
        CE.append(length[col])

    try:
        final_score = [mode(CE)]
    except:
        final_score = CE

    return(final_score)


#Visualization (graphs & plots)
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


