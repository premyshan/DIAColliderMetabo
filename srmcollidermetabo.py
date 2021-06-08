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
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pyper as pr
import math
from math import sqrt
#import scipy.spatial as sp
from statistics import mode, mean
#import ast
from operator import itemgetter
from collections import Counter

def my_round(val, decimal=2):
    multiplier = 10**decimal
    return math.floor(val*multiplier+0.5)/multiplier

"""
function read:
input: a list of compounds, a list of spectra
output: pandas dataframes (allcomp and spectra)
"""
def read(compounds, spectra):
    allcomp = pd.read_pickle(compounds) 
    allcomp = allcomp.dropna(subset = ['mol_id'])
    allcomp = allcomp.loc[allcomp.sanitize==True]
    allcomp.loc[:,"mol_id"] = allcomp.mol_id.astype(int)
    spectra = pd.read_pickle(spectra)
    spectra = spectra.dropna(subset = ['mol_id'])
    spectra.loc[:,"mol_id"] = spectra.mol_id.astype(int)

    cf = allcomp
    assert not cf["inchi"].isna().any()
    assert not cf["inchikey"].isna().any()
    spectra = spectra.loc[spectra['mol_id'].isin(cf.mol_id)]

    return cf, spectra

"""
function filter:
Filter the compound list based on the inchikey to take out chiral isomers but keep structural isomers. 

input: Original list of compounds 
output: Filtered compound list
"""
def filter2(compounds_filt, spectra, col_energy = 35, col_gas = '', ion_mode = 'P',inst_type = ['Q-TOF', 'HCD'], adduct = ['[M+H]+', '[M+Na]+']):
    compounds_filt['inchikey'] = compounds_filt['inchikey'].str[:14]

    compounds_filt = compounds_filt.drop_duplicates(subset='inchikey', keep=False)
    spectra_filt_all = spectra.loc[spectra['mol_id'].isin(compounds_filt.mol_id)]

    if ion_mode != '':
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['ion_mode'] == str(ion_mode)]

    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['res']>=2]

    if inst_type != '':
        inst_type = [str(x) for x in inst_type]
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['inst_type'].isin(inst_type)]

    spectra_filt_all['col_energy'] = spectra_filt_all['col_energy'].apply(lambda x: str(x).split('%')[-1])
    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_energy']!=""]
    spectra_filt_all['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9.]',value=r'')

    if col_energy != 0:
        low = int(col_energy)-5
        high = int(col_energy)+5
        spectra_filt_all = spectra_filt_all.loc[pd.to_numeric(spectra_filt_all['col_energy']).between(low, high, inclusive = True)]

    if col_gas != '':
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_gas'] == str(col_gas)]

    spectra_filt_all.loc[:,'peaks'] = spectra_filt_all['peaks'].apply(lambda x: [(a,b/(max(x,key=itemgetter(1))[1])) for (a,b) in x])

    spectra_filt_add = spectra_filt_all.loc[spectra_filt_all['spec_type'] == 'MS2']
    
    if adduct != []:
        adduct = [str(x) for x in adduct]
        spectra_filt_add = spectra_filt_add.loc[spectra_filt_add['prec_type'].isin(adduct)]

    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_add.mol_id)]

    spectra_filt = spectra_filt_all
    return compounds_filt, spectra_filt

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

def choose_background_and_query(spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy = 35, q3 = False, top_n = 0.1, UIS_num = 0, choose = True):
    query_opt = spectra_filt.loc[(spectra_filt['mol_id'] == mol_id)]

    # QQ: why wouldn't we do this in the filtering step?
    if adduct != []:
        adduct = [str(x) for x in adduct]
        query_opt = query_opt.loc[query_opt['prec_type'].isin(adduct)]

    same = spectra_filt.loc[spectra_filt['mol_id']==mol_id]
    background_filt = spectra_filt.drop(same.index) #drop spectra from same mol_id
    
    if (choose==True) and (len(query_opt)!=0):
        if len(query_opt)>1:
            # TBD could switch to resolution (maybe)
            query_opt['inst'] = pd.Categorical(query_opt['inst_type'], ordered=True, categories=['HCD','Q-TOF'])
            query_opt['col_energy'] = pd.to_numeric(query_opt['col_energy'])
            query_opt['ce']=(query_opt['col_energy'] - col_energy).abs()
            query_opt['add'] = pd.Categorical(query_opt['prec_type'], ordered=True, categories=['[M+H]+','[M+Na]+'])
            query_opt = query_opt.sort_values(['inst','ce','add'], ascending=True)
            query=query_opt.iloc[:1]
        else:
            query=query_opt

        query_prec_mz = float(query['prec_mz'].item())
        #choosing background
        if ppm != 0:
            change = (ppm/1000000.0)*(query_prec_mz) 
        low = query_prec_mz - (change/2.0)
        high = query_prec_mz + (change/2.0)
        background_filt = background_filt.loc[pd.to_numeric(background_filt['prec_mz']).between(low, high, inclusive = True)]

        #choosing the fragment
        query_frag_mz =  list(query['peaks'])[0]
        query_frag_mz = [(a,b) for (a,b) in query_frag_mz if (b>(top_n))]
        query_frag_mz.sort(key = lambda x: x[1], reverse = True)
            
        # QQ: what is this for?
        f1 = my_round(query_frag_mz[0][0])
        f2 = my_round(query_prec_mz)

        if f1 != f2:
            start = 0
        else:
            start = 1
            UIS_num += 1
        query_frag_mz = query_frag_mz[start:UIS_num]
        query_frag_mz_values = [query[0] for query in query_frag_mz]
        Transitions=len(query_frag_mz_values)

        if q3 == True:
            for transition in query_frag_mz_values:
                if ppm_q3 != 0:
                    change_q3 = (ppm_q3/1000000.0)*(transition)

                low = transition - (change_q3/2.0)
                high = transition + (change_q3/2.0)
                transitions_q1 = [[(a,b) for (a,b) in peaklist if a>=low and a<=high and (b>(top_n))] for peaklist in background_filt['peaks']] #do transitions here
                transitions_q1 = [x for x in transitions_q1 if x!= []]
                transitions_q1 = list(itertools.chain.from_iterable(transitions_q1))
                transitions_q1.sort(key = lambda x: x[1], reverse = True) # sort intensity
                background_filt = background_filt.loc[(background_filt['peaks'].apply(lambda x: any(transition in x for transition in transitions_q1)))]

        Interferences = len(np.unique(background_filt.mol_id))

        if Interferences == 0:
            UIS=1
        else:
            UIS=0
        
    elif (choose==False) and (len(query_opt)!=0):
        
        assert len(adduct) == 1, adduct
        query=query_opt
        # QQ: why do you choose the first one, arbitrarily?
        query_prec_mz=float(list(query_opt['prec_mz'])[0])
            
        if ppm != 0:
            change = (ppm/1000000.0)*(query_prec_mz)
        low = query_prec_mz - (change/2.0)
        high = query_prec_mz + (change/2.0)
        background_filt = background_filt.loc[pd.to_numeric(background_filt['prec_mz']).between(low, high, inclusive = True)]
        if q3 == True:
            query_frag_mz =  list(query['peaks'])[0]
            query_frag_mz = [(a,b) for (a,b) in query_frag_mz if (b>(top_n))]
            query_frag_mz.sort(key = lambda x: x[1], reverse = True)

            # QQ: what is this for?
            f1 = my_round(query_frag_mz[0][0])
            f2 = my_round(query_prec_mz)

            if f1 != f2:
                start = 0
            else:
                start = 1
                UIS_num+=1
                
            query_frag_mz = query_frag_mz[start:UIS_num]
            query_frag_mz_values = [query[0] for query in query_frag_mz]
            Transitions = len(query_frag_mz_values)
            
            # QQ: what does this do?
            for transition in query_frag_mz_values:
                if ppm_q3 != 0:
                    change_q3 = (ppm_q3/1000000.0)*(transition)
                low = transition - (change_q3/2.0)
                high = transition + (change_q3/2.0)
                transitions_q1 = [[(a,b) for (a,b) in peaklist if a>=low and a<=high and (b>(top_n))] for peaklist in background_filt['peaks']] #do transitions here
                transitions_q1 = [x for x in transitions_q1 if x!= []]
                transitions_q1 = list(itertools.chain.from_iterable(transitions_q1))
                transitions_q1.sort(key = lambda x: x[1], reverse = True) # these are descending intensities

                background_filt = background_filt.loc[(background_filt['peaks'].apply(lambda x: any(transition in x for transition in transitions_q1)))]

        UIS = -1
        Interferences = -1
        Transitions=-1
    else:
        query=query_opt
        UIS = -1
        Interferences = -1
        Transitions = -1

    # QQ: in python it is convention to only capitalize class names, not variables
    return query, background_filt, UIS, Interferences, Transitions  

"""
function profile:
Based on the given parameters calculates the number of USI and Interferences by mol_id.
Input: parameters for choose_background_and_query
Output: compounds list with added columns of 'USI1' and 'Average Interference'
"""
def profile(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy=35, q3 = True, top_n = 0.1, mol_id = 0, UIS_num=0):

    copy = compounds_filt.copy()
    
    UISall = []
    Intall = []
    Transall = []
    for i, molecule in copy.iterrows():
        molid = molecule['mol_id']
        query, background, UIS, interferences, Transitions = choose_background_and_query(mol_id = molid, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                                                         adduct = adduct, col_energy = col_energy, q3 = q3, top_n = top_n, spectra_filt = spectra_filt.copy(), UIS_num=UIS_num)
        
        UISall.append(UIS)
        Intall.append(interferences)
        Transall.append(Transitions)
    copy['UIS'] = UISall
    copy['Interferences'] = Intall
    copy['Transitions'] = Transall
    
    return copy

"""
function profile_specific:
Based on the given parameters calculates the number of USI and Interferences for a specific mol_id that is provided as input.
Input: parameters for choose_background_and_query, a specific mol_id (mol_id), plot_back = True (will plot interferences if present as range of their prec_mz)
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences for a specific mol_id, based on range of prec_mz) 
"""
def profile_specific(compounds_filt, spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy=35, q3 = False, top_n = 0.1, UIS_num = 0):

    query, background, frag_mz, frag_mz_int = choose_background_and_query(mol_id = mol_id, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                                          adduct = adduct, col_energy = col_energy, q3 = q3, top_n = top_n, spectra_filt = spectra_filt, UIS_num=UIS_num)
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
Profiles datasets according to specific Q1/Q3 windows 
Input: parameters for choose_background_and_query (q3 stays False in this case, no q3 window is taken into account) 
Output: compounds list with added columns of 'UIS' and 'Average Interference'
"""
def MethodProfiler(compounds_filt, spectra_filt,change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy = 35, q3 = True, top_n = 0.1, mol_id = 0, UIS_num = 0,):
    start = time.time()
    profiled = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, adduct = adduct, col_energy = col_energy,
                       q3 = q3, top_n = top_n, mol_id = mol_id, compounds_filt = compounds_filt, spectra_filt = spectra_filt, UIS_num = UIS_num)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled_filtered.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown for MRM profiling")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['UIS'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))    
    return profiled


#CE Optimization
#filter2 --> filters the data based on instrument type, pos ion mode and adduct (just M+H)
#choose_back_and_query --> chooses the interferring compounds for the query based on the given conditions

def collision_energy_optimizer():
    allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    compounds_filt, spectra_filt = filter2(compounds_filt = allcomp, spectra=spectra, inst_type=['Q-TOF'], col_energy=0, ion_mode='P', adduct=['[M+H]+'])
    # QQ: why do we use different kinds of filtering for different functions (in terms of CE, prec_type)
    spectra_filt = spectra_filt.loc[spectra_filt['col_energy'] != '']
    spectra_filt = spectra_filt.loc[spectra_filt['prec_type']== '[M+H]+']

    trans = []
    for i, row in spectra_filt.iterrows(): #dropping comp with less than 3 transitions other than the precursor
        query_prec_mz = float(row['prec_mz'])
        f2 = my_round(query_prec_mz)
        query_frag_mz = list(row['peaks'])
        query_frag_mz.sort(key = lambda x: x[1], reverse = True)
        # QQ: I don't think this calculation does anything
        f1 = [(my_round(a),b) for (a,b) in query_frag_mz]
        f1 = [(a,b) for (a,b) in f1 if a==f2] #if precursor mz in fragment list
        trans.append(row['num_peaks']-len(f1))

    spectra_filt['trans']=trans
    spectra_filt= spectra_filt.loc[spectra_filt['trans']>=3]
    spectra_filt = spectra_filt[spectra_filt['mol_id'].map(spectra_filt['mol_id'].value_counts()) > 1] #drop comp that have only one CE 
    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(list(spectra_filt.mol_id))]
    
    collision_energy = []
    isotope_num = []
    numcomp = []
    collision_all = []
    # QQ: why do you keep making copies?
    copy = compounds_filt.copy()
    spectra_filt = spectra_filt.loc[spectra_filt['mol_id'].isin(list(copy.mol_id))]
    mz=[]


    # parallelize this with multiprocessing
    for i, molecule in copy.iterrows(): #find optimal CE for each compound
        molidp = molecule['mol_id']
        query, background, UIS, Interferences, Transitions  = choose_background_and_query(mol_id = molidp, col_energy = 0, change=25, q3 = False, spectra_filt = spectra_filt.copy(),
                                                                                          choose=False, top_n=0, adduct=['[M+H]+'])

        mz.append(set(query['prec_mz']))
        collision_opt=[]

        # QQ: why is this called isotope_num
        isotope_num.append(len(background['mol_id']))
        numcomp2 = 0 
            
        if len(background)>=1: #if theres is an intef
            background_id = list(set(background['mol_id']))
            numcomp2 = len(background_id) #NUMBER OF ISOTOPES

            for molid in background_id: #for each isotope
                compared = background.loc[background['mol_id']==molid]
                score_matrix = similarity_score3(query_spec=query, compared_spec = compared)
                if score_matrix.empty:
                    score=-1
                else:
                    score = optimized_score_3(score_matrix)
                    c = Counter(score)
                    score = [a for (a,b) in c.most_common() if b==c.most_common(1)[0][1]]
                if score != -1:
                    collision_opt.append(score)                
        numcomp.append(numcomp2)
        collision_all.append(collision_opt)
        collision_opt = (list(itertools.chain.from_iterable(collision_opt)))
        
        if len(collision_opt) == 0:
            collision_energy.append(-1)
        else:          
            try:
                collision_mode = mode(collision_opt)
            except:
                collision_mode = collision_opt
            collision_energy.append(collision_mode)
    
    copy['AllCE'] = collision_all
    copy['Optimal Collision Energy'] = collision_energy
    copy['Isotopes'] = isotope_num
    copy['NumComp'] = numcomp
    copy['m/z'] = mz
    copy.to_csv("CE_opt_604_qtof_25da_mh.csv")
    return copy

#find (diff in CE, cosine similarity score) for query vs. compared compound 
def similarity_score3(query_spec, compared_spec): 
    # QQ: this entire function can be replaced with sparse matrix multiplication

    scores = []
    coll_diff = []
    both_all = []
    
    query_spec.loc[:,"col_energy"] = query_spec["col_energy"].apply(pd.to_numeric)
    query_spec = query_spec.sort_values(by=['col_energy']) # these are ascending CE
    compared_spec.loc[:,"col_energy"] = compared_spec["col_energy"].apply(pd.to_numeric)
    compared_spec = compared_spec.sort_values(by=['col_energy'])
    collision_energies_q = list(query_spec['col_energy'])
    collision_energies_c = list(compared_spec['col_energy'])

    for query_c in collision_energies_q:
        query_spectra2 = query_spec.loc[query_spec['col_energy'] == query_c]
        query = list(query_spectra2['peaks'])[0]
        query.sort(key = lambda  x: x[1], reverse = True) # these are ascending intensity
        query_norm = [(round(a),(b*100)) for (a,b) in query] #ROUND TO NEAREST DA
        query_df = pd.DataFrame(query_norm, columns = ['m/z', 'int'])
        query_score = []
        query_coll_diff = []
        both = []

        for compare_c in collision_energies_c:
            compared_spectra2 = compared_spec.loc[compared_spec['col_energy'] == compare_c]
            compare = list(compared_spectra2['peaks'])[0]
            compare.sort(key = lambda  x: x[1], reverse = True)
            compare_norm = [(round(a),(b*100)) for (a,b) in compare] # ROUND TO NEAREST DA
            compare_df = pd.DataFrame(compare_norm, columns = ['m/z', 'int'])

            aligned_df = pd.merge(query_df, compare_df, on='m/z', how = 'outer') #OUTER MERGE
            aligned_df.fillna(0, inplace = True)

            #score
            u = aligned_df['int_x'].values.reshape(1,-1)
            v = aligned_df['int_y'].values.reshape(1,-1)

            score = cosine_vectorized(u,v)

            if math.isnan(score[0][0])==True:
                score[0][0]=0.0
            
            query_score.append(score[0][0])
            query_coll_diff.append(abs(int(compare_c)-int(query_c)))
            both.append([(abs(int(compare_c)-int(query_c))), score[0][0]])

        scores.append(query_score)
        coll_diff.append(query_coll_diff)
        both_all.append(both)

    #check if all scores are 0 - if so , throw out
    if all(all(v == 0.0 for v in sublist) for sublist in scores):
        all_df= pd.DataFrame
    else:
        all_df = pd.DataFrame(both_all)
        all_df.columns = collision_energies_c
        all_df.index = collision_energies_q
    return(all_df)

def cosine_vectorized(array1, array2):
    y = (array2**2).sum(1)
    x = (array1**2).sum(1, keepdims=1)
    xy = array1.dot(array2.T) #dot product is same as transpose of one times other
    np.seterr(divide='ignore', invalid='ignore')
    return (xy/np.sqrt(x))/np.sqrt(y)

def optimized_score_3(scored_matrix):
    ces_row_list = np.array(scored_matrix.index) 
    cols = [np.array(col.to_list()) for col_name,col in scored_matrix.iteritems()] 
    matrix = np.stack(cols,axis=1) 
    
    min_ce_diff_row = np.min(matrix[:,:,0], axis=1)
    min_ce_diff_mask_row = matrix[:,:,0].T == min_ce_diff_row 

    min_ce_diff_col = np.min(matrix[:,:,0], axis=0)
    min_ce_diff_mask_col = matrix[:,:,0] == min_ce_diff_col 

    min_ce_diff_mask_entries = min_ce_diff_mask_row.T + min_ce_diff_mask_col

    #ces_row, ces_col
    ces_row = ces_row_list.reshape(1,ces_row_list.shape[0])
    ces_col = np.array(scored_matrix.columns)
    ces_col = ces_col.reshape(1,ces_col.shape[0])
    row_mat = np.broadcast_to(ces_row.T,[ces_row.T.shape[0],ces_col.shape[0]])
    col_mat = np.broadcast_to(ces_col,[ces_row.shape[0],ces_col.T.shape[0]])
    diff_mat = row_mat - col_mat
    row_lt = (diff_mat <= 0).astype(np.float) #rows less than
    col_lt = (diff_mat > 0).astype(np.float) #cols less than
    threshold = 0.25
    thresh_mat = threshold*(row_lt*row_mat + col_lt*col_mat) #min of col and row, 10% is threshold

    min_ce_diff_mask_thresh = matrix[:,:,0] <= thresh_mat
    min_ce_diff_mask = min_ce_diff_mask_entries & min_ce_diff_mask_thresh
    fails_thresh = not np.any(min_ce_diff_mask)

    if fails_thresh:
        optimalCE = [-1] #drop if fails threshold
    else:
        min_score = np.min(matrix[:,:,1][min_ce_diff_mask]) 
        min_score_mask = matrix[:,:,1] == min_score 
        both_mask = min_ce_diff_mask & min_score_mask 
        # argmin_row, argmin_col = np.nonzero(both_mask)
        argmin_row = np.max(both_mask,axis=1) 
        # these are the query CEs that achieve minimum (1 or more)
        min_ces_row = ces_row_list[argmin_row]
        optimalCE = min_ces_row.tolist()
    return optimalCE
