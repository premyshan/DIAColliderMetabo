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
import math
from math import sqrt
from scipy.stats import mode 
from operator import itemgetter
from collections import Counter
from tqdm import tqdm

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
def filter_comp(compounds_filt, spectra, col_energy = 35, col_gas = 'N2', ion_mode = 'P',inst_type = ['Q-TOF', 'HCD'], adduct = ['[M+H]+', '[M+Na]+']):
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
    spectra_filt_all.loc[:,'col_energy'] = spectra_filt_all['col_energy'].astype(float)
    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_energy']!=0.0]

    if col_energy != 0:
        low = int(col_energy)-5
        high = int(col_energy)+5
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_energy'].between(low, high, inclusive = True)]

    if col_gas != '':
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_gas'] == str(col_gas)]

    spectra_filt_all.loc[:,'peaks'] = spectra_filt_all['peaks'].apply(lambda x: [(a,b/(max(x,key=itemgetter(1))[1])) for (a,b) in x])
    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['spec_type'] == 'MS2']
    spectra_filt_all.loc[:,'prec_mz'] = spectra_filt_all['prec_mz'].astype(float)
    
    if adduct != []:
        adduct = [str(x) for x in adduct]
        spectra_filt_add = spectra_filt_all.loc[spectra_filt_all['prec_type'].isin(adduct)]
    else:
        spectra_filt_add = spectra_filt_all
       
    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_add.mol_id)]

    return compounds_filt, spectra_filt_all

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

def choose_background_and_query(spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy = 35, q3 = False, top_n = 0.1, uis_num = 0, choose = True):
    # TBD this function can be replaced by something that does a grouping based on mol_id

    query_opt = spectra_filt.loc[(spectra_filt['mol_id'] == mol_id)]

    if adduct != []:
        adduct = [str(x) for x in adduct]
        query_opt = query_opt.loc[query_opt['prec_type'].isin(adduct)]
        # note: it is possible for query_opt to have 0 elements here!

    query_opt = query_opt.reset_index(drop=True)
    same = spectra_filt.loc[spectra_filt['mol_id']==mol_id]
    background_filt = spectra_filt.drop(index=same.index) #drop spectra from same mol_id
    
    if (choose==True) and (len(query_opt)!=0):
        if len(query_opt)>1:
            query_opt['ce']=(query_opt['col_energy'] - col_energy).abs()
            query_opt['add'] = pd.Categorical(query_opt['prec_type'], ordered=True, categories=['[M+H]+','[M+Na]+'])
            query_opt = query_opt.sort_values(['res','ce','add'], ascending=[False,True,True])
            query=query_opt.iloc[:1]
        else:
            query=query_opt

        query_prec_mz = query['prec_mz'].item()
        #choosing background
        if ppm != 0:
            change = (ppm/1000000.0)*(query_prec_mz) 
        low = query_prec_mz - (change/2.0)
        high = query_prec_mz + (change/2.0)
        background_filt = background_filt.loc[background_filt['prec_mz'].between(low, high, inclusive = True)]

        #choosing the fragment
        query_frag_mz =  list(query['peaks'])[0]
        query_frag_mz = [(a,b) for (a,b) in query_frag_mz if (b>(top_n))]
        query_frag_mz.sort(key = lambda x: x[1], reverse = True)

        f1 = my_round(query_frag_mz[0][0])
        f2 = my_round(query_prec_mz)

        if f1 != f2:
            start = 0
        else:
            start = 1
            uis_num += 1
        query_frag_mz = query_frag_mz[start:uis_num]
        query_frag_mz_values = [query[0] for query in query_frag_mz]
        transitions=len(query_frag_mz_values)

        if q3 == True:
            for transition in query_frag_mz_values:
                if ppm_q3 != 0:
                    change_q3 = (ppm_q3/1000000.0)*(transition)

                low = transition - (change_q3/2.0)
                high = transition + (change_q3/2.0)

                transitions_q1 = [[(a,b) for (a,b) in peaklist if a>=low and a<=high and (b>(top_n))] for peaklist in background_filt['peaks']] 
                transitions_q1 = [x for x in transitions_q1 if x!= []]
                transitions_q1 = list(itertools.chain.from_iterable(transitions_q1))
                transitions_q1.sort(key = lambda x: x[1], reverse = True) 
                background_filt = background_filt.loc[(background_filt['peaks'].apply(lambda x: any(transition in x for transition in transitions_q1)))]

        interferences = len(np.unique(background_filt.mol_id))

        if interferences == 0:
            uis=1
        else:
            uis=0
        
    elif (choose==False) and (len(query_opt)!=0):
        assert len(adduct) == 1, adduct
        query=query_opt
        query_prec_mz=list(query_opt['prec_mz'])[0]
            
        if ppm != 0:
            change = (ppm/1000000.0)*(query_prec_mz)
        low = query_prec_mz - (change/2.0)
        high = query_prec_mz + (change/2.0)
        background_filt = background_filt.loc[background_filt['prec_mz'].between(low, high, inclusive = True)]
        if q3 == True:
            query_frag_mz = list(query['peaks'])[0]
            query_frag_mz = [(a,b) for (a,b) in query_frag_mz if (b>(top_n))]
            query_frag_mz.sort(key = lambda x: x[1], reverse = True)

            f1 = my_round(query_frag_mz[0][0])
            f2 = my_round(query_prec_mz)

            if f1 != f2:
                start = 0
            else:
                start = 1
                uis_num+=1
                
            query_frag_mz = query_frag_mz[start:uis_num]
            query_frag_mz_values = [query[0] for query in query_frag_mz]
            transitions = len(query_frag_mz_values)
            
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
        uis = -1
        interferences = -1
        transitions=-1
    else:
        query=query_opt
        uis = -1
        interferences = -1
        transitions = -1
    return query, background_filt, uis, interferences, transitions  

"""
function profile:
Based on the given parameters calculates the number of USI and Interferences by mol_id.
Input: parameters for choose_background_and_query
Output: compounds list with added columns of 'USI1' and 'Average Interference'
"""

def profile(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy=35, q3 = True, top_n = 0.1, mol_id = 0, uis_num=0):    
    uis_all = []
    int_all = []
    trans_all = []
    for i, molecule in compounds_filt.iterrows():
        molid = molecule['mol_id']
        query, background, uis, interferences, transitions = choose_background_and_query(mol_id = molid, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                                                         adduct = adduct, col_energy = col_energy, q3 = q3, top_n = top_n, spectra_filt = spectra_filt, uis_num=uis_num)
        
        uis_all.append(uis)
        int_all.append(interferences)
        trans_all.append(transitions)
    compounds_filt['UIS'] = uis_all
    compounds_filt['Interferences'] = int_all
    compounds_filt['Transitions'] = trans_all
    return compounds_filt

"""
function profile_specific:
Based on the given parameters calculates the number of UIS and Interferences for a specific mol_id that is provided as input.
Input: parameters for choose_background_and_query, a specific mol_id (mol_id), plot_back = True (will plot interferences if present as range of their prec_mz)
Output: compounds list with added columns of 'USI1' and 'Average Interference', count plot (interferences for a specific mol_id, based on range of prec_mz) 
"""
def profile_specific(compounds_filt, spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy=35, q3 = False, top_n = 0.1, uis_num = 0):
    query, background, uis, interferences, transitions = choose_background_and_query(mol_id = mol_id, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                                                     col_energy = col_energy,adduct = adduct, q3 = q3, top_n = top_n, spectra_filt = spectra_filt, uis_num=uis_num)
    return query, background, uis, interferences, transitions 

"""
function method_profiler:
Profiles datasets according to specific Q1/Q3 windows 
Input: parameters for choose_background_and_query (q3 stays False in this case, no q3 window is taken into account) 
Output: compounds list with added columns of 'UIS' and 'Average Interference'
"""

def method_profiler(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy = 35, q3 = True, top_n = 0.1, mol_id = 0, uis_num = 0):
    start = time.time()
    profiled = profile(change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3, adduct = adduct, col_energy = col_energy,
                       q3 = q3, top_n = top_n, mol_id = mol_id, compounds_filt = compounds_filt, spectra_filt = spectra_filt, uis_num = uis_num)
    profiled_filtered = profiled.loc[profiled['Interferences'] != -1]
    end = time.time()
    list_mol_ids = list(profiled_filtered.mol_id)
    print("The unique identities and interferences for all mol_id will now be shown:")
    print("The number of unique mol_id is: " + str(len([x for x in profiled['UIS'] if x == 1])))
    print("Time to completion of profiler: " + str(end-start))    
    return profiled

def optimal_ce_filter(compounds_filt, spectra_filt, adduct):
    
    spectra_filt = spectra_filt[spectra_filt["prec_type"] == adduct].reset_index(drop=True)
    # this adds mzs and ints column to the spectra
    def get_mzs(peaks):
        mzs = [my_round(mz) for mz in list(zip(*peaks))[0]]
        return mzs
    def get_ints(peaks):
        ints = list(zip(*peaks))[1]
        return ints
    spectra_filt.loc[:,"mzs"] = spectra_filt["peaks"].apply(get_mzs) 
    spectra_filt.loc[:,"ints"] = spectra_filt["peaks"].apply(get_ints)
    def compute_num_trans(row):
        prec_mz = my_round(row["prec_mz"])
        mzs = row["mzs"]
        same_count = np.sum(mz == prec_mz for mz in mzs)
        return len(mzs) - same_count
    spectra_filt['num_trans'] = spectra_filt.apply(compute_num_trans,axis=1)
    spectra_filt = spectra_filt.loc[spectra_filt['num_trans'] >= 3]
    spectra_filt = spectra_filt[spectra_filt['mol_id'].map(spectra_filt['mol_id'].value_counts()) > 1]
    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt.mol_id)]
    spectra_filt = spectra_filt.reset_index(drop=True)
    compounds_filt = compounds_filt.reset_index(drop=True)
    return compounds_filt, spectra_filt

def collision_energy_optimizer(compounds_filt, spectra_filt):

    # quick check that spectra mz are bounded
    max_mz = spectra_filt["mzs"].apply(max).max()
    assert max_mz < 2000., max_mz

    def compute_spec(row, mz_max=2000.):
        mzs = np.array(row["mzs"])
        ints = 100*np.array(row["ints"])
        mz_bins = np.arange(0.5,mz_max+0.5,step=1.0)
        mz_bin_idxs = np.digitize(mzs,bins=mz_bins,right=False)
        spec = np.zeros([len(mz_bins)],dtype=float)
        for i in range(len(mz_bin_idxs)):
            spec[mz_bin_idxs[i]] += ints[i]
        assert np.isclose(np.sum(spec),np.sum(ints)), np.abs(np.sum(spec)-np.sum(ints))
        return spec

    # compute ce diff matrix
    ce_vec = spectra_filt["col_energy"].to_numpy().reshape(-1,1)
    query_mat =  np.broadcast_to(ce_vec,[ce_vec.shape[0],ce_vec.shape[0]])
    background_mat = np.broadcast_to(ce_vec.T,[ce_vec.shape[0],ce_vec.shape[0]])
    ce_diff_mat = query_mat - background_mat
    # compute cosine sim matrix
    spec = spectra_filt.apply(compute_spec,axis=1)
    spec_vec = np.stack(spec.tolist(),axis=0).reshape(spec.shape[0],-1)
    cos_vec = spec_vec / np.sqrt(np.sum(spec_vec**2,axis=1)).reshape(-1,1)
    cos_sim_mat = np.matmul(cos_vec,cos_vec.T)
    # stack them all
    all_mat = np.stack([query_mat,background_mat,ce_diff_mat,cos_sim_mat],axis=-1)
    # get mapping from spectrum id to idx of the matrix
    spec_id2idx = {spec_id:spec_idx for spec_idx,spec_id in enumerate(spectra_filt["spectrum_id"].tolist())}

    # number of interfering spectra, per query
    num_spectra = []
    # number of interfering compounds, per query
    num_comps = []
    # the set of minimal CEs per interfering compound, per query
    all_min_ces = []
    # the precursor mz of the query
    prec_mzs = []

    # find optimal CE for each compound
    for i, mol_id in tqdm(compounds_filt["mol_id"].iteritems(),desc="> optimal_ce",total=compounds_filt.shape[0]):
        
        query, background, _, _, _  = choose_background_and_query(
            mol_id = mol_id, col_energy = 0, change=25, 
            q3 = False, spectra_filt = spectra_filt.copy(),
            choose=False, top_n=0, adduct=['[M+H]+']
        )
        if query.shape[0] == 0:
            # this happens when the mol_id only corresponds to adducts that are not "[M+H]+"
            import pdb; pdb.set_trace()

        query_spec_idx = query["spectrum_id"].map(spec_id2idx).to_numpy()
        background["spec_idx"] = background["spectrum_id"].map(spec_id2idx)
        bg_mol_ids = background["mol_id"].unique().tolist()
        
        assert query["prec_mz"].nunique() == 1, query.nunique()

        num_comps.append(len(bg_mol_ids))
        prec_mzs.append(query["prec_mz"].tolist()[0])
        num_spectra.append(background['spectrum_id'].nunique())

        cur_min_ces = []
        for bg_mol_id in bg_mol_ids:  
            background_spec_idx = background[background["mol_id"] == bg_mol_id]["spec_idx"].to_numpy()
            score_mat = all_mat[query_spec_idx][:,background_spec_idx]
            assert not score_mat.size == 0
            cur_min_ces.append(compute_optimal_ces(score_mat))
        all_min_ces.append(cur_min_ces)

    compounds_filt['AllCE'] = all_min_ces
    compounds_filt['NumSpectra'] = num_spectra
    compounds_filt['NumComp'] = num_comps
    compounds_filt['m/z'] = prec_mzs
    return compounds_filt

def compute_optimal_ces(score_mat):
    
    row_mat = score_mat[:,:,0]
    col_mat = score_mat[:,:,1]
    ce_diff_mat = score_mat[:,:,2] # this is difference
    cos_sim_mat = score_mat[:,:,3]
    ce_abs_diff_mat = np.abs(ce_diff_mat) # this is absolute difference

    min_ce_diff_row = np.min(ce_abs_diff_mat, axis=1)
    min_ce_diff_mask_row = ce_diff_mat.T == min_ce_diff_row 

    min_ce_diff_col = np.min(ce_abs_diff_mat, axis=0)
    min_ce_diff_mask_col = ce_diff_mat == min_ce_diff_col 

    min_ce_diff_mask_entries = min_ce_diff_mask_row.T + min_ce_diff_mask_col

    row_lt = (ce_diff_mat <= 0).astype(np.float) #rows less than
    col_lt = (ce_diff_mat > 0).astype(np.float) #cols less than
    threshold = 0.25
    thresh_mat = threshold * (row_lt*row_mat + col_lt*col_mat) #min of col and row, 25% is threshold

    min_ce_diff_mask_thresh = ce_abs_diff_mat <= thresh_mat
    min_ce_diff_mask = min_ce_diff_mask_entries & min_ce_diff_mask_thresh
    fails_thresh = not np.any(min_ce_diff_mask)

    if fails_thresh:
        min_row_ces = []
    else:
        min_cos_sim = np.min(cos_sim_mat[min_ce_diff_mask]) 
        min_cos_sim_mask = cos_sim_mat == min_cos_sim 
        both_mask = min_ce_diff_mask & min_cos_sim_mask 
        argmin_row_mask = np.max(both_mask,axis=1) 
        # these are the query CEs that achieve minimum (1 or more)
        min_row_ces = row_mat[:,0][argmin_row_mask].tolist()
    # print(min_row_ces)
    # import sys; sys.exit(0)
    return min_row_ces

def test_optimal_ce_1():

    query_mat = np.array([
        [1,1,1,1,1,1],
        [3,3,3,3,3,3],
        [5,5,5,5,5,5],
        [7,7,7,7,7,7]
    ],dtype=float)
    background_mat = np.array([
        [1,2,4,6,7,10],
        [1,2,4,6,7,10],
        [1,2,4,6,7,10],
        [1,2,4,6,7,10]
    ],dtype=float) 
    ce_diff_mat = query_mat - background_mat
    sim_mat = np.array([
        [.1,.3,.5,.2,.1,.1],
        [.2,.1,.1,.1,.1,.1],
        [.2,.1,.4,.1,.1,.1],
        [.3,.3,.2,.3,.1,.1]
    ])

    score_mat = np.stack([query_mat,background_mat,ce_diff_mat,sim_mat],axis=-1)

    expected_minimal_ces = [1.,7.]
    computed_minimal_ces = compute_optimal_ces(score_mat)
    print(expected_minimal_ces,computed_minimal_ces)
    assert set(expected_minimal_ces) == set(computed_minimal_ces)

if __name__ == "__main__":

    test_optimal_ce_1()