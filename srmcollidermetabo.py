 #SRMColliderMetabo
"""
Evaluating complex backgrounds that may cause ambiguities in the measurement
of metabolites. This tool first filters a list of identified metabolites to
remove steroisomers (using the Inchikey) and the given experimental conditions.
This filtered list is then used to profile different methods for unique transitions
as follows using MS1 and MS2 windows with the mentioned filters to identify the
number of unique ion signatures (UIS) per molecular id (mol_id).

MS1/MS2
MS1 -  0.7 Da / - ; 25ppm / -
MRM -  0.7 Da / 0.7 Da
SWATH - 25 Da / 25 ppm; 25 ppm / 25 ppm

This tool will also measure the number of interferences for each transition
(the number of identical transitions within the range of metabolites filtered
as specified above). 

"""
import pandas as pd
import numpy as np
import rdkit
import re
import itertools
import time
import math
from operator import itemgetter
from tqdm import tqdm
import joblib
import contextlib

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
Filter the compound list (stereoisomers and experimental conditions given)
input: list of compounds, list of spectra, collision energy, collision gas, ion mode, instrument type, adducts
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
        low = col_energy-5
        high = col_energy+5
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
Choosing the background for each query (based on mol_id), based on the given MS1 (Q1) and MS2 (Q3) window sizes. 
Fragment spectra are filtered according the top_n value (% relative intensity) and the given n for UIS.
Input: spectra, mol_id, MS1/MS2 window sizes (Q1/Q3, MS1 - change/ppm, MS2 - change_q3/ppm_q3 - if ppm is filled, that will take priority over change),
       query parameters (col_energy, adducts), Q3 parameters (if q3 = True, will take into account change_q3 or ppm_q3 parameters, otherwise only Q1),
       top_n (the top n% of fragment ions), uis (n number of transitions chosen)
Output: query ids, background ids, number of transitions, uis (boolean if compound is unique), interferences (number of interferences per compound)
"""

def choose_background_and_query(spectra_filt, mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy = 35, q3 = False, top_n = 0.1, uis_num = 0, choose = True):

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
            if top_n < 0.1: #default of 0.1 for background relative intensity filter
                top_n = 0.1
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
        
    elif (choose==False) and (len(query_opt)!=0): #not choosing one query, MS1 only filter
        assert len(adduct) == 1, adduct
        query=query_opt
        query_prec_mz=list(query_opt['prec_mz'])[0]
            
        if ppm != 0:
            change = (ppm/1000000.0)*(query_prec_mz)
        low = query_prec_mz - (change/2.0)
        high = query_prec_mz + (change/2.0)
        background_filt = background_filt.loc[background_filt['prec_mz'].between(low, high, inclusive = True)]
        
        uis = -1
        interferences = -1
        transitions=-1
    else:
        query=query_opt
        uis = -1
        interferences = -1
        transitions = -1
    # convert full dfs to just ids
    query_ids = query[["spectrum_id","mol_id"]]
    background_ids = background_filt[["spectrum_id","mol_id"]]
    return query_ids, background_ids, uis, interferences, transitions  


"""
Function to integrate joblib with tqdm progress bar
https://stackoverflow.com/a/58936697/6937913
"""
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

"""
function profile:
Based on the given parameters calculates the number of uis and interferences by mol_id.
Input: parameters for choose_background_and_query
Output: query ids, background ids, number of transitions, uis (boolean if compound is unique), interferences (number of interferences per compound)
"""

def profile(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy=35, q3 = False, top_n = 0.1, mol_id = 0, uis_num=0):
    uis_all = []
    int_all = []
    trans_all = []

    # only keep necessary columns, to reduce memory footprint
    _spectra_filt = spectra_filt[["spectrum_id","mol_id","prec_type","col_energy","res","prec_mz","peaks"]]
    mol_ids = compounds_filt["mol_id"]

    with tqdm_joblib(tqdm(desc="bg & q", total=mol_ids.shape[0])) as pbar:
        par_func = joblib.delayed(choose_background_and_query)
        pool = joblib.Parallel()
        results = pool(
            par_func(
                mol_id = mol_id, change = change, ppm = ppm, 
                change_q3 = change_q3, ppm_q3 = ppm_q3, 
                adduct = adduct, col_energy = col_energy, 
                q3 = q3, top_n = top_n, spectra_filt = _spectra_filt, 
                uis_num=uis_num
            ) for idx, mol_id in mol_ids.iteritems()
        )

    query_ids_all, background_ids_all, uis_all, int_all, trans_all = zip(*results)
    compounds_filt['UIS'] = uis_all
    compounds_filt['Interferences'] = int_all
    compounds_filt['Transitions'] = trans_all
    return compounds_filt

"""
function method_profiler:
Profiles datasets according to specific MS1/MS2 (Q1/Q3) windows 
Input: compounds, spectra, parameters for profile/choose_background_and_query  
Output: compounds list with added columns of 'UIS' and 'Interferences'
"""

def method_profiler(compounds_filt, spectra_filt, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy = 35, q3 = False, top_n = 0.1, mol_id = 0, uis_num = 0):
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

"""
function optimal_ce_filter:
Filter function for collision_energy_optimizer
"""
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

"""
function collision_energy_optimizer:
Finds pairwise-optimal collision energies (POCE) per compound 
"""
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

    # only keep necessary columns, to reduce memory footprint
    spectra_filt = spectra_filt[["spectrum_id","mol_id","prec_type","col_energy","res","prec_mz","peaks"]].copy()

    # find optimal CE for each compound
    for i, mol_id in tqdm(compounds_filt["mol_id"].iteritems(),desc="> optimal_ce",total=compounds_filt.shape[0]):

        query_ids, background_ids, _, _, _  = choose_background_and_query(
            mol_id = mol_id, col_energy = 0, change=25, 
            q3 = False, spectra_filt = spectra_filt,
            choose=False, top_n=0, adduct=['[M+H]+']
        )
        if query_ids.shape[0] == 0:
            # this happens when the mol_id only corresponds to adducts that are not "[M+H]+"
            import pdb; pdb.set_trace()

        query_spec_idx = query_ids["spectrum_id"].map(spec_id2idx).to_numpy()
        background_ids["spec_idx"] = background_ids["spectrum_id"].map(spec_id2idx)
        bg_mol_ids = background_ids["mol_id"].unique().tolist()
        
        query_prec_mzs = spectra_filt[spectra_filt["mol_id"].isin(query_ids["mol_id"])]["prec_mz"]
        assert query_prec_mzs.nunique() == 1, query_prec_mzs.nunique()

        num_comps.append(len(bg_mol_ids))
        prec_mzs.append(query_prec_mzs.tolist()[0])
        num_spectra.append(background_ids['spectrum_id'].nunique())

        cur_min_ces = []
        for bg_mol_id in bg_mol_ids:  
            background_spec_idx = background_ids[background_ids["mol_id"] == bg_mol_id]["spec_idx"].to_numpy()
            score_mat = all_mat[query_spec_idx][:,background_spec_idx]
            assert not score_mat.size == 0
            cur_min_ces.append(compute_optimal_ces(score_mat))
        all_min_ces.append(cur_min_ces)

    compounds_filt['AllCE'] = all_min_ces
    compounds_filt['NumSpectra'] = num_spectra
    compounds_filt['NumComp'] = num_comps
    compounds_filt['m/z'] = prec_mzs
    return compounds_filt

"""
function compute_optimal_ces:
Helper function for computing optimal POCE (collision_energy_optimizer)
"""
def compute_optimal_ces(score_mat):
    
    row_mat = score_mat[:,:,0]
    col_mat = score_mat[:,:,1]
    ce_diff_mat = score_mat[:,:,2] # this is difference
    cos_sim_mat = score_mat[:,:,3]
    ce_abs_diff_mat = np.abs(ce_diff_mat) # this is absolute difference

    min_ce_diff_row = np.min(ce_abs_diff_mat, axis=1)
    min_ce_diff_mask_row = ce_abs_diff_mat.T == min_ce_diff_row 

    min_ce_diff_col = np.min(ce_abs_diff_mat, axis=0)
    min_ce_diff_mask_col = ce_abs_diff_mat == min_ce_diff_col 

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

"""
testing
"""
def test_optimal_ce_1():

    query_ce = np.array([1.,3.,5.,7.]).reshape(-1,1)
    bg_ce = np.array([1.,2.,4.,6.,7.,10.]).reshape(1,-1)
    query_mat = np.broadcast_to(query_ce,[query_ce.shape[0],bg_ce.shape[1]])
    background_mat = np.broadcast_to(bg_ce,[query_ce.shape[0],bg_ce.shape[1]])
    ce_diff_mat = query_mat - background_mat
    sim_mat = np.array([
        [.1,.3,.5,.2,.1,.1],
        [.2,.1,.1,.1,.1,.1],
        [.2,.1,.4,.1,.1,.1],
        [.3,.3,.2,.3,.1,.1]
    ])

    score_mat = np.stack([query_mat,background_mat,ce_diff_mat,sim_mat],axis=-1)

    expected_minimal_ces = [1.,5.,7.]
    computed_minimal_ces = compute_optimal_ces(score_mat)
    print(expected_minimal_ces,computed_minimal_ces)

def test_optimal_ce_2():

    query_ce = np.array([1.,3.,5.]).reshape(-1,1)
    bg_ce = np.array([2.,4.,6.]).reshape(1,-1)
    query_mat = np.broadcast_to(query_ce,[query_ce.shape[0],bg_ce.shape[1]])
    background_mat = np.broadcast_to(bg_ce,[query_ce.shape[0],bg_ce.shape[1]])
    ce_diff_mat = query_mat - background_mat
    sim_mat = np.array([
        [.1,.1,.1],
        [.1,.1,.1],
        [.1,.1,.1]
    ])
    score_mat = np.stack([query_mat,background_mat,ce_diff_mat,sim_mat],axis=-1)

    expected_minimal_ces = [5.]
    computed_minimal_ces = compute_optimal_ces(score_mat)
    print(expected_minimal_ces,computed_minimal_ces)

def test_optimal_ce_3():

    query_ce = np.array([1.,3.]).reshape(-1,1)
    bg_ce = np.array([8.,9.,11.]).reshape(1,-1)
    query_mat = np.broadcast_to(query_ce,[query_ce.shape[0],bg_ce.shape[1]])
    background_mat = np.broadcast_to(bg_ce,[query_ce.shape[0],bg_ce.shape[1]])
    ce_diff_mat = query_mat - background_mat
    sim_mat = np.array([
        [.1,.2,.3],
        [.1,.2,.3]
    ])
    score_mat = np.stack([query_mat,background_mat,ce_diff_mat,sim_mat],axis=-1)

    expected_minimal_ces = []
    computed_minimal_ces = compute_optimal_ces(score_mat)
    print(expected_minimal_ces,computed_minimal_ces)

if __name__ == "__main__":

    test_optimal_ce_1()
    test_optimal_ce_2()
    test_optimal_ce_3()
