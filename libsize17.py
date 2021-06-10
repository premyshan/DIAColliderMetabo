 #SRMColliderMetabo
from srmcollidermetabo import *
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
from statistics import mode, mean
from operator import itemgetter
import sys

fileid = sys.argv[1]

allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
compounds_filt_all, spectra_filt_all = filter_comp(compounds_filt=allcomp, spectra=spectra)

sizes = [1000,2000,3000,4000,5000,6000,7000]

for size in sizes:
    #random subsample of queries
    compounds_filt = compounds_filt_all.sample(n=size, replace=False)
    spectra_filt = spectra_filt_all.loc[spectra_filt_all['mol_id'].isin(compounds_filt.mol_id)]

    ms1_25 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=0, q3=False)
    ms1_25.to_csv("/home/shantham/scratch/ms1_25_"+str(size)+"_"+fileid+"_609nist17.csv")

    mrm_7_7_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=3, q3=True)
    mrm_7_7_3.to_csv("/home/shantham/scratch/mrm_7_7_3_"+str(size)+"_"+fileid+"_609nist17.csv")

    swath_25da_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)
    swath_25da_25_3.to_csv("/home/shantham/scratch/swath_25da_25_3_"+str(size)+"_"+fileid+"_609nist17.csv")

    swath_25_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)
    swath_25_25_3.to_csv("/home/shantham/scratch/swath_25_25_3_"+str(size)+"_"+fileid+"_609nist17.csv")

   
