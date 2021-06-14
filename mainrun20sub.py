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

allcomp20, spectra20 = read(compounds = 'comp_df20.pkl', spectra = 'spec_df20.pkl')
allcomp17, spectra17 = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
spectra17 = spectra17.loc[spectra17['inst_type'].isin(['IT-FT/ion trap with FTMS', 'HCD', 'Q-TOF'])] #nist20 high res inst used 
allcomp17 = allcomp17.loc[allcomp17['mol_id'].isin(spectra17.mol_id)]

spectra = spectra20.loc[~spectra20['nist_num'].isin(spectra17['nist_num'])]
allcomp = allcomp20.loc[allcomp20['mol_id'].isin(spectra.mol_id)]

#compounds might still exist in nist20 that are in nist17 (additional spec measured)
allcomp = allcomp.loc[~allcomp['inchikey'].str[:14].isin(allcomp17['inchikey'].str[:14])]
spectra = spectra.loc[spectra['mol_id'].isin(allcomp.mol_id)]

compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra)

ms1_7 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0.7, ppm=0, change_q3=0, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=0, q3=False)
ms1_7.to_csv('ms1_7_609nist20sub.csv')

ms1_25 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=0, q3=False)
ms1_25.to_csv("ms1_25_609nist20sub.csv")

mrm_7_7_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=1, q3=True)
mrm_7_7_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=2, q3=True)
mrm_7_7_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=3, q3=True)

mrm_7_7_1.to_csv("mrm_7_7_1_609nist20sub.csv")
mrm_7_7_2.to_csv("mrm_7_7_2_609nist20sub.csv")
mrm_7_7_3.to_csv("mrm_7_7_3_609nist20sub.csv")

swath_25da_25_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=1, q3=True)
swath_25da_25_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=2, q3=True)
swath_25da_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)

swath_25da_25_1.to_csv("swath_25da_25_1_609nist20sub.csv")
swath_25da_25_2.to_csv("swath_25da_25_2_609nist20sub.csv")
swath_25da_25_3.to_csv("swath_25da_25_3_609nist20sub.csv")

swath_25_25_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=1, q3=True)
swath_25_25_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=2, q3=True)
swath_25_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)

swath_25_25_1.to_csv("swath_25_25_1_609nist20sub.csv")
swath_25_25_2.to_csv("swath_25_25_2_609nist20sub.csv")
swath_25_25_3.to_csv("swath_25_25_3_609nist20sub.csv")

prm_2_20_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=2, ppm=0, change_q3=0, ppm_q3=20, top_n=0.1, mol_id=0, uis_num=1, q3=True)
prm_2_20_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=2, ppm=0, change_q3=0, ppm_q3=20, top_n=0.1, mol_id=0, uis_num=2, q3=True)
prm_2_20_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=2, ppm=0, change_q3=0, ppm_q3=20, top_n=0.1, mol_id=0, uis_num=3, q3=True)

prm_2_20_1.to_csv("prm_2_20_1_609nist20sub.csv")
prm_2_20_2.to_csv("prm_2_20_2_609nist20sub.csv")
prm_2_20_3.to_csv("prm_2_20_3_609nist20sub.csv")
