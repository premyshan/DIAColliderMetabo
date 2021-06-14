#SRMColliderMetabo
#test overlapping compounds HCD vs. IT/FT (to compare orbitrap instruments measured with CE and NCE) 
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

allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
spectra['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9.]',value=r'') #IT/FT has % which our original script removes 
compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra, inst_type = ['IT-FT/ion trap with FTMS', 'HCD'])

adduct = ['[M+H]+', '[M+Na]+']
spectra_add = spectra_filt.loc[spectra_filt['prec_type'].isin(adduct)]

itft = spectra_filt.loc[spectra_filt['inst_type']=='IT-FT/ion trap with FTMS']
hcd = spectra_filt.loc[spectra_filt['inst_type']=='HCD']

both = set(hcd['mol_id']).intersection(set(itft['mol_id']))
hcd = hcd.loc[hcd['mol_id'].isin(both)]

itftcomp = spectra_add.loc[spectra_add['inst_type']=='IT-FT/ion trap with FTMS']
hcdcomp = spectra_add.loc[spectra_add['inst_type']=='HCD']
bothcomp = set(hcdcomp['mol_id']).intersection(set(itftcomp['mol_id']))

compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(bothcomp)]
spectra_filt = hcd

ms1_7 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0.7, ppm=0, change_q3=0, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=0, q3=False)
ms1_7.to_csv('ms1_7_hcdit609.csv')

ms1_25 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=0, q3=False)
ms1_25.to_csv("ms1_25_hcdit609.csv")

mrm_7_7_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=1, q3=True)
mrm_7_7_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=2, q3=True)
mrm_7_7_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=3, q3=True)

mrm_7_7_1.to_csv("mrm_7_7_1_hcdit609.csv")
mrm_7_7_2.to_csv("mrm_7_7_2_hcdit609.csv")
mrm_7_7_3.to_csv("mrm_7_7_3_hcdit609.csv")

swath_25da_25_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=1, q3=True)
swath_25da_25_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=2, q3=True)
swath_25da_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)

swath_25da_25_1.to_csv("swath_25da_25_1_hcdit609.csv")
swath_25da_25_2.to_csv("swath_25da_25_2_hcdit609.csv")
swath_25da_25_3.to_csv("swath_25da_25_3_hcdit609.csv")

swath_25_25_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=1, q3=True)
swath_25_25_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=2, q3=True)
swath_25_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)

swath_25_25_1.to_csv("swath_25_25_1_hcdit609.csv")
swath_25_25_2.to_csv("swath_25_25_2_hcdit609.csv")
swath_25_25_3.to_csv("swath_25_25_3_hcdit609.csv")

prm_2_20_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=2, ppm=0, change_q3=0, ppm_q3=20, top_n=0.1, mol_id=0, uis_num=1, q3=True)
prm_2_20_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=2, ppm=0, change_q3=0, ppm_q3=20, top_n=0.1, mol_id=0, uis_num=2, q3=True)
prm_2_20_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=2, ppm=0, change_q3=0, ppm_q3=20, top_n=0.1, mol_id=0, uis_num=3, q3=True)

prm_2_20_1.to_csv("prm_2_20_1_hcdit609.csv")
prm_2_20_2.to_csv("prm_2_20_2_hcdit609.csv")
prm_2_20_3.to_csv("prm_2_20_3_hcdit609.csv")

