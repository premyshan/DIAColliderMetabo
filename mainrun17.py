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

allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra)

ms1_7 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0.7, ppm=0, change_q3=0, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=0, q3=False)
ms1_7.to_csv('ms1_7_609nist17.csv')

ms1_25 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=0, q3=False)
ms1_25.to_csv("ms1_25_609nist17.csv")

mrm_7_7_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=1, q3=True)
mrm_7_7_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=2, q3=True)
mrm_7_7_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3=0.7, ppm_q3=0, top_n=0.1, mol_id=0, uis_num=3, q3=True)

mrm_7_7_1.to_csv("mrm_7_7_1_609nist17.csv")
mrm_7_7_2.to_csv("mrm_7_7_2_609nist17.csv")
mrm_7_7_3.to_csv("mrm_7_7_3_609nist17.csv")

swath_25da_25_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=1, q3=True)
swath_25da_25_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=2, q3=True)
swath_25da_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)

swath_25da_25_1.to_csv("swath_25da_25_1_609nist17.csv")
swath_25da_25_2.to_csv("swath_25da_25_2_609nist17.csv")
swath_25da_25_3.to_csv("swath_25da_25_3_609nist17.csv")

swath_25_25_1 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=1, q3=True)
swath_25_25_2 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=2, q3=True)
swath_25_25_3 = method_profiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3=0, ppm_q3=25, top_n=0.1, mol_id=0, uis_num=3, q3=True)

swath_25_25_1.to_csv("swath_25_25_1_609nist17.csv")
swath_25_25_2.to_csv("swath_25_25_2_609nist17.csv")
swath_25_25_3.to_csv("swath_25_25_3_609nist17.csv")



