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
compounds_filt, spectra_filt = filter2(compounds_filt=allcomp, spectra=spectra)

ms1_7 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0.7, ppm=0, change_q3 =0, ppm_q3=0, top_n = 0.1, mol_id=0, UIS_num=1, q3=False)
ms1_7.to_csv('ms1_7_607nist17.csv')

ms1_25 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3 =0, ppm_q3=0, top_n = 0.1, mol_id=0, UIS_num=1, q3=False)
ms1_25.to_csv("ms1_25_607nist17.csv")

mrm_7_7_1 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3 =0.7, ppm_q3=0, top_n = 0.1, mol_id=0, UIS_num=1, q3=True)
mrm_7_7_2 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3 =0.7, ppm_q3=0, top_n = 0.1, mol_id=0, UIS_num=2, q3=True)
mrm_7_7_3 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(),change=0.7, ppm=0, change_q3 =0.7, ppm_q3=0, top_n = 0.1, mol_id=0, UIS_num=3, q3=True)

mrm_7_7_1.to_csv("mrm_7_7_1_607nist17.csv")
mrm_7_7_2.to_csv("mrm_7_7_2_607nist17.csv")
mrm_7_7_3.to_csv("mrm_7_7_3_607nist17.csv")

swath_25da_25_1 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3 =0, ppm_q3=25, top_n = 0.1, mol_id=0, UIS_num=1, q3=True)
swath_25da_25_2 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3 =0, ppm_q3=25, top_n = 0.1, mol_id=0, UIS_num=2, q3=True)
swath_25da_25_3 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=25, ppm=0, change_q3 =0, ppm_q3=25, top_n = 0.1, mol_id=0, UIS_num=3, q3=True)

swath_25da_25_1.to_csv("swath_25da_25_1_607nist17.csv")
swath_25da_25_2.to_csv("swath_25da_25_2_607nist17.csv")
swath_25da_25_3.to_csv("swath_25da_25_3_607nist17.csv")

swath_25_25_1 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3 =0, ppm_q3=25, top_n = 0.1, mol_id=0, UIS_num=1, q3=True)
swath_25_25_2 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3 =0, ppm_q3=25, top_n = 0.1, mol_id=0, UIS_num=2, q3=True)
swath_25_25_3 = MethodProfiler(compounds_filt = compounds_filt.copy(), spectra_filt=spectra_filt.copy(), change=0, ppm=25, change_q3 =0, ppm_q3=25, top_n = 0.1, mol_id=0, UIS_num=3, q3=True)

swath_25_25_1.to_csv("swath_25_25_1_607nist17.csv")
swath_25_25_2.to_csv("swath_25_25_2_607nist17.csv")
swath_25_25_3.to_csv("swath_25_25_3_607nist17.csv")



