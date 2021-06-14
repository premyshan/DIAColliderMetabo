#collision energy optimization - qtof/hcd overlap to see how optimal CE performs on HCD instruments
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
from collections import Counter

allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
compounds_filt, spectra_filt = filter_comp(compounds_filt = allcomp, spectra=spectra, inst_type=['Q-TOF', 'HCD'], col_energy=0, adduct=['[M+H]+'])
compounds_filt, spectra_filt = optimal_ce_filter(compounds_filt=compounds_filt, spectra_filt=spectra_filt)

qtof = spectra_filt.loc[spectra_filt['inst_type']=='Q-TOF']
qtof = qtof[qtof['mol_id'].map(qtof['mol_id'].value_counts()) > 1] #can't do mol_id filter for all, since only one instrument will be taken

hcd = spectra_filt.loc[spectra_filt['inst_type']=='HCD']
hcd = hcd[hcd['mol_id'].map(hcd['mol_id'].value_counts()) > 1]

both = set(hcd['mol_id']).intersection(set(qtof['mol_id']))
compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(both)]
hcd = hcd.loc[hcd['mol_id'].isin(compounds_filt.mol_id)]

optimal_ce = collision_energy_optimizer(compounds_filt = compounds_filt, spectra_filt = hcd)
optimal_ce.to_csv("ce_opt_609_hcdoverlap_25da.csv")
