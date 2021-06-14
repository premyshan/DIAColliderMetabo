#collision energy optimization - nist17
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
compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra, inst_type=['Q-TOF'], col_energy=0, adduct=['[M+H]+'])
compounds_filt, spectra_filt = optimal_ce_filter(compounds_filt=compounds_filt, spectra_filt=spectra_filt)
optimal_ce = collision_energy_optimizer(compounds_filt=compounds_filt, spectra_filt=spectra_filt)
optimal_ce.to_csv("ce_opt_609_qtof_25da.csv")

