#collision energy optimization - qtof/hcd overlap to see how optimal CE performs on HCD instruments
from srmcollidermetabo import *

allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
compounds_filt, spectra_filt = filter_comp(compounds_filt = allcomp, spectra=spectra, inst_type=['Q-TOF', 'HCD'], col_energy=0, adduct=['[M+H]+'])
compounds_filt, spectra_filt = optimal_ce_filter(compounds_filt, spectra_filt, "[M+H]+")

qtof_spec = spectra_filt.loc[spectra_filt['inst_type']=='Q-TOF']
qtof_comp = compounds_filt.loc[compounds_filt.mol_id.isin(qtof_spec.mol_id)]
qtof_comp, qtof_spec = optimal_ce_filter(qtof_comp, qtof_spec, '[M+H]+')

hcd_spec = spectra_filt.loc[spectra_filt['inst_type']=='HCD']
hcd_comp = compounds_filt.loc[compounds_filt.mol_id.isin(hcd_spec.mol_id)]
hcd_comp, hcd_spec = optimal_ce_filter(hcd_comp, hcd_spec, '[M+H]+')

both = set(hcd_comp['mol_id']).intersection(set(qtof_comp['mol_id']))
compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(both)]
qtof = qtof_spec.loc[qtof_spec['mol_id'].isin(compounds_filt.mol_id)]

optimal_ce = collision_energy_optimizer(compounds_filt = compounds_filt, spectra_filt = qtof)
optimal_ce.to_csv("ce_opt_609_qtofoverlap_25da.csv")

