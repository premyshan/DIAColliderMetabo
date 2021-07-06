# DIAColliderMetabo
Evaluating complex backgrounds that may cause ambiguities in the measurement
of metabolites. This tool first filters a list of identified metabolites to
remove steroisomers (using the Inchikey) and the given experimental conditions.
This filtered list is then used to profile different methods for unique transitions
as follows using MS1 and MS2 windows with the mentioned filters to identify the
number of unique ion signatures (UIS) per molecular id (mol_id).

srmcollidermetabo.py - main analysis script\
mainrun17.py, mainrun20sub.py - profiling MS methods using NIST 17 and validating using novel compounds in NIST 20 respectively\
libsize17.py - saturation analyses based on matrix complexity using NIST 17\
transnum17.py - saturation analyses based on transition number used for UIS using NIST 17\
CE17.py - finding pairwsie-optimal collision energies (POCE) per compound using NIST 17 (Q-TOF instruments)\
CE17_hcd.py and CE17_qtof.py - comparing POCE groups for overlapping compounds measured using Q-TOF and HCD instruments\
mainrun17_hcd & mainrun17_it.py - comparing profiling for overlapping compounds from instruments using HCD vs CID (IT-FT)\
plots.py - plotting script for manuscript
