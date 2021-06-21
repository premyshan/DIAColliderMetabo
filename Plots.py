from srmcollidermetabo import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import curve_fit
from statistics import mean
import ast
from collections import Counter
import itertools
import matplotlib
import rdkit.Chem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
import PIL.Image as Image
import io
from matplotlib.pyplot import imshow


def comp_spec_details(col_energy = 35, col_gas = 'N2', ion_mode = 'P',inst_type = ['Q-TOF', 'HCD'], adduct = ['[M+H]+', '[M+Na]+']):
    compounds_filt, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    #adduct = ['[M+H]+']
    print(compounds_filt["inchikey"])
    print(len(compounds_filt))
    compounds_filt['inchikey'] = compounds_filt['inchikey'].str[:14]
    print(compounds_filt["inchikey"])
    print(len(compounds_filt))

    compounds_filt = compounds_filt.drop_duplicates(subset='inchikey', keep=False)
    spectra_filt_all = spectra.loc[spectra['mol_id'].isin(compounds_filt.mol_id)]
    print(len(compounds_filt))
    print(len(spectra_filt_all))

    if ion_mode != '':
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['ion_mode'] == str(ion_mode)]
    cf = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_all.mol_id)]
    print(len(cf))
    print(len(spectra_filt_all))

    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['res']>=2]
    print(len(spectra_filt_all))
    print(spectra_filt_all.res.value_counts())

    cf = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_all.mol_id)]
    print(len(cf))
    print(len(spectra_filt_all))

    if inst_type != '':
        inst_type = [str(x) for x in inst_type]
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['inst_type'].isin(inst_type)]

    cf = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_all.mol_id)]
    print(len(cf))
    print(len(spectra_filt_all))

    print(spectra_filt_all['col_energy'].value_counts())
    print(len(spectra_filt_all))
    spectra_filt_all['col_energy'] = spectra_filt_all['col_energy'].apply(lambda x: str(x).split('%')[-1])
    print(spectra_filt_all.col_energy.value_counts())
    print(set(spectra_filt_all.col_energy))
    print(len(spectra_filt_all))
    print(len(set(spectra_filt_all.col_energy)))
    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_energy']!=""]
    print(len(spectra_filt_all))
    print(len(set(spectra_filt_all.col_energy)))
    spectra_filt_all['col_energy'].replace(regex=True,inplace=True,to_replace='[^0-9.]',value=r'')
    print(spectra_filt_all.col_energy.value_counts())
    print(set(spectra_filt_all.col_energy))
    print(len(spectra_filt_all))
    spectra_filt_all.loc[:,'col_energy'] = spectra_filt_all['col_energy'].astype(float)
    print(spectra_filt_all.col_energy.value_counts())
    print(set(spectra_filt_all.col_energy))
    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_energy']!=0.]
    cf = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_all.mol_id)]
    print(len(cf))
    print(len(spectra_filt_all))
    print(spectra_filt_all.col_energy.value_counts())
    print(set(spectra_filt_all.col_energy))
    print(len(set(spectra_filt_all.col_energy)))
    
    if col_energy != 0:
        low = int(col_energy)-5
        high = int(col_energy)+5
        print(spectra_filt_all['col_energy'].value_counts())
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_energy'].between(low, high, inclusive = True)]
    cf = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_all.mol_id)]
    print(len(cf))
    print(len(spectra_filt_all))
    print(spectra_filt_all)

    if col_gas != '':
        spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['col_gas'] == str(col_gas)]

    cf = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_all.mol_id)]
    print(len(cf))
    print(len(spectra_filt_all))
    print(spectra_filt_all)

    spectra_filt_all.loc[:,'peaks'] = spectra_filt_all['peaks'].apply(lambda x: [(a,b/(max(x,key=itemgetter(1))[1])) for (a,b) in x])
    print(spectra_filt_all)
    print(spectra_filt_all['peaks'])

    print("before")
    print(set(spectra_filt_all.spec_type))

    spectra_filt_all = spectra_filt_all.loc[spectra_filt_all['spec_type'] == 'MS2']
    print(len(spectra_filt_all))
    print(set(spectra_filt_all.spec_type))

    spectra_filt_all.loc[:,'prec_mz'] = spectra_filt_all['prec_mz'].astype(float)
    print(len(spectra_filt_all))
    print(spectra_filt_all.prec_mz)

    if adduct != []:
        adduct = [str(x) for x in adduct]
        spectra_filt_add = spectra_filt_all.loc[spectra_filt_all['prec_type'].isin(adduct)]
    else:
        spectra_filt_add = spectra_filt_all
    print(len(spectra_filt_add))
    print(len(set(spectra_filt_add.mol_id)))

    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_add.mol_id)]
    print(len(compounds_filt))
    print(len(spectra_filt_add))

    print(set(spectra_filt_all['col_energy']))
    print(set(spectra_filt_all['ion_mode']))
    print(set(spectra_filt_all['inst_type']))
    print(set(spectra_filt_all['prec_type']))
    print(set(spectra_filt_all['spec_type']))
    print(set(spectra_filt_all['col_gas']))
    print(set(spectra_filt_all['ion_type']))

    print(set(spectra_filt_add['col_energy']))
    print(set(spectra_filt_add['ion_mode']))
    print(set(spectra_filt_add['inst_type']))
    print(set(spectra_filt_add['prec_type']))

    print(spectra_filt_add.prec_type.value_counts())
    print(spectra_filt_add.inst_type.value_counts())
    print(spectra_filt_add.col_energy.value_counts())

    print(spectra_filt_all.prec_type.value_counts())
    print(spectra_filt_all.inst_type.value_counts())
    print(spectra_filt_all.col_energy.value_counts())

    sns.set_palette("rocket")
    pd.to_numeric(spectra_filt_all['col_energy']).hist(by=spectra_filt_all['inst_type'])
    plt.show()

    #print spectra all
    print("spectra all")

    #'purple', 'deeppink', 
    sns.set_palette("rocket", n_colors = 3)
    print(max(spectra_filt_all['prec_mz']))
    print(min(spectra_filt_all['prec_mz']))
    sns.distplot(spectra_filt_all['prec_mz'], kde=False)
    plt.show()

    #print spectra add
    print("spectra add")
    sns.set_palette("rocket", n_colors = 3)
    sns.distplot(spectra_filt_add['prec_mz'], kde=False)
    plt.show()

    #compounds_filt, spectra_filt = optimal_ce_filter(compounds_filt, spectra_filt_all, "[M+H]+")
    print(len(compounds_filt))
    print(len(spectra_filt))

    print('hcd')
    hcd = spectra_filt_all.loc[spectra_filt_all['inst_type']=='HCD']
    print(len(set(hcd['col_energy'])))
    print(len(hcd))
    print(len(set(hcd.mol_id)))
    print('qtof')
    qtof = spectra_filt_all.loc[spectra_filt_all['inst_type']=='Q-TOF']
    print(len(set(qtof['col_energy'])))
    print(len(qtof))
    print(len(set(qtof.mol_id)))
    
    both = set(hcd['mol_id']).intersection(set(qtof['mol_id']))
    compounds_filt = compounds_filt.loc[compounds_filt['mol_id'].isin(both)]
    print(len(compounds_filt))
    hcd = hcd.loc[hcd['mol_id'].isin(compounds_filt.mol_id)]
    print(len(hcd))
    print(len(set(hcd.mol_id)))
    
    qtof = qtof.loc[qtof['mol_id'].isin(compounds_filt.mol_id)]
    print(len(qtof))
    print(len(set(qtof.mol_id)))

    return compounds_filt, spectra_filt_all

def uis_plot(ms1 = ["ms1_7", "ms1_25"], ms2 = ["mrm_7_7", "swath_25da_25", "prm_2_20","swath_25_25"], sizes = [1,2,3], file_suffix = "_609nist17.csv", labels = ['SWATH- 25Da/25ppm', 'SWATH- 25Da/10ppm','SWATH- 25Da/1ppm','SWATH- 25ppm/25ppm','SWATH- 25ppm/10ppm','SWATH- 25ppm/1ppm']):
    uis_all = []
    for size in sizes:
        uis = []
        if ms1 != []:
            for file in ms1: #xaxis
                name = str(file)+file_suffix
                query = pd.read_csv(name)
                #print(len(query))
                #print(name)
                
                query = query.loc[query['UIS']!=-1]
                #print(len(query))
                unique = len(query.loc[query['UIS'] == 1])
                #print(unique)
                unique = ((len(query.loc[query['UIS'] == 1]))/len(query))*100
                uis.append(unique)

        if ms2 != []:            
            for file in ms2:
                name = str(file)+"_"+str(size)+file_suffix
                query = pd.read_csv(name)
                #print(len(query))
                #print(name)
                
                query = query.loc[query['UIS']!=-1]
                #print(len(query))
                unique = len(query.loc[query['UIS'] == 1])
                #print(unique)
                unique = ((len(query.loc[query['UIS'] == 1]))/len(query))*100
                uis.append(unique)
        uis_all.append(uis)

    #print(uis_all)

    if ms2 == []:
        d = {'UIS': uis}
    else:
        d = {'UIS1': uis_all[0], 'UIS2':uis_all[1], 'UIS3':uis_all[2]}

    df = pd.DataFrame(data=d)
    print(df)

    df.index = labels
    sns.set_palette("rocket", n_colors = 3)
    ax = df.plot.bar()
    ax.set_ylim(0, 100)

    plt.show()

    df = 100-df
    print(df)

def library_size_saturation(sizes = [1000,2000,3000,4000,5000,6000,7000,8000,9000], files = ["ms1", "mrm", "swath25da", "swath25"], files_suffix = "_611.csv", labels = ["MS1-25ppm", "MRM-0.7Da/0.7Da-UIS3", "SWATH-25Da/25ppm-UIS3","SWATH-25ppm/25ppm-UIS3"]):
    sizes = sizes
    sns.set_palette("rocket", n_colors = 8)
    colours = iter(sns.color_palette("rocket", n_colors=5))

    def func(x, p1,p2):
        return p1*np.log(x)+p2
    def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2
    UIS_all = []

    for file in files:
        UIS = []
        for size in sizes:
            name = str(file)+"_"+str(size)+files_suffix
            print(name)
            df = pd.read_csv(name, header=0)
            df = df.loc[df['UIS']!=-1]
            print(len(df))
            
            splitting = df.loc[df['cas_num']==str('cas_num')]
            print(len(splitting))
            splitting = splitting[:100]
            df = df.loc[(df['UIS']=='1') | (df['UIS']=='0') | (df['UIS']=='UIS')]
            print(len(df))

            UISsample = []
            count=1
            start=0
            for i,row in splitting.iterrows():
                if count<=100:
                    end = i-1
                    df1= df.loc[start:end]
                    df1 = df1.loc[df1['cas_num']!=str('cas_num')]
                    print(len(df1))
                    
                    unique = ((len(df1.loc[df1['UIS'] == '1']))/len(df1))*100
                    print(len(df1.loc[df1['UIS'] == '1']))
                    print(unique)

                    if len(df1) in sizes:
                        UISsample.append(unique)
                        count += 1
                    start = i+1

                else:
                    break
            UIS.append(np.median(UISsample))
            print(count)

        print(UIS)
        x=np.arange(0,len(sizes))
        print(x)
        y=np.array(UIS)
        print(y)

        d = {'UIS': UIS, 'Sizes': sizes}
        df = pd.DataFrame(data=d)
        print(df)
        model = smf.ols('UIS ~ np.log(Sizes)', data=df).fit()
        print(model.summary())
        

        parameters = model.params
        print(parameters)
        r2 = model.rsquared
        equation= "y = "+str(round(parameters[1],3))+"log(x) + "+str(round(parameters[0],3))+"; R2="+str(round(r2,3))

        plot = sns.scatterplot(x=sizes, y=UIS)    
        
        # plot curve
        curvex=list(np.linspace(1000,15000,15))
        curvey=list(func(curvex,parameters[1],parameters[0]))
        print(curvex)
        print(sizes)
        print(curvey)
        plt.plot(curvex,curvey,'r', color=next(colours),linewidth=2, label=equation)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

        
        plot.set(ylim=(0, 100))
        plot.set_xlabel('MS Method', fontsize=12)
        plot.set_ylabel('Percentage of Unique Compounds', fontsize=12)
        plot.set_ylim(0,100)
        plot.set_title(file, fontsize = 16)
        print(UIS)
        UIS_all.append(UIS)
    plt.show()
    print("all")
    print(UIS_all)


def ce_dist():
    allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra, col_energy=0)
##    compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra, inst_type=['Q-TOF'], col_energy=0, adduct=['[M+H]+'])
##    compounds_filt, spectra_filt = optimal_ce_filter(compounds_filt, spectra_filt, "[M+H]+")   
    sns.set_palette("rocket")
    pd.to_numeric(spectra_filt['col_energy']).hist(by=spectra_filt['inst_type'])
    hcd = spectra_filt.loc[spectra_filt['inst_type']=='HCD']
    print(len(set(hcd['col_energy'])))
    qtof = spectra_filt.loc[spectra_filt['inst_type']=='Q-TOF']
    print(len(set(qtof['col_energy'])))
    plt.show()

    
def ce_opt_plot(file_name = "ce_opt_615_qtof_25da.csv"):
    sns.set_palette("rocket")
    ce2 = pd.read_csv(file_name)
    print(len(ce2))
    ce2 = ce2.loc[ce2['NumSpectra']!=0] #comp with no interferences
    print(len(ce2))
    ce2['AllCE'] = ce2['AllCE'].apply(lambda x: ast.literal_eval(x))

    #find POCE and Opt CE
    all_settings = []
    for i, row in ce2.iterrows():
        count=0
        ce_settings=[]
        row['AllCE'] = [x for x in row['AllCE'] if x!= []]

        while len(row['AllCE'])>0:
            row['freq_ce'] = Counter(list(itertools.chain.from_iterable(row['AllCE'])))
            row['Optimal Collision Energy'] = row['freq_ce'].most_common()[0][0]
            count += 1
            ce_settings.append(row['Optimal Collision Energy'])
            row['AllCE'] = [item for item in row['AllCE'] if row['Optimal Collision Energy'] not in item]            
            #print(row['AllCE'])
        #print(ce_settings)
        #print(count)
        all_settings.append(count)
    ce2['all_ce_settings'] =all_settings
    #print(ce2['all_ce_settings'])
    #print(ce2['all_ce_settings'].value_counts())

    #number of POCE required to differentiate compounds
    ce2 = ce2.sort_values(by=['all_ce_settings'],ascending=True)
    ax = ce2['all_ce_settings'].hist()
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+0.25, height+ 3, '%.0f'%(height))
    ax.grid(False)
    plt.show()
    
def compare_UIS_specific(d, index):
    sns.set_palette("rocket", n_colors = 8)
    df = pd.DataFrame(data=d)
    df.index = index
    cmap = sns.cm.rocket_r
    print( max(d['UIS1']))
    ax = sns.heatmap(df, cmap=cmap,  linewidths=0.1, linecolor='black', annot=True, vmin=0, vmax= max(d['UIS1']))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()

def spectra(queryid, comparedid):
    allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra)
    
    query_spectra = spectra_filt.loc[spectra_filt['spectrum_id'] == queryid]
    query_smiles = compounds_filt.loc[compounds_filt.mol_id==query_spectra.mol_id.item()].smiles.item()
    query_spectra.loc[:,'peaks'] = query_spectra['peaks'].apply(lambda x: [(a,b/(max(x,key=itemgetter(1))[1])*100) for (a,b) in x])
    compared_spectra = spectra_filt.loc[spectra_filt['spectrum_id'] == comparedid]
    compare_smiles = compounds_filt.loc[compounds_filt.mol_id==compared_spectra.mol_id.item()].smiles.item()
    compared_spectra.loc[:,'peaks'] = compared_spectra['peaks'].apply(lambda x: [(a,b/(max(x,key=itemgetter(1))[1])*100) for (a,b) in x])

    print(query_spectra)
    print(compared_spectra)
    query = list(query_spectra['peaks'])[0]
    query2 = pd.DataFrame(query, columns = ['m/z', 'int'])
    compare = list(compared_spectra['peaks'])[0]
    compare2 = pd.DataFrame(compare, columns = ['m/z', 'int'], )
    print(query2)
    print(compare2)

    fig = plt.figure(figsize=(10,10),dpi=100)
    gs = matplotlib.gridspec.GridSpec(3,1,height_ratios=[1,2,2], hspace=0)

    ax_top = fig.add_subplot(gs[1])
    spectra_top = ax_top.stem(query2['m/z'], query2['int'], linefmt = 'blue', markerfmt =' ')
    ax_top.set_xlabel('m/z')
    ax_top.set_xlim(0,350)
    ax_top.set_ylim((0,120))
    ax_top.set_ylabel('Relative Abundance')
    ax_top.set_yticks([0,50,100])
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_bottom = fig.add_subplot(gs[2])
    spectra_bottom = ax_bottom.stem(compare2['m/z'], compare2['int'], linefmt = 'red', markerfmt =' ')
    ax_bottom.set_xlabel('m/z')
    ax_bottom.set_xlim(0,350)
    ax_bottom.set_ylim((120,0))
    ax_bottom.set_ylabel('Relative Abundance')
    ax_bottom.set_yticks([100,50,0])
    yticks_bottom = ax_bottom.yaxis.get_major_ticks()
    yticks_bottom[0].label1.set_visible(False)

    plt.show()

    get_mol_im(query_smiles, queryid)
    get_mol_im(compare_smiles, comparedid)

def transition_num(sizes = [1,2,3,4,5,6,7,8], files = ["mrm_7_7", "swath_25da_25","swath_25_25"], file_suffix ="_trans_609nist17.csv", labels=["MRM-0.7Da/0.7Da", "SWATH-25Da/25ppm","SWATH-25ppm/25ppm"]):
    UIS_all = []
    for file in files:
        UIS = []
        for size in sizes: #xaxis/rows
            name = str(file)+"_UIS"+str(size)+file_suffix
            query = pd.read_csv(name)
            print(len(query))
            query = query.loc[query['UIS']!=-1]
            print(len(query))
            unique = len(query.loc[query['UIS'] == 1])
            #print(unique)
            unique = ((len(query.loc[query['UIS'] == 1]))/len(query))*100
            UIS.append(unique)
        print(UIS)
        UIS_all.append(UIS)
    print("all")
    print(len(UIS_all))

    df = pd.DataFrame(data=UIS_all)
    print(df)
    df = df.transpose() 
    print(df)

    df.columns = labels
    df.index = sizes
    print(df)
    df.to_csv("allUIS.csv")
    sns.set_palette("rocket", n_colors = 3)
    current_palette = sns.color_palette()
    first = current_palette[0]
    second = current_palette[1]
    third = current_palette[2]
    sns.set_palette([third, first, second])

    ax = df.plot.line(linewidth=2)
    ax.set_xlabel('Number of Transitions', fontsize=12)
    ax.set_ylabel('Percentage of Unique Compounds', fontsize=12)
    ax.set_ylim(0,100)

    plt.legend(df.columns, title='Method')
    plt.xlim(0.5,8.5)
    plt.show()
    print(df)

    df = 100-df
    print(df)

def spec_details(top_n=0.1):
    allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra)
    adduct = ['[M+H]+', '[M+Na]+']
    spectra_filt_add = spectra_filt.loc[spectra_filt['prec_type'].isin(adduct)]
    compounds_filt_add = compounds_filt.loc[compounds_filt['mol_id'].isin(spectra_filt_add.mol_id)]

    print("spectra add")
    sns.set_palette("rocket", n_colors = 3)
    sns.distplot(spectra_filt_add['prec_mz'], kde=False)
    plt.show()
    
    print(spectra_filt.peaks)
    print(spectra_filt.columns)

    spectra_filt['peaks'] = [[(a,b) for (a,b) in peaklist if (b>top_n)] for peaklist in spectra_filt['peaks']]
    print(spectra_filt['peaks'])
    transitions = []
    print(len(spectra_filt))

    for i, row in spectra_filt.iterrows():
      transitions.append(len(row['peaks']))

    print(len(transitions))
    print(len(spectra_filt))
    print(transitions)
    length = len(spectra_filt)

    zero = ((transitions.count(0))/length)*100
    one = ((transitions.count(1))/length)*100
    two= ((transitions.count(2))/length)*100
    three = ((transitions.count(3))/length)*100
    four = ((transitions.count(4))/length)*100
    five = ((transitions.count(5))/length)*100
    six = ((transitions.count(6))/length)*100
    seven = ((transitions.count(7))/length)*100
    eight = ((transitions.count(8))/length)*100
    nine = ((transitions.count(9))/length)*100
    print(one+two+three+four+five+six+seven+eight+nine)
    morethan = ((len([i for i in transitions if i>=10]))/length)*100
    print(morethan)

    low = zero+one+two+three+four+five
    medium = six+seven+eight+nine

    d=[low, medium, morethan]
    df = pd.DataFrame(data=d)
    print(df)
    sns.set_palette("rocket", n_colors = 3)
    labels = df.index
    percentages = [low, medium, morethan]

    x = plt.pie(percentages, labels=labels,  
    autopct='%1.0f%%', 
    shadow=False, startangle=0, pctdistance=1.2,labeldistance=1.4)
    plt.show()

    d = [zero, one, two, three, four, five, six, seven, eight, nine, morethan]
    d = {'Transitions':d}
    df = pd.DataFrame(data=d)
    df.index=['0','1','2','3','4','5','6','7','8','9','10']
    print(d)
    sns.set_palette("rocket", n_colors = 10)
    ax = df.plot.bar(legend=False)
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', size=10)
    plt.show()
        

def get_mol_im(smiles, queryid):
	width = 500
	height = 500
	mols = [Chem.MolFromSmiles(smiles)]
	d = rdMolDraw2D.MolDraw2DCairo(width,height)
	d.DrawMolecules(mols)
	d.FinishDrawing()
	png_buf = d.GetDrawingText()
	im = Image.open(io.BytesIO(png_buf))
	im = im.crop((0,0+50,width,height-50))
	im.save(str(queryid)+'.jpg')
	return im

def profile_specific(mol_id, change = 0, ppm = 0, change_q3 = 0, ppm_q3 = 0, adduct = ['[M+H]+', '[M+Na]+'], col_energy=35, q3 = False, top_n = 0.1, uis_num = 0):
    allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra)
    query, background, uis, interferences, transitions = choose_background_and_query(mol_id = mol_id, change = change, ppm = ppm, change_q3 = change_q3, ppm_q3 = ppm_q3,
                                                                                     col_energy = col_energy,adduct = adduct, q3 = q3, top_n = top_n, spectra_filt = spectra_filt, uis_num=uis_num)
    for i,comp in background.iterrows():
        print(comp)
        query_smiles = allcomp.loc[allcomp.mol_id==comp.mol_id].smiles.item()
        get_mol_im(query_smiles, comp.mol_id)
        
    return interferences, uis

###################MAIN UIS PLOT
###main supp all plot
##uis_plot()
###main manuscript plot
##uis_plot(ms1 = ["ms1_25"], ms2 = ["mrm_7_7", "swath_25da_25", "swath_25_25"], sizes = [1,2,3], file_suffix = "_609nist17.csv", labels = ['MS1-25ppm', 'MRM-0.7/0.7Da','SWATH- 25Da/25ppm', 'SWATH- 25ppm/25ppm'])
###main + top_n=0
##uis_plot(ms1 = ["ms1_7", "ms1_25"], ms2 = ["mrm_7_7", "swath_25da_25", "prm_2_20","swath_25_25"], sizes = [1,2,3], file_suffix = "_609nist17_0p.csv", labels = ['MS1-0.7Da', 'MS1-25ppm', 'MRM-0.7/0.7Da','SWATH- 25Da/25ppm', 'PRM-2Da/20ppm','SWATH- 25ppm/25ppm'])
#hcd overlap
##uis_plot(ms1 = ["ms1_7", "ms1_25"], ms2 = ["mrm_7_7", "swath_25da_25", "swath_25_25"], sizes = [1,2,3], file_suffix = "_hcdqtof615.csv", labels = ['MS1-0.7Da', 'MS1-25ppm', 'MRM-0.7/0.7Da','SWATH- 25Da/25ppm', 'SWATH- 25ppm/25ppm'])
#it overlap
##uis_plot(ms1 = ["ms1_7", "ms1_25"], ms2 = ["mrm_7_7", "swath_25da_25", "swath_25_25"], sizes = [1,2,3], file_suffix = "_qtof615.csv", labels = ['MS1-0.7Da', 'MS1-25ppm', 'MRM-0.7/0.7Da','SWATH- 25Da/25ppm', 'SWATH- 25ppm/25ppm'])
#ms1 test
##uis_plot(ms1 = ['ms1_25','ms1_10', 'ms1_1'], ms2 = [], sizes = [1], file_suffix = "_609nist17.csv", labels = ['MS1-25ppm','MS1-10ppm','MS1-1ppm'])
###ms2 test
##uis_plot(ms1 = [], ms2 = ['swath_25da_25', 'swath_25da_10', 'swath_25da_1', 'swath_25_25','swath_25_10', 'swath_25_1'], sizes = [1,2,3], file_suffix = "_609nist17.csv", labels = ['SWATH- 25Da/25ppm', 'SWATH- 25Da/10ppm','SWATH- 25Da/1ppm','SWATH- 25ppm/25ppm','SWATH- 25ppm/10ppm','SWATH- 25ppm/1ppm'])
##uis_plot(ms1 = [], ms2 = ['swath_25da_25', 'swath_25da_10', 'swath_25da_1', 'swath_25_25','swath_25_10', 'swath_25_1', 'swath_10_10', 'swath_1_1'], sizes = [1,2,3], file_suffix = "_609nist17.csv", labels = ['SWATH- 25Da/25ppm', 'SWATH- 25Da/10ppm','SWATH- 25Da/1ppm','SWATH- 25ppm/25ppm','SWATH- 25ppm/10ppm','SWATH- 25ppm/1ppm','SWATH- 10ppm/10ppm','SWATH- 1ppm/1ppm' ])

##
###Library Size
##library_size_saturation(sizes = [1000,2000,3000,4000,5000,6000,7000,8000,9000], files = ["ms1", "mrm", "swath25da", "swath25"], files_suffix = "_8611.csv")
##library_size_saturation(sizes = [1000,2000,3000,4000,5000,6000,7000,8000,9000], files = ["ms1", "mrm", "swath25da", "swath25"], files_suffix = "_611.csv")

#Case by Case Ex.
##d = {'UIS1':[4,3,61,2], 'UIS2':[4,0,0,0], 'UIS3': [4,0,0,0]}
##index = ['MS1-25ppm', 'MRM-0.7/0.7Da','SWATH- 25Da/25ppm', 'SWATH- 25ppm/25ppm']
##compare_UIS_specific(d=d, index=index)
##
##d = {'UIS1':[0,12,30,0], 'UIS2':[0,3,9,0], 'UIS3': [0,3,5,0]}
##index = ['MS1-25ppm', 'MRM-0.7/0.7Da','SWATH- 25Da/25ppm', 'SWATH- 25ppm/25ppm']
##compare_UIS_specific(d=d, index=index)
##
#spectra(queryid=21463, comparedid=23756) #hydroxy
#spectra(queryid=35452, comparedid=22957) #threonine

#profile_specific(mol_id = 2133,change=0,ppm=25,change_q3=0, ppm_q3 = 0, uis_num=1, q3=False)
#profile_specific(mol_id = 3993,change=0.7,ppm=0,change_q3=0.7, ppm_q3 = 0, uis_num=3, q3=True)
#profile_specific(mol_id = 3993,change=25,ppm=0,change_q3=0, ppm_q3 = 25, uis_num=3, q3=True)

#transitions plot
##transition_num()

#spec_details
##spec_details()
##
###CE PLOT
##ce_opt_plot(file_name = "ce_opt_615_qtof_25da.csv")
##ce_opt_plot(file_name = "ce_opt_615_hcdoverlap_25da.csv")
##ce_opt_plot(file_name = "ce_opt_615_qtofoverlap_25da.csv")
##
###ce_det
##ce_dist()
#comp_spec_details --> UIS plot
    
##comp_spec_details()
