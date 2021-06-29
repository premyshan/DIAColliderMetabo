from srmcollidermetabo import *
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import ast
from collections import Counter
import rdkit.Chem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
import PIL.Image as Image
import io

def uis_plot(ms1 = ["ms1_7", "ms1_25"], ms2 = ["mrm_7_7", "swath_25da_25", "prm_2_20","swath_25_25"], sizes = [1,2,3], file_suffix = "_622nist17.csv",
             labels = ['MS1-0.7Da', 'MS1-25ppm', 'MRM-0.7/0.7Da','SWATH- 25Da/25ppm', 'PRM-2Da/20ppm','SWATH-25ppm/25ppm']):
    uis_all = []
    for size in sizes:
        uis = []
        if ms1 != []:
            for file in ms1:
                name = str(file)+file_suffix
                query = pd.read_csv(name)                
                query = query.loc[query['UIS']!=-1]
                unique = len(query.loc[query['UIS'] == 1])
                unique = ((len(query.loc[query['UIS'] == 1]))/len(query))*100
                uis.append(unique)

        if ms2 != []:            
            for file in ms2:
                name = str(file)+"_"+str(size)+file_suffix
                query = pd.read_csv(name)
##                sns.set_palette("rocket", n_colors = 4)
##                ax = sns.countplot(x=query['Transitions'])
##                print(query['Transitions'].value_counts())
##                plt.show()
                
                query = query.loc[query['UIS']!=-1]
                unique = len(query.loc[query['UIS'] == 1])
                unique = ((len(query.loc[query['UIS'] == 1]))/len(query))*100
                uis.append(unique)
        uis_all.append(uis)

    if ms2 == []:
        d = {'UIS': uis}
    else:
        d = {'UIS1': uis_all[0], 'UIS2':uis_all[1], 'UIS3':uis_all[2]}

    df = pd.DataFrame(data=d)
    print(df)

    df.index = labels
    sns.set_palette("rocket", n_colors = 3)

    #x=index, y=all numerical values
    ax = df.plot.bar() 
    ax.set_ylim(0, 100)

    plt.show()

    df = 100-df
    print(df)

def library_size_saturation(sizes = [1000,2000,3000,4000,5000,6000,7000,8000,9000], files = ["ms1", "mrm", "swath25da", "swath25"], files_suffix = "_622.csv",
                            labels = ["MS1-25ppm", "MRM-0.7Da/0.7Da-UIS3", "SWATH-25Da/25ppm-UIS3","SWATH-25ppm/25ppm-UIS3"]):
    sns.set_palette("rocket", n_colors = 8)
    colours = iter(sns.color_palette("rocket", n_colors=5))

    def func(x, p1,p2):
        return p1*np.log(x)+p2

    UIS_all = []

    for file in files:
        UIS = []
        for size in sizes:
            name = str(file)+"_"+str(size)+files_suffix
            df = pd.read_csv(name, header=0) 
            df = df.loc[df['UIS']!='-1'].reset_index(drop=True)
            
            splitting = df.loc[df['cas_num']==str('cas_num')] #each sample/100 has a header
            df = df.loc[(df['UIS']=='1') | (df['UIS']=='0') | (df['UIS']=='UIS')]

            UISsample = []
            count=0
            start=0
            for i,row in splitting.iterrows():
                end = i-1
                df1= df.loc[start:end] #inclusive
                df1 = df1.loc[df1['cas_num']!=str('cas_num')]
                assert len(df1)==size, len(df1)
                
                unique = ((len(df1.loc[df1['UIS'] == '1']))/len(df1))*100
                UISsample.append(unique)
                count += 1
                start = i+1

            #last one that does not have 'cas' at end
            df1= df.loc[start:]
            df1 = df1.loc[df1['cas_num']!=str('cas_num')]
            assert len(df1)==size, len(df1)
            
            unique = ((len(df1.loc[df1['UIS'] == '1']))/len(df1))*100
            UISsample.append(unique)
            count += 1

            assert len(UISsample)==100, len(UISsample)
            UIS.append(np.median(UISsample))

        d = {'UIS': UIS, 'Sizes': sizes}
        df = pd.DataFrame(data=d)
        model = smf.ols('UIS ~ np.log(Sizes)', data=df).fit()
        print(model.summary())
        
        parameters = model.params
        r2 = model.rsquared
        #param[0] = b, param[1] = a --> a(np.log(x))+b
        equation= "y = "+str(my_round(parameters[1],3))+"log(x) + "+str(my_round(parameters[0],3))+"; R2="+str(my_round(r2,3))
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
        
        plot.set_xlabel('MS Method', fontsize=12)
        plot.set_ylabel('Percentage of Unique Compounds', fontsize=12)
        plot.set_ylim(0,100)
        print(UIS)
        UIS_all.append(UIS)
    plt.show()
    print(UIS_all)

def ce_dist():
    allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra, col_energy=0)
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
    ce2 = ce2.loc[ce2['NumSpectra']!=0] #comp with no interferences
    ce2['AllCE'] = ce2['AllCE'].apply(lambda x: ast.literal_eval(x))
    ce2['unique_ce_all'] = ce2['AllCE'].apply(lambda x: Counter(list(itertools.chain.from_iterable(x))))
    ce2['unique_POCE'] = ce2['unique_ce_all'].apply(lambda x: len(x)) #number of unique POCE per comp

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
        all_settings.append(count)
        
    ce2['all_ce_settings'] =all_settings
    print(ce2['all_ce_settings'].value_counts())

    #number of POCE required to differentiate compounds
    bins = range(1,max(ce2['all_ce_settings'])+2)
    ce2 = ce2.sort_values(by=['all_ce_settings'],ascending=True)
    n, bins, patches = plt.hist(list(ce2['all_ce_settings']), bins=bins, edgecolor='black')
    ticks = [(patch._x0 + patch._x1)/2 for patch in patches]
    ticklabels = [i for i in range(1,max(ce2['all_ce_settings'])+1)]
    plt.xticks(ticks, ticklabels)
    plt.show()
    
def compare_UIS_specific(d, index):
    sns.set_palette("rocket", n_colors = 8)
    df = pd.DataFrame(data=d)
    df.index = index
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(df, cmap=cmap,  linewidths=0.1, linecolor='black', annot=True, vmin=0, vmax= max(d['UIS1']))
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()

def spectra_display(queryid, comparedid):
    allcomp, spectra = read(compounds = 'comp_df17.pkl', spectra = 'spec_df17.pkl')
    compounds_filt, spectra_filt = filter_comp(compounds_filt=allcomp, spectra=spectra)
    
    query_spectra = spectra_filt.loc[spectra_filt['spectrum_id'] == queryid]
    query_smiles = compounds_filt.loc[compounds_filt.mol_id==query_spectra.mol_id.item()].smiles.item()
    query_spectra.loc[:,'peaks'] = query_spectra['peaks'].apply(lambda x: [(a,b/(max(x,key=itemgetter(1))[1])*100) for (a,b) in x])
    compared_spectra = spectra_filt.loc[spectra_filt['spectrum_id'] == comparedid]
    compare_smiles = compounds_filt.loc[compounds_filt.mol_id==compared_spectra.mol_id.item()].smiles.item()
    compared_spectra.loc[:,'peaks'] = compared_spectra['peaks'].apply(lambda x: [(a,b/(max(x,key=itemgetter(1))[1])*100) for (a,b) in x])

    query = list(query_spectra['peaks'])[0]
    query2 = pd.DataFrame(query, columns = ['m/z', 'int'])
    compare = list(compared_spectra['peaks'])[0]
    compare2 = pd.DataFrame(compare, columns = ['m/z', 'int'], )

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
        for size in sizes:
            name = str(file)+"_UIS"+str(size)+file_suffix
            query = pd.read_csv(name)
            query = query.loc[query['UIS']!=-1]
            unique = len(query.loc[query['UIS'] == 1])
            unique = ((len(query.loc[query['UIS'] == 1]))/len(query))*100
            UIS.append(unique)
        UIS_all.append(UIS)

    df = pd.DataFrame(data=UIS_all)
    df = df.transpose() #sizes=rows

    df.columns = labels
    df.index = sizes
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

    sns.set_palette("rocket", n_colors = 3)
    sns.distplot(spectra_filt['prec_mz'], kde=False)
    plt.show()

    spectra_filt['peaks'] = [[(a,b) for (a,b) in peaklist if (b>top_n)] for peaklist in spectra_filt['peaks']]
    transitions = []

    for i, row in spectra_filt.iterrows():
      transitions.append(len(row['peaks']))

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
    morethan = ((len([i for i in transitions if i>=10]))/length)*100

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
    print(query)
    for i,comp in background.iterrows():
        print(comp)
        comp_x = allcomp.loc[allcomp.mol_id==comp.mol_id]
        spec_x = spectra.loc[spectra.spectrum_id==comp.spectrum_id]
        comp_name = comp_x.name.item()
        comp_smiles = comp_x.smiles.item()
        spec_prec_mz = spec_x.prec_mz.item()
        spec_adduct = spec_x.prec_type.item()
        print(comp_name, spec_prec_mz, spec_adduct)
        get_mol_im(comp_smiles, comp.mol_id)
    print(interferences)
    print(uis)
    return interferences, uis
