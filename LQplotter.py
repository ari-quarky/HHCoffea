#!/usr/bin/env/ python
#Plotter for HH variables
#%config InlineBackend.figure_format = 'retina'
import sys

import glob
import json
import logging
import os
from pathlib import Path
import re
import yaml

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pandas as pd
#import seaborn as sns
from tqdm import tqdm

try:
    import uproot3 as uproot
    import uproot3_methods as uproot_methods
except:
    import uproot
    import uproot_methods
logging.getLogger().setLevel(logging.INFO)

COLORS=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
#COLORS=list(sns.color_palette('Set2').as_hex())
#plt.style.use('physics.mplstyle')

LUMI = {'2016':36, '2017':41.5, '2018': 59.8}

def cleanup(samples_directory, hist_dir, xsections):
    sample_files = set([Path(f).stem
                        for f in glob.glob(os.path.join(samples_directory, '*.root'))])
    histogram_files = set([Path(f).stem.replace('_WS_selections','')
                           for f in glob.glob(os.path.join(hist_dir, '*.root'))
                           ])
    overlap = sample_files.intersection(histogram_files)
    data_kwds = [ 'SingleElectron', 'SingleMuon']
    to_keep = []
    for f in overlap:
        if any([kwd in f for kwd in data_kwds]):
            to_keep.append(f)
        else:
            if f in xsections:
                to_keep.append(f)
    sample_paths = [os.path.join(samples_directory, f'{f}.root')
                    for f in to_keep]
    hist_paths = [os.path.join(hist_dir, f'{f}_WS_selections.root')
                  for f in to_keep]
    return sample_paths, hist_paths

def get_histograms(hist_paths, year, channel):
    histograms={}

    logging.info('Loading histograms into memory.')
    for filename in tqdm(hist_paths):

        if channel == 'muon': skip = ['SingleElectron']
        elif channel == 'electron': skip = ['SingleMuon']
        else:
            raise ValueError(f'channel must be \'muon\' or \'electron\', not {channel}')

        if any([s in filename for s in skip]): continue
        if year == '2016':
            pass
        elif year == '2017':
            pass
        elif year == '2018' :
            pass

        print('filename: ',filename)
        samplename = Path(filename).stem.split('_WS')[0]

        f_sample = uproot.open(filename)
        histogram_sample = f_sample.allitems( filterclass=lambda cls: issubclass(cls, uproot_methods.classes.TH1.Methods))
        if not histogram_sample:
            continue
        histograms[samplename] = histogram_sample

    logging.info('Finished loading histograms.')
    logging.info(f'Number of samples remaining: {len(histograms)}')
    return histograms

def xs_scale(xsections, year, ufile, proc):
#    xsec  = xsections[proc]["xsec"]
#    xsec *= xsections[proc]["kr"]
#    xsec *= xsections[proc]["br"]
#    xsec *= 1000.0
    #print (proc, xsec)
#    assert xsec > 0, "{} has a null cross section!".format(proc)
    scale = LUMI[year]*1000/ufile["Runs"].array("genEventSumw").sum()
#    print('scale: ',scale)
#    print('proc: ',proc)
    return scale

def get_xsections(infile):
    with open(infile) as f:
        try:
            xsections = yaml.safe_load(f.read())
        except yaml.YAMLError as exc:
            print(exc)
    return xsections

def get_normalizations(sample_paths, xsections, histogram_names, year):
    norm_dict = {}

    logging.info('Obtaining normalizations.')

    for fn in tqdm(sample_paths):
#        if 'TTTo' in fn : continue
#        if 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8' in fn : continue
        if os.path.isfile(fn):
            _proc = os.path.basename(fn).replace(".root","")
            _file = uproot.open(fn)
            if ("SingleElectron") in fn:
                _scale = 1
            elif ("SingleMuon") in fn:
                _scale = 1
            else:
                _scale  = xs_scale(xsections, year, ufile=_file, proc=_proc)
            for idx, name in enumerate(histogram_names):
                if name == Path(fn).stem:
                    norm_dict[name] = _scale
                    break
    logging.info('Finished obtaining normalizations.')
    logging.info(f'Number normalizations in memory: {len(norm_dict)}.')
# These normalizations scale by lumi/genweight. The xsec is already applied in the plots from coffea
    return(norm_dict)

def rebin( a, newshape ):
        '''Rebin an array to a new shape.
        '''
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        return a[tuple(indices)]

def rebin_factor( a, newshape ):
        '''Rebin an array to a new shape.
        newshape must be a factor of a.shape.
        '''
        assert len(a.shape) == len(newshape)
        assert not sometrue(mod( a.shape, newshape ))

        slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
        return a[slices]

def normalize_event_yields(event_yields, normalizations, file_to_category, var=False):
    categorized_yields = {}

    for sample in event_yields:
        category = file_to_category[sample]
        if category not in categorized_yields:
            try:
                if var:
                    categorized_yields[category] = (normalizations[sample]**2) * event_yields[sample]
                else:
                    categorized_yields[category] = normalizations[sample] * event_yields[sample]
            except:
                if sample not in normalizations and sample not in event_yields:
                    print(f'{sample} not in both')
                elif sample not in event_yields:
                    print(f'{sample} not in event_yields')
                elif sample not in normalizations:
                    print(f'{sample} not in normalizations')
        else:
            if sample not in normalizations and sample not in event_yields:
                print(f'{sample} not in both')
            elif sample not in event_yields:
                print(f'{sample} not in event_yields')
            elif sample not in normalizations:
                print(f'{sample} not in normalizations')
            if var:
                categorized_yields[category] += (normalizations[sample]**2) * event_yields[sample]
            else:
                categorized_yields[category] += normalizations[sample] * event_yields[sample]

    return categorized_yields

def get_bins_and_event_yields(histograms, normalizations, year, filter_categories=False, print_yields=True):
    with open(f'UL{year}_sample_reference.json', 'r') as infile:
        file_to_category = json.load(infile)

    categories = set(file_to_category.values())
    if filter_categories:
        for category in ['QCD', 'NonResVBF', 'Radion', 'Graviton', 'NonRes', 'NonResSM']:
            categories.remove(category)

    df_dict = {}
    df_dict['sample_name'] = []
    df_dict['bins'] = []
    df_dict['var'] = []
    for category in categories:
        df_dict[category] = []
        df_dict['var_'+category] = []
    df_dict['Other'] = []
    df_dict['var_Other'] = []

    logging.info('Getting bins and event yields.')

    arb_key = next(iter(histograms))
    for idx, (name, roothist) in enumerate(tqdm(histograms[arb_key])):
        name = name.decode("utf-8")
        name = name.replace(";1", "")

        # TODO make this more robust
        if "genEventSumw" == name:
            continue
        event_yields = {}
        event_variances = {}
        # TODO Only one for loop here.
        for key, value in histograms.items():
            event_yields[key] = np.abs(value[idx][1].numpy())[0]
            event_variances[key] = np.abs(value[idx][1].variances)
        
        output = normalize_event_yields(event_yields, normalizations, file_to_category)
        output_var = normalize_event_yields(event_variances, normalizations, file_to_category, var=True)
        output['Other'] = output['SingleTop'] + output['WJets']
        output_var['Other'] = output_var['SingleTop'] + output_var['WJets']
        total_var = output_var['Other'] + output_var['DY'] + output_var['TT'] + output_var['VV']
        for category in output:
            df_dict[category].append(output[category])
            df_dict['var_'+category].append(output_var[category])
        df_dict['var'].append(total_var)
        df_dict['bins'].append(roothist.numpy()[1])
        df_dict['sample_name'].append(name)

        if print_yields:
            if year == '2016':
                y = ['GluGluToHHTo2B2ZTo2L2J_node_SM_TuneCUETP8M1_PSWeights_13TeV-madgraph-pythia8',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-260_narrow_13TeV-madgraph-v2',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-600_narrow_13TeV-madgraph-v2',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-1000_narrow_13TeV-madgraph-v2',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-3000_narrow_TuneCUETP8M1_PSWeights_13TeV-madgraph-pythia8',
                     'TT_TuneCUETP8M2T4_13TeV-powheg-pythia8',
                     'DYToLL_0J_13TeV-amcatnloFXFX-pythia8',
                     'DYToLL_1J_13TeV-amcatnloFXFX-pythia8',
                     'DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8',
                     'DYToLL_2J_13TeV-amcatnloFXFX-pythia8'
                 ]
            if year == '2017':
                y = [
                    'DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                    'DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                    'DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                    #'UL17_LQToBMu_M-400_single_TuneCP2_13TeV-madgraph-pythia8'
                ]
            if year == '2018':
                y = ['GluGluToHHTo2B2ZTo2L2J_node_SM_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-260_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-600_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-1000_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                     'GluGluToRadionToHHTo2B2ZTo2L2J_M-3000_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                     'DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                     'DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                     'DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                     'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8',
                     'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8',
                     'TTToHadronic_TuneCP5_13TeV-powheg-pythia8',
                     'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8']
            for idx, y in enumerate(y):
                print(f'Yield {y}: {event_yields[y]*normalizations[y]}')

    logging.info('Finished getting bins and event yields.')
    for key in df_dict:
        print(key, len(df_dict[key]))
    return pd.DataFrame(df_dict)
    #print(df_dict)

def btag_ratio(all_event_yields, year, filepath, overwrite):
    names_for_njets = np.array(['ngood_jets', 'ngood_jets_btagSF'])
    df_subset = (all_event_yields.set_index('sample_name')
                                 .loc[names_for_njets]
                                 .reset_index())

    njets, njets_btagSF = *[df_subset.iloc[idx] for idx in range(df_subset.shape[0])],
    sum_func = lambda x: (x['DY'] + x['TT'] + x['VV'] + x['Other'])

    logging.info('Calculating btag renormalizations by jet bin.')

    sum_njets = sum_func(njets)
    sum_njets_btagSF = sum_func(njets_btagSF)

    sum_high_njet = sum(sum_njets[10:])
    sum_high_njet_btagSF = sum(sum_njets_btagSF[10:])
    ratio_high_njet = sum_high_njet/sum_high_njet_btagSF

    btag_ratio = sum_njets/sum_njets_btagSF
    btag_ratio[10:]=[ratio_high_njet]*len(btag_ratio[10:])
    btag_ratio[np.isnan(btag_ratio)] = -999
    print(f'btag Ratio: {btag_ratio}')

    weights = {str(idx): ratio for idx, ratio in enumerate(btag_ratio)}
    btag_weights = {'year': int(year), 'weights': weights}

    if overwrite:
        mode = 'w'
    else:
        mode = 'a'
    with open(filepath, mode) as f:
        f.write(str(btag_weights).replace('\'', '\"') + '\n')

    logging.info('Saved btag renormalizations to JSON.')

def btag_ratio_plot(all_event_yields, year, outdir=''):
    names_for_njets = np.array(['ngood_jets', 'ngood_jets_btagSF', 'ngood_jets_nobtagSF'])
    df_subset = (all_event_yields.set_index('sample_name')
                                 .loc[names_for_njets]
                                 .reset_index())

    njets, njets_btagSF, njets_nobtagSF = *[df_subset.iloc[idx] for idx in range(df_subset.shape[0])],
    sum_func = lambda x: (x['DY'] + x['TT'] + x['VV'] + x['Other'])

    sum_njets = sum_func(njets)
    sum_njets_btagSF = sum_func(njets_btagSF)
    sum_njets_nobtagSF = sum_func(njets_nobtagSF)

    sum_high_njet = sum(sum_njets[10:])
    sum_high_njet_btagSF = sum(sum_njets_btagSF[10:])
    sum_high_njet_nobtagSF = sum(sum_njets_nobtagSF[10:])

    sum_njets[10:]=[sum_high_njet]*len(sum_njets[10:])
    sum_njets_btagSF[10:]=[sum_high_njet_btagSF]*len(sum_njets_btagSF[10:])
    sum_njets_nobtagSF[10:]=[sum_high_njet_nobtagSF]*len(sum_njets_nobtagSF[10:])

    logging.info('Making btag ratio plot.')

    bins = njets['bins']
    x = np.arange(len(bins)-1)

    fig, axarr = plt.subplots(2, dpi=150, figsize=(6, 5), sharex=True,
                                  gridspec_kw={'hspace': 0.08, 'height_ratios': (0.8,0.2)},
                                  constrained_layout=False)
    upper = axarr[0]
    lower = axarr[1]

    ns, bins, patches = upper.hist(x=x, bins=bins, weights=sum_njets_nobtagSF,
                          histtype='step', linestyle=('solid'), color='black',
                          linewidth=1, label=['no SF']
                          )
    ns1, bins, patches = upper.hist(x=x, bins=bins, weights=sum_njets_btagSF,
                          histtype='step', linestyle=('solid'), color='blue',
                          linewidth=1, label=['btag SF']
                          )
    ns2, bins, patches = upper.hist(x=x, bins=bins, weights=sum_njets,
                          histtype='step', linestyle=('solid'), color='crimson',
                          linewidth=1, label=['btag SF and nJet corr.']
                          )

    upper.set_xlim([3.5, 10.5])
    upper.set_ylabel("Events/bin", y=1, ha='right')
    upper.legend()

    lower.step(bins[1:], ns1 / ns,
               color='blue',linewidth=1
              )
    lower.step(bins[1:], ns2 / ns,
               color='crimson',linewidth=1
              )

    lower.set_ylim(0.5, 1.5)
    lower.set_xlabel('n Jets', x=1, ha='right')
    lower.set_ylabel('Ratio (/no SF)')

    fig.savefig(os.path.join(outdir, f'njets_btag_ratios_{year}.png'), bbox_inches='tight')

def estimate_background(all_event_yields, tol=1e-16, maxiter=100, disp=False):
    names_for_bkgd_est = np.array(['Zlep_cand_mass_DYcontrol_QCD_C', 'Zlep_cand_mass_TTcontrol_QCD_C',
                          'Zlep_cand_mass_DYcontrol', 'Zlep_cand_mass_TTcontrol',
                          'Zlep_cand_mass_QCD_B','Zlep_cand_mass_QCD_D'])
    df = all_event_yields.set_index('sample_name')
    df_subset = df.loc[names_for_bkgd_est]
    df_subset = df_subset.reset_index()

    dy_c, tt_c, dy, tt, qcd_b, qcd_d = *[df_subset.iloc[idx] for idx in range(df_subset.shape[0])],


    residual_func = lambda x, y, z, a, b: (x['Data']
                                        - (x['DY'] * y + a + x['TT'] * z + b + x['VV'] + x['Other']))

    def optimizer(dDY, dTT, dDY_qcd_b, dTT_qcd_b, dDY_qcd_d, dTT_qcd_d):
        counter = 0
        qcd_norm = 100
        dy_norm = 100
        tt_norm = 100
        error = 100

        dy_data = dy['Data']
        tt_data = tt['Data']
        
        while (error > tol) and (counter < maxiter):
            tt_c_shape = residual_func(tt_c, dy_norm, tt_norm, 0, 0)
            updated_tt_norm = np.sum((tt_data - (qcd_norm * tt_c_shape + dy_norm*tt['DY'] + tt['VV'] + tt['Other'])) / np.sum(tt['TT']))

            dy_c_shape = residual_func(dy_c, dy_norm, updated_tt_norm, 0, 0)
            updated_dy_norm = np.sum((dy_data - (qcd_norm * dy_c_shape + updated_tt_norm*dy['TT'] + dy['VV'] + dy['Other'])) / np.sum(dy['DY']))

            qcd_b_val = np.sum(residual_func(qcd_b, updated_dy_norm, updated_tt_norm, dDY_qcd_b, dTT_qcd_b))
            qcd_d_val = np.sum(residual_func(qcd_d, updated_dy_norm, updated_tt_norm, dDY_qcd_d, dTT_qcd_d))

            updated_qcd_norm = qcd_b_val / qcd_d_val
            error = np.sqrt((updated_qcd_norm - qcd_norm)**2 + np.abs(updated_dy_norm - dy_norm)**2 + np.abs(updated_tt_norm - tt_norm)**2)

            #print(f'DY norm: {dy_norm}, TT norm: {tt_norm}, fBD: {qcd_norm}, err: {error}, counter: {counter}')
            #print(f'DY norm: {dy_norm}, TT norm: {tt_norm}, fBD: {qcd_norm}, err: {error}, tol: {tol}, counter: {counter}')
            #print(f'DY norm: {dy_norm}, TT norm: {tt_norm}, fBD: {qcd_norm}, err: {current_err}, counter: {counter}')

            qcd_norm = updated_qcd_norm
            dy_norm = updated_dy_norm
            tt_norm = updated_tt_norm

            counter += 1

        if disp:
            print(f'qcd_norm: {qcd_norm:.3f} \n'
                  f'dy_norm: {dy_norm:.3f} \n'
                  f'tt_norm: {tt_norm:.3f} \n'
                  f'Error: {error} \n'
                  f'Converged: {error <= tol} \n'
                  f'Iterations: {counter}')

        return qcd_norm, dy_norm, tt_norm 
        
        

    qcd_norms = []
    dy_norms = []
    tt_norms = []

    errDY = np.sqrt(dy['var_DY'])
    errTT = np.sqrt(tt['var_TT'])
    errDY_qcd_b = np.sqrt(qcd_b['var_DY'])
    errTT_qcd_b = np.sqrt(qcd_b['var_TT'])
    errDY_qcd_d = np.sqrt(qcd_d['var_DY'])
    errTT_qcd_d = np.sqrt(qcd_d['var_TT'])
    
    for _ in range (0, 1000): 
        randDY = np.random.normal(0, 1, size=dy['Data'].shape)
        randTT = np.random.normal(0, 1, size=tt['Data'].shape)
        randDY_qcd_b = np.random.normal(0, 1, size=qcd_b['Data'].shape)
        randTT_qcd_b = np.random.normal(0, 1, size=qcd_b['Data'].shape)
        randDY_qcd_d = np.random.normal(0, 1, size=qcd_d['Data'].shape)
        randTT_qcd_d = np.random.normal(0, 1, size=qcd_d['Data'].shape)

        dDY = np.multiply(errDY,randDY)
        dTT = np.multiply(errTT,randTT)
        dDY_qcd_b = np.multiply(errDY_qcd_b,randDY_qcd_b)
        dTT_qcd_b = np.multiply(errTT_qcd_b,randTT_qcd_b)
        dDY_qcd_d = np.multiply(errDY_qcd_d,randDY_qcd_d)
        dTT_qcd_d = np.multiply(errTT_qcd_d,randTT_qcd_d)

        qcd_norm, dy_norm, tt_norm = optimizer(dDY, dTT, dDY_qcd_b, dTT_qcd_b, dDY_qcd_d, dTT_qcd_d)

        qcd_norms.append(qcd_norm)
        dy_norms.append(dy_norm)
        tt_norms.append(tt_norm)

    print(f'dy_norm: {np.mean(dy_norms):.3g} +/- {np.std(dy_norms):.1g} \n'
          f'tt_norm: {np.mean(tt_norms):.3g} +/- {np.std(tt_norms):.1g} \n'
          f'qcd_norm: {np.mean(qcd_norms):.3g} +/- {np.std(qcd_norms):.1g} \n')

    norms = (np.mean(qcd_norms), np.mean(dy_norms), np.mean(tt_norms))

    return norms

def data_mc_residual(x, norm1=1, norm2=1):
    return (x['Data'] - (x['DY']  * norm1 + x['TT'] * norm2 + x['VV'] + x['Other']))

def scale_cregions (df, qcd_norm, dy_norm, tt_norm):
    df = df.copy()
    c_samples_bool = df['sample_name'].str.contains('QCD_C')
    c_samples_to_drop = df[c_samples_bool & ~df['sample_name'].str.contains('Zlep_cand_mass_QCD_C')]['sample_name']
    c_samples = df[c_samples_bool]['sample_name']
    df_subset = df[c_samples_bool].set_index('sample_name')
    residuals = (data_mc_residual(df_subset, dy_norm, tt_norm)) * qcd_norm
    df['QCD_estimate'] = df.apply(lambda x: np.zeros(shape=x['Data'].shape[0]), axis=1)
    df = df.set_index('sample_name')
    df.loc[residuals.index.str.replace('_QCD_C', ''), 'QCD_estimate'] = residuals.values
    df = df.drop(c_samples_to_drop).reset_index()
    return df

def new_plotting(event_yields, bkgd_norm, year, channel, outdir='', print_yields=False):
    fig, axarr = plt.subplots(2, dpi=150, figsize=(6, 5), sharex=True,
                              gridspec_kw={'hspace': 0.05, 'height_ratios': (0.8,0.2)},
                              constrained_layout=False)
    upper = axarr[0]
    lower = axarr[1]

    if print_yields:
        print('dy norm: ',bkgd_norm[1])
        print('tt norm: ',bkgd_norm[2])
        print('DY yield: ', event_yields['DY'])
        print('TT yield: ', event_yields['TT'])

    event_yields['DY'] *= bkgd_norm[1]
    event_yields['TT'] *= bkgd_norm[2]
    Signal = event_yields['Signal_LQ_Mass400']
    Signal1 = event_yields['Signal_LQ_Mass600']
    Signal2 = event_yields['Signal_LQ_Mass700']
    Signal3 = event_yields['Signal_LQ_Mass800']    

    Signal4 = event_yields['Signal_LQ_Mass900']

    mc_categories = ['DY', 'TT', 'VV', 'QCD_estimate']
    MC = event_yields[mc_categories].sum()
    Data = event_yields['Data']
    #LQ = event_yields['LQ']
    Other = event_yields['Other']
    name = event_yields['sample_name']
    bins = event_yields['bins']
    #print(LQ)
    MC += Other

    # The first bin has a value of 0 and will give a warning.
    ratio = Data/MC
    binc = bins[:-1] + np.diff(bins) * 0.5
    binup = bins[1:]
    xerr = np.diff(bins)*0.5

    #Dots on plot for Data
    upper.errorbar(binc, Data, xerr = None, yerr = np.sqrt(np.abs(Data)), fmt = 'o',
                   zorder=10, color='black', label='Data', markersize=3)  #changed to np.abs
    #Dots on plot for LQ              
    #upper.errorbar(binc, LQ, xerr = None, yerr = np.sqrt(np.abs(LQ)), fmt = 'o',
    #               zorder=10, color='red', label='LQ', markersize=3) 
    all_weights = np.vstack([event_yields['VV'],
                             event_yields['Other'],
                             event_yields['QCD_estimate'],
                             event_yields['DY'],
                             event_yields['TT']]).transpose()
    all_x = np.vstack([binc] * all_weights.shape[1]).transpose()

    COLORMAP = {'VV': COLORS[0],
                'Other': COLORS[1],
                'DY': COLORS[2],
                'TT': COLORS[3],
                'QCD': COLORS[4],
                'QCD_estimate':COLORS[5]}

    labels = ['VV', 'Other', 'QCD', 'DY', 'TT']
    plotting_colors = [COLORMAP[s] for s in labels]

    upper.hist(x=all_x, bins=bins, weights=all_weights,
               histtype='stepfilled', edgecolor='black', zorder=1,
               stacked=True, color=plotting_colors, label=labels)

    upper.stairs(
           edges= bins,
           values= Signal,
           # hatch="///",
           label="LQ M400",
           #label="VBF cv,c2v,c3=1 (1pb)",
           facecolor="red",
           linewidth=1,
           color="red",
    )

    upper.stairs(
           edges= bins,
           values= Signal1,
           # hatch="///",
           label="LQ M600", # coupling 0.1",
           #label="VBF cv,c2v,c3=1 (1pb)",
           facecolor="orchid",
           linewidth=1,
           color="orchid",
    )

    upper.stairs(
           edges= bins,
           values= Signal2,
           # hatch="///",
           label="LQ M700", #coupling 0.3",
           #label="VBF cv,c2v,c3=1 (1pb)",
           facecolor="green",
           linewidth=1,
           color="green",
    )

    upper.stairs(
           edges= bins,
           values= Signal3,
           # hatch="///",
           label="LQ M800", #coupling 0.5",
           #label="VBF cv,c2v,c3=1 (1pb)",
           facecolor="blue",
           linewidth=1,
           color="blue",
    )

#    upper.stairs(
#           edges= bins,
#           values= Signal4,
#           # hatch="///",
#           label="LQ M900",
#           #label="VBF cv,c2v,c3=1 (1pb)",
#           facecolor="orange",
#           linewidth=1,
#           color="orange",
#    )

    upper.fill_between(binup, MC - np.sqrt(event_yields['var']), MC + np.sqrt(event_yields['var']), step='pre', hatch='///', alpha=0, zorder=2, label="MC Stat Err")

    upper.set_yscale("log")
    upper.set_ylim([0.01, 1000000])

    #Set parameters that will be used to make the plots prettier
    max_y = upper.get_ylim()[1]
    max_x = max(bins)
    min_x = min(bins)
    x_range = max_x - min_x
    lower_label = min_x - x_range*0.05
    upper_label = max_x - x_range*0.35


    #X and Y labels (Do not use the central matplotlob default)
    upper.set_ylabel("Events/bin", y=1, ha='right')
    upper.tick_params(axis='both', which='major', direction='in',
                      bottom=True, right=False, top=False, left=True,
                      color='black')
    upper.tick_params(axis='both', which='minor', direction='in',
                      bottom=True, right=False, top=False, left=True,
                      color='black')
    lower.set_xlabel(name, x=1, ha='right')
    lower.set_ylabel("Data/MC", fontsize = 10)
    lower.set_ylim(0.5, 1.5)
    yerr = np.abs(np.sqrt(Data) / MC)  #changed to np.abs

    chi2 = 0
    nBins = 0
    ratio_check = np.isfinite(ratio)
    for indx, r in enumerate(ratio):
        if ratio_check[indx] and yerr[indx] > 0:
            chi2 = chi2 + ((r-1)*(r-1))/(yerr[indx]*yerr[indx])
            nBins = nBins + 1

 
    lower.errorbar(binc, ratio, yerr = yerr, marker = '.', color = 'black', linestyle ='none')
    lower.plot([min_x, max_x],[1,1],linestyle=':', color = 'black')
    lower.xaxis.set_minor_locator(AutoMinorLocator())
    lower.yaxis.set_minor_locator(AutoMinorLocator())
    lower.tick_params(axis='both', which='major', direction='in', length=4,
                      bottom=True, right=False, top=False, left=True,
                      color='black')
    lower.tick_params(axis='both', which='minor', direction='in', length=2,
                      bottom=True, right=False, top=False, left=True,
                      color='black')

    lower.grid(visible=True, which='both', axis='y', linestyle=':')
    lower.grid(visible=True, which='major', axis='x', linestyle=':')


    cms = upper.text(
        lower_label, max_y*1.08, u"CMS $\it{Preliminary}$",
        fontsize=16, fontweight='bold',
    )

    upper.text(
        upper_label, max_y*1.08, f'{LUMI[year]:.1f} fb$^{{-1}}$ (13 TeV)',
        fontsize=14,
    )

    if channel == 'electron':
        text_channel = r'ee '+u'channel'
    elif channel == 'muon':
        text_channel = r'$\mu \mu$ '+u'channel'

    upper.text(
        lower_label*0.95, max_y*0.25, text_channel,
        fontsize=14,
    )


    upper.text(
        lower_label*0.95,max_y*0.1,r'$\chi^{2}$/ndf = '+f'{chi2:.2f}/{nBins} = {chi2/nBins:.2f}',
        fontsize=8,
    )
    print("Signal: ",Signal)
    upper.legend()
    fig.savefig(os.path.join(outdir, f'{name}_{year}.png'), bbox_inches='tight')
    plt.close()
    #plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vivan\'s plotting tool')
    parser.add_argument('--sample_dir', type=str, required=False,
                        default='/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v1/2016/',
                        help='Directory containing sample files.')
    parser.add_argument('--hist_dir', type=str, required=False,
                        default='/eos/user/v/vinguyen/coffeafiles/2016-fixed-rename/',
                        help='Directory containing histogram files.')
    parser.add_argument('--xfile', type=str, required=False,
                        default='/afs/cern.ch/work/v/vinguyen/private/CMSSW_10_6_4/src/PhysicsTools/MonoZ/data/xsections_2016.yaml',
                        help='File containing cross sections.')
    parser.add_argument('--year', type=str, required=False,
                        default='2016',
                        help='Run year')
    parser.add_argument('--series', action='store_true', required=False,
                        help='Make plots in series.')
    parser.add_argument('--outdir', type=str, required=False, default=None,
                        help='Path to save plots.')
    parser.add_argument('--nonorm', action='store_true', required=False,
                        help='Turns off using background normalizations.')
    parser.add_argument('--filter', dest='filter_categories', action='store_true', required=False,
                        help='Filters QCD, NonResVBF, Radion, Graviton, NonRes, NonResSM categories.')
    parser.add_argument('--channel', type=str, choices=['muon', 'electron'], required=True,
                        help='Muon or Electron channel')
    parser.add_argument('--btag_filename', type=str, required=False, default='btag_weights',
                        help='File containing btag weights by jet bin')
    parser.add_argument('--btag', action='store_true', required=False,
                        help='Calculates btag weights by jet bin')
    parser.add_argument('--btag_overwrite', action='store_true', required=False,
                        help='Overwrite btag weights file.')
    parser.add_argument('--yields', action='store_true', required=False,
                        help='Print yields for select samples.')
    args = parser.parse_args()


    if args.btag and not args.nonorm:
        raise ValueError('Cannot compute btag weights with normalizations.')

    if args.outdir is not None and not os.path.isdir(args.outdir):
        print(f'{args.outdir} does not exist. Making new directory.')
        os.mkdir(args.outdir)
    if args.outdir is None:
        outdir = ''
    else:
        outdir = args.outdir

    logging.info(f'Histogram directory: {args.hist_dir}')
    logging.info(f'Sample directory: {args.sample_dir}')
    logging.info(f'Cross-section file: {args.xfile}')
    logging.info(f'Year: {args.year}')

    xsections = get_xsections(args.xfile)
    sample_paths, hist_paths = cleanup(args.sample_dir, args.hist_dir, xsections)
    histograms = get_histograms(hist_paths, args.year, args.channel)
    normalizations = get_normalizations(sample_paths, xsections, list(histograms.keys()), args.year)
    #print(histograms)
    df = get_bins_and_event_yields(histograms, normalizations, args.year, filter_categories=args.filter_categories, print_yields=args.yields)
    
    if args.nonorm:
        df['QCD_estimate'] = df.apply(lambda x: np.zeros(shape=x['Data'].shape[0]), axis=1)
        bkgd_norm = np.array([1.0, 1.0, 1.0])
       # btag_ratio_plot(df, args.year, outdir=outdir)
        if args.btag:
            btag_path = os.path.join(os.getcwd(), Path(args.btag_filename).stem + '.jsonl')
            btag_ratio(df, args.year, btag_path, args.btag_overwrite)
    else:
        bkgd_norm = estimate_background(df)
        df = scale_cregions(df, *bkgd_norm)
    print(df)
    logging.info('Making plots.')

    if not args.series:
        num_cpus = os.cpu_count()
        batch_size = 1 #len(all_bins) // num_cpus + 1
        (Parallel(n_jobs=num_cpus, batch_size=batch_size)
         (delayed(new_plotting)(df.iloc[rowidx], bkgd_norm, args.year, args.channel, outdir=outdir, print_yields=args.yields)
         for rowidx in range(df.shape[0])))
    else:
        for rowidx in range(df.shape[0]):
            new_plotting(df.iloc[rowidx], bkgd_norm, args.year, args.channel, outdir=outdir, print_yields=args.yields)
    
    logging.info(f'Finished making plots and saved to {outdir}.')

if __name__ == '__main__':
	main()
