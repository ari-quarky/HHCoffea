"""
HH_Producer.py
Workspace producers using coffea.
"""
from coffea.hist import Hist, Bin, export1d
from coffea.processor import ProcessorABC, LazyDataFrame, dict_accumulator
from uproot3 import recreate
import numpy as np
import awkward as ak
import numpy as np
import pandas as pd
import awkward
import yaml
import xgboost as xgb


class WSProducer(ProcessorABC):
    """
    A coffea Processor which produces a workspace.
    This applies selections and produces histograms from kinematics.
    """

    histograms = NotImplemented
    selection = NotImplemented

    def __init__(self, isMC, era=2017, sample="DY", do_syst=False, syst_var='', weight_syst=False, haddFileName=None, flag=False):
        self._flag = flag
        self.do_syst = do_syst
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self._accumulator = dict_accumulator({
            name: Hist('Events', Bin(name=name, **axis))
            for name, axis in ((self.naming_schema(hist['name'], region), hist['axis'])
                               for _, hist in list(self.histograms.items())
                               for region in hist['region'])
        })
        self.outfile = haddFileName

    def __repr__(self):
        return f'{self.__class__.__name__}(era: {self.era}, isMC: {self.isMC}, sample: {self.sample}, do_syst: {self.do_syst}, syst_var: {self.syst_var}, weight_syst: {self.weight_syst}, output: {self.outfile})'

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df, *args):
        output = self.accumulator.identity()

        weight = self.weighting(df)

        features = ['met_pt','met_phi','Higgsbb_cand_pt','Higgsbb_cand_mass','HiggsZZ_cand_pt','HiggsZZ_cand_mass','Zlep_cand_pt','Zlep_cand_mass','Zjet_cand_pt','Zjet_cand_mass','leading_lep_pt','trailing_lep_pt','leading_jet_pt','leading_jet_qgl','trailing_jet_pt','trailing_jet_qgl','leading_Hbb_pt','leading_Hbb_btag','trailing_Hbb_pt','trailing_Hbb_btag','dR_l1l2','dR_l1j1','dR_l1j2']

        X = df[features]
        X = ak.to_numpy(X).tolist()
        #load BDT model
        model = xgb.XGBClassifier()
        model.load_model('BDTmodels/test')
        bdtscore=model.predict_proba(X)[:,1]
        df['BDTscore']=bdtscore


        for h, hist in list(self.histograms.items()):
            for region in hist['region']:
                name = self.naming_schema(hist['name'], region)
                selec = self.passbut(df, hist['target'], region)
                output[name].fill(**{
                    'weight': weight[selec],
                    name: df[hist['target']][selec]#.flatten()
                })

        return output

    def postprocess(self, accumulator):
        return accumulator

    def passbut(self, event: LazyDataFrame, excut: str, cat: str):
        """Backwards-compatible passbut."""
        return eval('&'.join('(' + cut.format(sys=('' if self.weight_syst else self.syst_suffix)) + ')' for cut in self.selection[cat] ))#if excut not in cut))

class HH_NTuple(WSProducer):
    histograms = {
        'Zlep_cand_mass_QCD_B': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_QCD_B',  # name to write to histogram
            'region': ['QCD_B'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
        },
        'Zlep_cand_mass_QCD_C': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
        },
        'Zlep_cand_mass_QCD_D': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_QCD_D',  # name to write to histogram
            'region': ['QCD_D'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
        },
        'Zlep_cand_mass_DYcontrol': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_DYcontrol',  # name to write to histogram
            'region': ['DYcontrol'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 120, 'lo': 20, 'hi': 140}
        },
        'Zlep_cand_mass_TTcontrol': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_TTcontrol',  # name to write to histogram
            'region': ['TTcontrol'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
        },
        'Zlep_cand_mass': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 100, 'lo': 50, 'hi': 100}
        },
        'Zjet_cand_mass': {
            'target': 'Zjet_cand_mass',
            'name'  : 'Zjet_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Zjet_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'Higgsbb_cand_mass': {
            'target': 'Higgsbb_cand_mass',
            'name'  : 'Higgsbb_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Higgsbb_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'HiggsZZ_cand_mass': {
            'target': 'HiggsZZ_cand_mass',
            'name'  : 'HiggsZZ_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'HiggsZZ_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'met_pt': {
            'target': 'met_pt',
            'name'  : 'met_pt',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'met_pt', 'n_or_arr': 60, 'lo': 0, 'hi': 600}
        },
        'dR_l1l2': {
            'target': 'dR_l1l2',
            'name'  : 'dR_l1l2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1l2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_j1j2': {
            'target': 'dR_j1j2',
            'name'  : 'dR_j1j2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_j1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_b1b2': {
            'target': 'dR_b1b2',
            'name'  : 'dR_b1b2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_b1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1b1': {
            'target': 'dR_l1b1',
            'name'  : 'dR_l1b1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1b2': {
            'target': 'dR_l1b2',
            'name'  : 'dR_l1b2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2b1': {
            'target': 'dR_l2b1',
            'name'  : 'dR_l2b1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2b2': {
            'target': 'dR_l2b2',
            'name'  : 'dR_l2b2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1j1': {
            'target': 'dR_l1j1',
            'name'  : 'dR_l1j1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1j2': {
            'target': 'dR_l1j2',
            'name'  : 'dR_l1j2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2j1': {
            'target': 'dR_l2j1',
            'name'  : 'dR_l2j1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2j2': {
            'target': 'dR_l2j2',
            'name'  : 'dR_l2j2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'h_bdtscore' : {
            'target': 'BDTscore',
            'name': 'BDTscore',
            'region': ['signal'],
            'axis': {'label': 'BDTscore','n_or_arr':20, 'lo':0., 'hi':1.}
        },
    }
    selection = {
            "signal" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 1",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "QCD_B" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 2",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "QCD_C" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 3",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "QCD_D" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 4",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "DYcontrol" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 1",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20",
                "event.Zlep_cand_mass > 80",
                "event.Zlep_cand_mass < 100"
            ],
            "TTcontrol" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 1",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20",
                "event.Zlep_cand_mass > 100",
                "event.met_pt > 100"
            ],
        }


    def weighting(self, event: LazyDataFrame):
        weight = 1.0
        try:
            weight = event.xsecscale
        except:
            return "ERROR: weight branch doesn't exist"

        if self.isMC:
#            try:
#                weight = weight * event.genWeight
#            except:
#                return "ERROR: genWeight doesn't exist"

            if "puWeight" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.puWeightUp
                else:
                    weight = weight * event.puWeightDown
            else:
                weight = weight * event.puWeight

            # PDF uncertainty
            if "PDF" in self.syst_suffix:
                try:
                    if "Up" in self.syst_suffix:
                        weight = weight * event.pdfw_Up
                    else:
                        weight = weight * event.pdfw_Down
                except:
                    pass

            #Muon SF    
            if "MuonSF" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.w_muon_SFUp
                else:
                    weight = weight * event.w_muon_SFDown
            else:
                weight = weight * event.w_muon_SF

            # Electron SF
            if "ElecronSF" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.w_electron_SFUp
                else:
                    weight = weight * event.w_electron_SFDown
            else:
                weight = weight * event.w_electron_SF

	    #Prefire Weight
            try:
                if "PrefireWeight" in self.syst_suffix:
                    if "Up" in self.syst_suffix:
                        weight = weight * event.PrefireWeight_Up
                    else:
                        weight = weight * event.PrefireWeight_Down
                else:
                    weight = weight * event.PrefireWeight
            except:
                pass

        return weight

    def naming_schema(self, name, region):
        return f'{name}{self.syst_suffix}'
