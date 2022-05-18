"""
LQ_Producer.py
based on HH_Producer.py
Workspace producers using coffea.
"""
from coffea.hist import Hist, Bin, export1d
from coffea.processor import ProcessorABC, LazyDataFrame, dict_accumulator
from uproot3 import recreate
import numpy as np

class WSProducer(ProcessorABC):
    """
    A coffea Processor which produces a workspace.
    This applies selections and produces histograms from kinematics.
    """

    histograms = NotImplemented
    selection = NotImplemented

    def __init__(self, isMC, era=2017, sample="DY", do_syst=False, syst_var='', channel=0, weight_syst=False, haddFileName=None, flag=False):
        self._flag = flag
        self.do_syst = do_syst
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.channel = channel
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
        return f'{self.__class__.__name__}(era: {self.era}, isMC: {self.isMC}, sample: {self.sample}, channel: {self.channel}, do_syst: {self.do_syst}, syst_var: {self.syst_var}, weight_syst: {self.weight_syst}, output: {self.outfile})'

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df, *args):
        output = self.accumulator.identity()

        weight = self.weighting(df)

        x = 0
        df['eventcount']=x

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

class LQ_NTuple(WSProducer):

    zlep_bin = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 132, 146, 164, 184, 209, 239, 275, 318, 370, 432]

    histograms = {
        'Nvertex':{
            'target': 'PV_npvsGood', # name of variables from tree
            'name'  : 'Nvertex',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Nvertex', 'n_or_arr': 70, 'lo':0, 'hi': 70}
        },
        'Z_cand_mass': {
            'target': 'Z_mass',
            'name'  : 'Z_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Z mass (GeV)', 'n_or_arr': 80, 'lo':50, 'hi': 130}
        },
        'Z_cand_pt': {
            'target': 'Z_pt',
            'name'  : 'Z_pt',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Z #it{p}_{T} (GeV)', 'n_or_arr': zlep_bin}
        },
    }
    selection = {
            "signal" : [
                "event.lep_category    == self.channel", ## 1 = dimuons, 2 = dielectrons
                "event.event_category    == 2", ## 2= OS leptons, 1= SS leptons
                "event.met_filter == 1 ",
                "event.ngood_jets > 0" 
            ],
        }


    def weighting(self, event: LazyDataFrame):
        weight = 1.0
        try:
            weight = event.xsecscale
        except:
            return "ERROR: weight branch doesn't exist"

        if self.isMC:
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

            # Electron SF # we dont have these guys yet
#            if "ElecronSF" in self.syst_suffix:
#                if "Up" in self.syst_suffix:
#                    weight = weight * event.w_electron_SFUp
#                else:
#                    weight = weight * event.w_electron_SFDown
#            else:
#                weight = weight * event.w_electron_SF

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

            #TriggerSFWeight
#            if "TriggerSFWeight" in self.syst_suffix:
#                if "Up" in self.syst_suffix:
#                    weight = weight * event.TriggerSFWeightUp
#                else:
#                    weight = weight * event.TriggerSFWeightDown
#            else:
#                weight = weight * event.TriggerSFWeight

        return weight

    def naming_schema(self, name, region):
        return f'{name}{self.syst_suffix}'
