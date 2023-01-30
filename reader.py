import os, glob
from array import array
import sys

indirs = [
"/eos/user/a/argonzal/LQ_rootFiles/2017"
#"/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/LQ/2017"
]

def main():
    pattern = ""
    for indir in indirs:
        for sample in os.listdir(indir):
            print(" -- start -- " + sample)        
            if "Run2016" in sample:
                is_MC = " --isMC=0"
            elif "Run2017" in sample:
                is_MC = " --isMC=0"
            elif "Run2018" in sample:
                is_MC = " --isMC=0"
            else:
                is_MC = " --isMC=1"
            if "2016" in indir:
                ERA = "2016"
            if "2017" in indir:
                ERA = "2017"
            if "2018" in indir:
                ERA = "2018"
            #            continue
            njetw = "btag_weights.json"
            for channel in [1]:
                in_file = "python3 condor_LQ_WS.py {MC} --era={Era} --infile={indir}/{sample} --channel={channel} --njetw=btag_weights.json".format(sample=sample, indir=indir, MC=is_MC, Era=ERA,channel=channel, njetw=njetw)
                #in_file = "python3 condor_LQ_WS.py {MC} --era={Era} --infile={indir}/{sample} --channel={channel}".format(sample=sample, indir=indir, MC=is_MC, Era=ERA,channel=channel)
                print("command is: " + in_file )
                os.system(in_file)
                new_name = sample.replace(".root","_WS_selections.root")
                if channel == 1:
                    move_file = "mv tree_1_WS.root Plots/Muon/{Era}/{name}".format(name=new_name,Era=ERA)
                else:
                    move_file = "mv tree_1_WS.root Plots/Electron/{Era}/{name}".format(name=new_name,Era=ERA)
                os.system(move_file)
            print(" -- finished -- " + sample)
        
if __name__ == "__main__":
   main()
