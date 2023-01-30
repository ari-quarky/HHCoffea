import os
import re

from coffea.processor import run_uproot_job, futures_executor

from python.LQ_Producer import *
from python.SumWeights import *

import uproot3 as uproot
import argparse

import time

parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--jobNum', type=int, default=1, help="")
parser.add_argument('--era', type=str, default="2018", help="")
parser.add_argument('--doSyst', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--nevt', type=str, default=-1, help="")
parser.add_argument('--channel', type=int, default=1, help="")
parser.add_argument('--njetw', type=str, default=None, help="")

options = parser.parse_args()

"""
def inputfile(nanofile):
    tested = False
    forceaaa = False
    pfn = os.popen("edmFileUtil -d %s" % (nanofile)).read()
    pfn = re.sub("\n", "", pfn)
    print((nanofile, " -> ", pfn))
    if (os.getenv("GLIDECLIENT_Group", "") != "overflow" and
            os.getenv("GLIDECLIENT_Group", "") != "overflow_conservative" and not
            forceaaa):
        if not tested:
            print("Testing file open")
            testfile = uproot.open(pfn)
            if testfile:
                print("Test OK")
                nanofile = pfn
            else:
                if "root://cms-xrd-global.cern.ch/" not in nanofile:
                    nanofile = "root://cms-xrd-global.cern.ch/" + nanofile
                forceaaa = True
        else:
            nanofile = pfn
    else:
        if "root://cms-xrd-global.cern.ch/" not in nanofile:
            nanofile = "root://cms-xrd-global.cern.ch/" + nanofile
    return nanofile
"""

#options.dataset='QCD'

pre_selection = ""

if float(options.nevt) > 0:
    print((" passing this cut and : ", options.nevt))
    pre_selection += ' && (Entry$ < {})'.format(options.nevt)

pro_syst = []
ext_syst = []

modules_era = []

modules_era.append(LQ_NTuple(isMC=options.isMC, era=int(options.era), do_syst=1, syst_var='', channel=options.channel, sample=options.dataset, njetw=options.njetw))#,
#modules_era.append(LQ_NTuple(isMC=options.isMC, era=int(options.era), do_syst=1, syst_var='', channel=options.channel, sample=options.dataset))#, njetw=options.njetw))#,
if options.isMC and options.doSyst==1:
   for sys in pro_syst:
       for var in ["Up", "Down"]:
           modules_era.append(LQ_NTuple(options.isMC, str(options.era), do_syst=1,
                                    syst_var=sys + var, sample=options.dataset))#, njetw=options.njetw))#,

   
   for sys in ext_syst:
       for var in ["Up", "Down"]:
           modules_era.append(
               LQ_NTuple(
                   options.isMC, str(options.era),
                   do_syst=1, syst_var=sys + var,
                   weight_syst=True,
                   sample=options.dataset,
                   njetw=options.njetw#,
               )
           )

for i in modules_era:
    print("modules : ", i)

print("Selection : ", pre_selection)
tstart = time.time()
f = uproot.recreate("tree_%s_WS.root" % str(options.jobNum))
for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 4},
        chunksize=500000
    )
    for h, hist in output.items():
        f[h] = export1d(hist)


modules_gensum = []

if options.isMC:
    modules_gensum.append(GenSumWeight(isMC=options.isMC, era=int(options.era), do_syst=1, syst_var='',sample=options.dataset))

    
    for instance in modules_gensum:
        output = run_uproot_job(
            {instance.sample: [options.infile]},
            treename='Runs',
            processor_instance=instance,
            executor=futures_executor,
            executor_args={'workers': 4},
            chunksize=500000000
        )
        for h, hist in output.items():
            f[h] = export1d(hist)

