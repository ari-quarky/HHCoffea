import os
import re

from coffea.processor import run_uproot_job, futures_executor

#SUEP Repo Specific
from python.Tracks import *

import argparse
import uproot3 as uproot
from coffea.hist import Hist, Bin, export1d
from coffea.nanoevents import NanoEventsFactory, BaseSchema

parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--jobNum', type=int, default=1, help="")
parser.add_argument('--era', type=str, default="2018", help="")
parser.add_argument('--doSyst', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--nevt', type=str, default=-1, help="")

options = parser.parse_args()

fileset = {
        'files': [
        options.infile
        ]
}

modules_era = []
modules_era.append(SUEP_cluster(isMC=options.isMC, era=int(options.era), do_syst=1, syst_var='', sample=options.dataset))

f = uproot.recreate("tree_%s_WS.root" % str(options.jobNum))
for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 10},
        chunksize=500000
    )
    for h, hist in output.items():
        f[h] = export1d(hist)
