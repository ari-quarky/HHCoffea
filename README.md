# HH Coffea 
adapted from SUEP Coffea. HH analysis using [Coffea](https://coffeateam.github.io/coffea/)

## Quick start
Here are the directions to create histograms using Ultra Legacy MC. Histograms are defined in python/LQ_Producer.py. So to begin we create the enviroment in which to run our code. 

```bash
mkdir LQ
cd LQ 
git clone git@github.com:ari-quarky/HHCoffea.git
cd HHCoffea/
mkdir -p Plots/Muon/2017
singularity shell -B ${PWD} -B /afs -B /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```
## Another Way
```bash
cmsrel CMSSW_10_6_4
cd CMSSW_10_6_4/src
cmsenv
mkdir LQ
cd LQ 
git clone git@github.com:ari-quarky/HHCoffea.git
cd HHCoffea/
mkdir -p Plots/Muon/2017
singularity shell -B ${PWD} -B /afs -B /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```

## Make/Edit Histograms and Plots
If you want to add different types of histograms, adjust the selections, or adjust the weight/scale factors the exception being the Btag scaling factor, go into HHCoffea/python/LQ_Producer.py and make your edits. What reader.py does is make and get the histograms out of the NTuples and stores them for our next step. Make sure you update xsections_UL2017.yaml and UL2017_sample_reference.json when you add new NTuples into your list. The NTuples for me are stored under: /eos/user/a/argonzal/LQ_rootFiles/2017/
```bash
python3 reader.py
```

## Cross-sections and JSON Files
xsections_UL2017.yaml stores the cross-sections and branching fraction for each of the NTuples you want to run.
UL2017_sample_reference.json lists all the NTuples you want to use for your run and a set name to call the NTuples. This will be used for LQPlotter.py

## Plotting Histograms 
Once you have updated the Cross-sections and JSON Files, now you're able to add a few more detail into the histograms. Your able to add which signal is shown against the background. You can see an example of this on under LQplotter.py on this site. Using --sample_dir takes the NTuples stored from this location to make histograms. --hist_dir takes the histograms already created from running reader.py and adds more information to them such as adding legends or signal lines. --xfile place the cross-section and branching fraction file here. --outdir is the path/directory you want your final plots to go into. --year Match up to the year of the UL run, the NTuples I made are all from 2017. --channel place specific channel you want to run here (Ex: --channel muon, or --channel electron). Add --nonorm if you don't want to calculate the normalization and don't have a b-tagging scale factor made. Run without --nonorm to get normalizations and if you calculated the b-tagging scale factor. This will be called btag_weights.json. Look at the Btag section below for more on this.

The Btag scaling factor should have already been created and is in btag_weights.json so try to run without the --nonorm option.
```bash
python LQplotter.py --sample_dir /eos/user/a/argonzal/LQ_rootFiles/2017/ --hist_dir Plots/Muon/2017/ --xfile xsections_UL2017.yaml --outdir plots_2017_LQ_test/ --year 2017 --channel muon 
```

## Merge files for coffea
edit paths for input ntuples and output
```bash
python merger.py
```

## To run the producer
histograms are defined in HH_Producer.py
```bash
python3 condor_LQ_WS.py --isMC=0/1 --era=201X --infile=XXX.root
```

If you do not have the requirements set up then you can also run this through the docker container that the coffea team provides. This is simple and easy to do. You just need to enter the Singularity and then issue the command above. To do this use:
```bash
singularity shell -B ${PWD} -B /afs -B /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```

Inside the singularity shell, can run a shell script over all files. This gives output root files from coffea. There is a runner for each year data and MC (TODO update this)
```bash
./runner_2016mc.sh
./runner_2016data.sh
```

## Plotter
With Miniconda, use the configuration file to create the virtual environment 'plotting'
```bash
conda env create -f plotting_env.yml
```

Activate the environment
```bash
conda activate plotting
```

Alternatively, you can pip install the packages listed inside the yml file.

To plot run HHplotter.py. Options (python HHplotter.py --help) for input are histogram directory of files from coffea made above, input samples directory, input xsection yaml, year, muon or electron channel, output directory, option to run without background normalizations ('--nonorm'), and option to run in series (default runs in parallel). Example command:
```bash
python HHplotter.py --sample_dir /eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v3/2017/ --hist_dir 2017-v3/ --xfile /afs/cern.ch/work/v/vinguyen/private/CMSSW_10_6_4/src/PhysicsTools/MonoZ/data/xsections_2017.yaml --year 2017 --outdir plots_2017-v3 --channel muon
```

## Applying Btag Event Weight renormalization by jet bin
This must be done once per channel and for all years. It requires running the coffea script and  running the plotting script, which outputs a JSON of the renormalizations by jet bin. Then the coffea script must be run again to make histograms with the renormalizations, and the plotting script is run once more as usual.

After running coffea script the first time, get renormalizations by running the below (--btag must be run with --nonorm) for every year. By default, each year will write out to the same JSON. There's an option to overwrite this file (TODO: add channel to JSON). An example is shown below:
```bash
python LQplotter.py --sample_dir /eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v3/2017/ --hist_dir 2017-v3/ --xfile /afs/cern.ch/work/v/vinguyen/private/CMSSW_10_6_4/src/PhysicsTools/MonoZ/data/xsections_2017.yaml --outdir plots_2017-btag --year 2017 --channel muon --nonorm --btag --filter
```

Once you have the output JSON with weights for all years, run coffea producer again with option --njetw:
```bash
python3 condor_LQ_WS.py --isMC=0/1 --era=201X --njetw --infile=XXX.root
```

Now the renormalizations are applied in the histograms, and the plotting script can be run as usual.

## Btag Added Note
Added Note: If you do need to create a new Btag. I have already added --njetw into reader.py so to do the first step you would have to eliminate --njetw from a couple files. They are located in LQ_Producer.py, condor_LQ_WS.py and reader.py. njetw is located in these line, comment/uncomment them out as you will use them later on.

In python/LQ_Producer.py comment:
https://github.com/ari-quarky/HHCoffea/blob/master/python/LQ_Producer.py#L22
https://github.com/ari-quarky/HHCoffea/blob/master/python/LQ_Producer.py#L40-L47

In condor_LQ_WS.py comment:
https://github.com/ari-quarky/HHCoffea/blob/master/condor_LQ_WS.py#L23
https://github.com/ari-quarky/HHCoffea/blob/master/condor_LQ_WS.py#L68
https://github.com/ari-quarky/HHCoffea/blob/master/condor_LQ_WS.py#L74
https://github.com/ari-quarky/HHCoffea/blob/master/condor_LQ_WS.py#L85

In reader.py uncomment:
https://github.com/ari-quarky/HHCoffea/blob/master/reader.py#L30
https://github.com/ari-quarky/HHCoffea/blob/master/reader.py#L32

In reader.py comment out:
https://github.com/ari-quarky/HHCoffea/blob/master/reader.py#L33

## More Info on Selections
After you have a Btag, in the updated ntuples, you can also find the “event_category” variable. This variable is used to classify events into 4 category:
event_category==1: 2 muons, opposite sign(electric charge), isolated
event_category==2: 2 muons, opposite sign, at least one muon isnt isolated
event_category==3: 2 muons, same sign, isolated
event_category==4: 2 muons, same sign, at least one muon isnt isolated
We will use this variable for the data driven estimation (ABCD method) of QCD background. The selections (in python/LQ_Producer.py) I used are based on this, with selection "signal_btag" representive of QCD_A = event_category==1, "QCD_B" = event_category==2, "QCD_C" = event_category==3, and "QCD_D" = event_category==4

## Requirements

- Python 3
- uproot
- coffea
- HTCondor cluster

Alternatively, everything can be run through the docker container provided by the coffea team:
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest


