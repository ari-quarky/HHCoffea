import ROOT
import os, glob
from array import array
import sys
import uproot3 as uproot
import numpy as np
import pandas as pd 
import vector
from matplotlib import pyplot as plt
import awkward as ak
vector.register_awkward()

#inFileName = "/eos/cms/store/mc/RunIISummer20UL17NanoAODv9/LQToBMu_M-600_single_TuneCP2_13TeV-madgraph-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/2550000/4F7E9260-11A1-2341-ACEB-818CA1A3BF8A.root"
#inFile = ROOT.TFile.Open(inFileName ,"READ")
#tree = inFile.Get("Events")
#entry = tree.GetEntries()

# numerically stable implementation
@np.vectorize
def msq2(px1, py1, pz1, px2, py2, pz2, m1, m2):
    p1_sq = px1 ** 2 + py1 ** 2 + pz1 ** 2
    p2_sq = px2 ** 2 + py2 ** 2 + pz2 ** 2
    m1_sq = m1 ** 2
    m2_sq = m2 ** 2
    x1 = m1_sq / p1_sq
    x2 = m2_sq / p2_sq
    x = x1 + x2 + x1 * x2
    a = angle(px1, py1, pz1, px2, py2, pz2)
    cos_a = np.cos(a)
    if cos_a >= 0:
        y1 = (x + np.sin(a) ** 2) / (np.sqrt(x + 1) + cos_a) 
    else:
        y1 = -cos_a + np.sqrt(x + 1) 
    y2 = 2 * np.sqrt(p1_sq * p2_sq)
    return m1_sq + m2_sq + y1 * y2

# numerically stable calculation of angle
def angle(x1, y1, z1, x2, y2, z2):
    # cross product
    cx = y1 * z2 - y2 * z1
    cy = x1 * z2 - x2 * z1
    cz = x1 * y2 - x2 * y1
    
    # norm of cross product
    c = np.sqrt(cx * cx + cy * cy + cz * cz)
    
    # dot product
    d = x1 * x2 + y1 * y2 + z1 * z2
    
    return np.arctan2(c, d)


f = uproot.open("/eos/cms/store/mc/RunIISummer20UL17NanoAODv9/LQToBMu_M-600_single_TuneCP2_13TeV-madgraph-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/2550000/4F7E9260-11A1-2341-ACEB-818CA1A3BF8A.root")
#f = uproot.open("/eos/cms/store/mc/RunIISummer20UL17NanoAODv9/LQToBMu_M-600_single_TuneCP2_13TeV-madgraph-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/40000/53E373DF-2483-5B4A-B11F-9E3693AFEDBE.root")
#t = f.get("Events")
t = f["Events"]
#tree.keys()
#t = tree.arrays()
entry = t.GetEntries()
Jet_pt = t["Jet_pt"].array()
Muon_pt = t["Muon_pt"].array()
Jet_eta = t["Jet_eta"].array()
Muon_eta = t["Muon_eta"].array()
Jet_phi = t["Jet_phi"].array()
Muon_phi = t["Muon_phi"].array()
Jet_mass = t["Jet_mass"].array()
Muon_mass = t["Muon_mass"].array()
 
jet_muon_mass = np.concatenate(Jet_mass, Muon_mass)


#branches = t["nJet"]["nMuon"]["Jet_pt"]["Jet_eta"]["Jet_phi"]["Muon_pt"]["Muon_eta"]["Muon_phi"]

two_muons_mask = t['nMuon'] == 2
one_jet_mask = t['nJet'] == 1
N=0

#for(entry in t):
#    if(t['nJet'] == 1 and t['nMuon'] == 2):
#        muon_p4 = ak.zip({'pt': t['Muon_pt'].array(), 'eta': t['Muon_eta'].array(), 'phi': t['Muon_phi'].array(), 'mass': t['Muon_mass'].array()})
#        jet_p4 = ak.zip({'pt': t['Jet_pt'].array(), 'eta': t['Jet_eta'].array(), 'phi': t['Jet_phi'].array(), 'mass': t['Jet_mass'].array()})
        #jet_p4_pt = Jet_p4[entry]
        #muon_p4_pt 
        #N=N+1


muon_p4 = ak.zip({'pt': t['Muon_pt'].array(), 'eta': t['Muon_eta'].array(), 'phi': t['Muon_phi'].array(), 'mass': t['Muon_mass'].array()})
jet_p4 = ak.zip({'pt': t['Jet_pt'].array(), 'eta': t['Jet_eta'].array(), 'phi': t['Jet_phi'].array(), 'mass': t['Jet_mass'].array()})
#print(muon_p4.mass)
#print(jet_p4)
#jet_p4 = np.array([(Jet_pt),(Jet_eta),(Jet_phi),(Jet_mass)])
#muon_p4 = np.array([(Muon_pt),(Muon_eta),(Muon_phi),(Muon_mass)])

jet_m_p4 = jet_p4[one_jet_mask][two_muons_mask]
muon_j_p4 = muon_p4[two_muons_mask][one_jet_mask]
#print(ak.num(jet_m_p4))
#print(ak.num(muon_j_p4))
#mass_mu = vector.muon_p4.mass
#mass_jet vector.jet_p4.mass

#first_jet_p4 = ak.flatten(jet_p4[:, 0], axis=None)
#second_muon_p4 = ak.flatten(muon_p4[:, :1], axis=None)
#sum = ak.sum()
#print(first_jet_p4)
#print(second_muon_p4)
#fjet_p4 = ak.to_numpy(first_jet_p4)
#fmuon_p4 = ak.to_numpy(second_muon_p4)
#ak.broadcast_arrays(jet_p4,muon_p4)
#print(fjet_p4)
#print(fmuon_p4)

plt.hist(jet_muon_mass)
#sum_p4 = jet_p4 + muon_p4
#plt.hist(jet_muon_mass, bins=np.logspace(np.log10(0.1), np.log10(1000), 200))
plt.xlabel('LQ invariant mass [GeV]')
plt.ylabel('Number of LQ events')
#plt.xscale('log')
#plt.yscale('log')
plt.savefig("test.png")
#mass = 2*Muon.pt*Jet.pt*(np.cosh(Muon.eta-Jet.eta)) - np.cos(Muon.phi - Jet.phi)



#LQ_coupling = t["Jet_pt"].array()
#LQ_mass = t["LHEPart_mass"].array()

#print(LQ_coupling)
#print(LQ_mass)


#a = np.asarray([ LQ_coupling ])
#np.savetxt("LQ_coupling_values.csv", a, delimiter=" , ")

#b = np.asarray([ LQ_mass ])
#np.savetxt("LQ_mass_values.csv", b, delimiter=" , ")


#What I use to save into file
#pd.DataFrame(LQ_coupling).to_csv("LQ_coupling_values.csv")
#pd.DataFrame(LQ_mass).to_csv("LQ_mass_values.csv")
