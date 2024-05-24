import awkward as ak
import numpy as np
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from distributed.diagnostics.plugin import UploadDirectory
from collections import defaultdict
import os
import re
from coffea.analysis_tools import PackedSelection
from python.smp_utils import *
from python.cms_utils import *
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory



class QCDProcessor(processor.ProcessorABC):
        
    def __init__(self):
        
        ###################################
        ### Defining the Histogram Axes ###
        ###################################
        
        binning = util_binning()
        dataset_axis = binning.dataset_axis
        frac_axis = binning.frac_axis
        eta_axis = binning.eta_axis
        #phi_axis = binning.phi_axis
        pt_axis = binning.pt_axis
        
        rho_axis = binning.rho_axis
        rho_fine_axis = binning.rho_fine_axis
        npvs_axis = binning.npvs_axis
        #npu_axis = binning.npu_axis
        npu_fine_axis = binning.npu_fine_axis
        
        ######################################
        ### Defining the Histogram Objects ###
        ######################################
        
        h_responses_histogram_rho = hist.Hist(dataset_axis, frac_axis, eta_axis, pt_axis, rho_axis, storage="weight", label="Counts")
        #h_responses_histogram_pu = hist.Hist(dataset_axis, frac_axis, eta_axis, pt_axis, npu_axis, storage="weight", label="Counts")
        h_corrections_histogram = hist.Hist(dataset_axis, rho_fine_axis, npvs_axis, npu_fine_axis, storage="weight", label="Counts")
        
        cutflow = {}
        
        self.hists = {
            "responses_histogram_rho":h_responses_histogram_rho,
            #"responses_histogram_pu":h_responses_histogram_pu,
            "corrections_histogram":h_corrections_histogram,
            "cutflow":cutflow,
        }
        
    @property
    def accumulator(self):
        return self.hists
    
    def process(self, events):
        
        dataset = events.metadata['dataset']
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = defaultdict(int)
        
        ####################################################
        ### Applying Cuts to Jet Kinematic Distributions ###
        ####################################################
        
        ### Vertex and JetId Masks
        
        vtx_mask = np.abs(events.GenVtx.z - events.PV.z) < 0.2

        events = events[vtx_mask]
        gen_jets = events.GenJet
        reco_jets = events.Jet
        
        id_mask = reco_jets.jetId > 0
        
        reco_jets = reco_jets[id_mask]
        events = events[ak.num(reco_jets, axis=1) > 0]
        gen_jets = events.GenJet
        reco_jets = events.Jet
        
        ### Keeping Three Leading Jets 
        
        gen_jets = gen_jets[:, :3]
        reco_jets = gen_jets.nearest(reco_jets, threshold=0.2)
        pt_response = reco_jets.pt / gen_jets.pt
        
        ### Final Masks
        
        sel_1 = ~ak.is_none(reco_jets, axis=1)

        reco_jets = reco_jets[sel_1]
        gen_jets = gen_jets[sel_1]
        pt_response = pt_response[sel_1]

        sel_2 = ak.num(pt_response) > 2

        reco_jets = reco_jets[sel_2]
        gen_jets = gen_jets[sel_2]
        pt_response = pt_response[sel_2]
        
        ###########################################
        ### Applying Cuts to Pileup Observables ###
        ###########################################
        
        ### Observable Initialization and Mask Application
        
        n_reco_vtx = events.PV.npvs
        n_pileup = events.Pileup.nPU
        rho = events.fixedGridRhoFastjetAll
        pu_nTrueInt = events.Pileup.nTrueInt
        
        n_reco_vtx = n_reco_vtx[sel_2]
        n_pileup = n_pileup[sel_2]
        rho = rho[sel_2]
        pu_nTrueInt = pu_nTrueInt[sel_2]
        
        ### Broadcasting Across reco_jets
        
        n_reco_vtx = ak.broadcast_arrays(n_reco_vtx, reco_jets.pt)[0]
        n_pileup = ak.broadcast_arrays(n_pileup, reco_jets.pt)[0]
        rho = ak.broadcast_arrays(rho, reco_jets.pt)[0]
        pu_nTrueInt = ak.broadcast_arrays(pu_nTrueInt, reco_jets.pt)[0]
        
        ### Pileup Weights
        
        puWeight = GetPUSF(dataset, np.array(ak.flatten(pu_nTrueInt)))
        
        ##############################
        ### Filling the Histograms ###
        ##############################
        
        self.hists["responses_histogram_rho"].fill(dataset=dataset, frac=ak.flatten(pt_response), eta=np.abs(ak.flatten(gen_jets.eta)), pt=ak.flatten(gen_jets.pt), rho=ak.ravel(rho), weight=puWeight)
        #self.hists["responses_histogram_pu"].fill(dataset=dataset, frac=ak.flatten(pt_response), eta=np.abs(ak.flatten(gen_jets.eta)), pt=ak.flatten(gen_jets.pt), npu=ak.ravel(n_pileup), weight=puWeight)
        self.hists["corrections_histogram"].fill(dataset=dataset, npvs=ak.ravel(n_reco_vtx), npu_fine=ak.ravel(n_pileup), rho_fine=ak.ravel(rho), weight=puWeight)
        
        return self.hists
    
    def postprocess(self, accumulator):
        return accumulator
    