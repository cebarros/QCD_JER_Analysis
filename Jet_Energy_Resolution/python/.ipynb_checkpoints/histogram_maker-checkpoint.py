import awkward as ak
import numpy as np
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import glob
import pickle
from dask.distributed import Client
import datetime
from python.histogram_lib import *



def qcd_jets_maker(prependstr=None):
    
    prefix = prependstr if prependstr else "root://xcache/"
    filedir = "../samples/"
    filestr = "flatPU_JMENano_%s.txt"
    eras = ['2016', '2016APV', '2017', '2018']
    fileset = {}
    
    for era in eras:
        filename = filedir + filestr % era
        with open(filename) as f:
            files = [prefix + i.rstrip() if prefix else i.rstrip() for i in f if not i.startswith("#")]
            fileset[era] = files
            
    run = processor.Runner(
        executor=processor.FuturesExecutor(compression=None, workers=2),
        schema=NanoAODSchema,
        skipbadfiles=True,
        #maxchunks=10,
        #chunksize=1000000,
    )
        
    out = run(
        fileset=fileset,
        treename="Events",
        processor_instance=QCDProcessor(),
    )

    date_today = datetime.datetime.now().strftime("%m-%d")
    fname_out = "pkl_files/QCD_pt_response_" + date_today + ".pkl"

    with open(fname_out, "wb") as f:
        pickle.dump(out, f)
