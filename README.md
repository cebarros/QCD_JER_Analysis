# Files for Creating Jet Energy Correction Plots

The raw jet kinematics and observables of our files can be viewed in the `raw_jets.ipynb` notebook. This file is then scaled-up using Coffea in `raw_jets_processor.ipynb`, where a pkl file containing all the histograms of our dataset is produced. 

Next we apply cuts/masks to our analysis, which can be seen in the `qcd_jets.ipynb` notebook; the goal of this file is to obtain a $p_T$ response curve. As before, this is scaled-up to include all our dataset `qcd_jets_processor.ipynb`. 

The `qcd_jets_processor.ipynb` notebook produces histograms using the ROOT files in `/samples`, and corrects them using the pileup weights found in `/data`; the histograms are then dumped into pkl files.

The notebook `jer_computations.ipynb` uses the pkl files produced in `qcd_jets_processor.ipynb` to produce $p_T$ response curves, and then fits them to gaussian functions. The means and widths, and their respective errors are then put in a CSV file for analyzing.

Lastly, the `jer_plotting.ipynb` file is used to plot the JER curves as functions of the jet $p_T$ for all datasets. 

# Running the Files

### Running on CoffeaCasa

To run on CoffeaCasa, simply access the site **[coffea-casa](https://coffea.casa/hub/login)** and sign in using either your x-certificate or CERN single sign-on credentials.

### Running on LPC

For running on LPC, we follow the instructions from **[lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue)**. First SSH to cmslpc by replacing `<username>` with your credential:

```
kinit <username>@FNAL.GOV
ssh -L localhost:8888:localhost:8888  <username>@cmslpc-sl7.fnal.gov
```

Next you must obatin a voms ticket, make a working area, and clone this repository to that area:

```
voms-proxy-init -voms cms
mkdir working_directory
cd working_directory
git clone https://github.com/cebarros/Jet_Analysis.git
```

We can now run the following commands to create a Singularity apptainer shell with a coffea environment:

```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
./shell
jupyter-lab
```

If successful, you should have an output with a URL simlar to:

```
http://127.0.0.1:8888/lab?token=37193e107b0cbe947108fcaef2d6ecf3d507c2306bbae0ea
```

Copy this URL and paste it into your browser to start the analysis.
