import awkward as ak
import numpy as np
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import correctionlib
from coffea import nanoevents, util
np.seterr(divide='ignore', invalid='ignore')
import glob as glob
import re
import itertools
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit


class util_binning :
    '''
    Class to implement the binning schema for QCD jets. 
    '''
    def __init__(self):
        
        self.dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        self.frac_axis = hist.axis.Regular(300, 0, 2, name="frac", label="Ratio")
        self.eta_axis = hist.axis.Variable(
            [0, 0.5, 0.8, 1.1, 1.3, 1.7, 1.9, 2.1, 2.3, 2.5, 2.8, 3.0, 3.2, 4.7], name="eta", label="$\eta$"
        ) 
        self.phi_axis = hist.axis.Regular(100, -2*np.pi, 2*np.pi, name="phi", label="$\phi$")
        self.pt_axis = hist.axis.Variable(
            [10, 20, 30, 35, 40, 45, 57, 72, 90, 120, 150, 200, 300, 400, 550, 750, 1000, 
            1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 
            4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 10000], name="pt", label=r"$p_{T}$ [GeV]"
        )
        self.rho_axis = hist.axis.Variable([0, 7.32, 13.20, 19.08, 24.95, 30.83, 36.71, 90], name="rho", label=r"$\rho$")
        self.rho_fine_axis = hist.axis.Regular(100, 0, 101, name="rho_fine", label=r"$\rho$")               
        self.npvs_axis = hist.axis.Regular(100, 0, 101, name="npvs", label="$N_{PV}$")                               
        self.npu_axis = hist.axis.Variable([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], name="npu", label="$N_{PU}$") 
        self.npu_fine_axis = hist.axis.Regular(60, 0, 120, name="npu_fine", label="$N_{PU}$")
        
        
        
def histogram_bin_extractor(output_file):
    '''
    Function for extracting the binnings from a pickle file. 
    '''
    dataset_axis = output_file.axes[0]
    
    bin_edges = []
    bin_centers = []
    bin_widths = []

    eras = []

    for i in range(len(dataset_axis)):
        era = dataset_axis[i]
        eras.append(era)
        
    for axis in output_file.axes:
        bin_edges.append(axis.edges)
        bin_centers.append(axis.centers)
        bin_widths.append(axis.widths)
    
    return eras, bin_edges, bin_centers, bin_widths


        
def gaussian_function(x, amplitude, mean, standard_dev):
    '''
    Function used to fit a pT response curve to a gaussian function.
    '''
    return amplitude * np.exp(- (x - mean)**2 / (2. * standard_dev**2))



class GaussianParametersReader:
    '''
    Class for reading csv files and extracting their content.
    '''
    def __init__(self, era, eta_ranges, rho_ranges):
        self.era = era
        self.eta_ranges = eta_ranges
        self.rho_ranges = rho_ranges
        self.dfs = []
        self.filenames = []

    def read_gaussian_parameters(self):
        '''
        Function for reading everything from a csv file along with its filename. 
        '''
        for eta_range in self.eta_ranges:
            eta_1, eta_2 = eta_range
            for rho_range in self.rho_ranges:
                rho_1, rho_2 = rho_range
                filepath = f"gaussian_fit_files/{self.era}/eta_{eta_1}-{eta_2}/gaussian_parameters_{self.era}_eta_{eta_1}-{eta_2}_rho_{rho_1}-{rho_2}.csv"
                df = pd.read_csv(filepath)
                self.dfs.append(df)
                self.filenames.append(os.path.basename(filepath))

    def extract_sigma_and_err(self):
        '''
        Function for extracting the experimental JER and JER errors of the dfs obtained from THE read_gaussian_parameters() method; the square roots in the uncertainties are already accounted for. 
        '''
        pt_bin_list = []
        sigmas_list = []
        sigma_errs_list = []
        for df, filename in zip(self.dfs, self.filenames):
            pt_bins = []
            sigmas = []
            sigma_errs = []
            for pt_bin, mean, sigma, mean_err, sigma_err in zip(df['PT Center Bin Value'], df['Mean'], df['Standard Deviation'], df['Var(Mean)'], df['Var(Standard Deviation)']):
                
                jer = abs(sigma) / mean
                jer_err_values = np.sqrt( (np.sqrt(sigma_err) / mean)**2 + ((sigma * np.sqrt(mean_err)) / (mean)**2)**2)
                
                pt_bins.append(pt_bin)
                sigmas.append(jer)
                sigma_errs.append(jer_err_values)
            pt_bin_list.append(pt_bins)
            sigmas_list.append(sigmas)
            sigma_errs_list.append(sigma_errs)

        return pt_bin_list, sigmas_list, sigma_errs_list, self.filenames, self.dfs

    

def plot_unfitted_jer(sigmas, sigma_errs, filenames, pt_bin_centers, row, col):
    '''
    Function for plotting the JER without any fits or residual plots.
    '''
    
    markers = ['o', 's', '^', 'D', 'x', '*', '>']
    marker_size = 5 
    
    for i, (sigma, sigma_err, filename) in enumerate(zip(sigmas, sigma_errs, filenames)):
        year = filename.split('_')[filename.split('_').index('parameters') + 1]
        eta_range = filename.split('_')[filename.split('_').index('eta') + 1].split('-')
        rho_range = filename.split('_')[filename.split('_').index('rho') + 1].split('-')
        eta_labels = f"{eta_range[0]} < $\eta$ < {eta_range[1]}"
        rho_labels = f"{rho_range[0]} < $\u03c1$ < {float(rho_range[1].replace('.csv', '')):.2f}"
        marker = markers[i % len(markers)]
        
        plt.errorbar(pt_bin_centers, sigma, yerr=sigma_err, fmt=marker, markersize=marker_size, markerfacecolor='none', label=rho_labels)
    
    plt.title(year, fontsize=20)
    plt.xlabel(r"$p_T$ [GeV]", fontsize=20)
    plt.ylabel("JER", fontsize=20)
    plt.xscale('log')
    plt.xlim(left=10)
    legend = plt.legend(title=eta_labels, title_fontsize='10', fontsize=10)
    legend.get_title().set_fontweight('bold')
    plt.gca().tick_params(axis='both', direction='in', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.show()
    
    

class JERCalculator:
    '''
    Class for extracting parameters from JER txt files, and calculating the expected JER from these files.
    '''
    def __init__(self, file_path, eta_min, eta_max, rho_min, rho_max):
        self.file_path = file_path
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.parameters = self.extract_parameters()

    def extract_parameters(self):
        '''
        Function for extracting the p0, p1, p2, and p3 parameters from  JER txt files. 
        '''
        parameters = []
        with open(self.file_path, 'r') as file:
            header = file.readline().split()
            for line in file:
                data = line.split()
                jet_eta_min = float(data[0])
                jet_eta_max = float(data[1])
                rho_min_val = float(data[2])
                rho_max_val = float(data[3])
                num_cols_left = int(data[4])
                pt_min = float(data[5])
                pt_max = float(data[6])

                if (self.eta_min <= jet_eta_min <= self.eta_max) and \
                   (self.eta_min <= jet_eta_max <= self.eta_max) and \
                   (self.rho_min <= rho_min_val <= self.rho_max) and \
                   (self.rho_min <= rho_max_val <= self.rho_max):
                    parameters.extend([float(param) for param in data[7:]])
        parameters = [parameters[i:i+4] for i in range(0, len(parameters), 4)]

        return parameters
    
    def compute_jer(self, pt_list):
        '''
        Function for computing the JER from the extracted parameters.
        '''
        jer_list = []
        for params in self.parameters:
            jer_sublist = []
            for pt in pt_list:
                jer = np.sqrt(params[0] * np.abs(params[0]) / (pt * pt) + 
                              params[1] * params[1] * np.power(pt, params[3]) + 
                              params[2] * params[2])
                jer_sublist.append(jer)
            jer_list.append(jer_sublist)
        return jer_list

    

def jer_function(pt, parameter_0, parameter_1, parameter_2, parameter_3):
    '''
    Function for fitting a plot of the gaussian widths to the JER function. The parameters come from the corresponding txt file. 
    '''
    return np.sqrt(parameter_0 * np.abs(parameter_0) / (pt * pt) + parameter_1 * parameter_1 * np.power(pt, parameter_3) + parameter_2 * parameter_2)



def plot_jer(pt_bin_list, sigmas_list, sigma_errs_list, filenames_list, jer_parameters_list, lower_xlimit, upper_xlimit, lower_pt_bin, upper_pt_bin):
    '''
    Function for plotting the gaussian widths, their fit to a JER function, and the residuals between the two.
    '''
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, .25])
    axs0 = fig.add_subplot(gs[0])
    axs1 = fig.add_subplot(gs[1])
    
    markers = ['o', 's', '^', 'D', 'x', '*', '>']
    marker_size = 5

    for i, (sigmas, sigma_errs, filename, initial_guess) in enumerate(zip(sigmas_list, sigma_errs_list, filenames_list, jer_parameters_list)):
        year = f"{filename.split('_')[filename.split('_').index('parameters') + 1]}"
        eta_labels = f"{filename.split('_')[filename.split('_').index('eta') + 1].split('-')[0]} < $\eta$ < {filename.split('_')[filename.split('_').index('eta') + 1].split('-')[1]}"
        rho_labels = f"{filename.split('_')[filename.split('_').index('rho') + 1].split('-')[0]} < $\u03c1$ < {float(filename.split('_')[filename.split('_').index('rho') + 1].split('-')[1].replace('.csv', '')):.2f}"
        marker = markers[i % len(markers)]
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        
        axs0.errorbar(
            pt_bin_list[lower_pt_bin:upper_pt_bin], sigmas[lower_pt_bin:upper_pt_bin], yerr=sigma_errs[lower_pt_bin:upper_pt_bin], 
            fmt=marker, markersize=marker_size, markerfacecolor='none', label=rho_labels, color=color
        )
        
        axs0.set_title(year, fontsize=20)
        
        popt, pcov = curve_fit(jer_function, xdata=pt_bin_list[lower_pt_bin:upper_pt_bin], ydata=sigmas[lower_pt_bin:upper_pt_bin], p0=initial_guess, maxfev=1000000)
        
        axs0.plot(pt_bin_list[lower_pt_bin:upper_pt_bin], jer_function(pt_bin_list[lower_pt_bin:upper_pt_bin], *popt), color=color)
        
        axs1.set_xscale('log') 
        
        experimental = sigmas[lower_pt_bin:upper_pt_bin]
        accepted = jer_function(pt_bin_list[lower_pt_bin:upper_pt_bin], *popt)
        error = ((experimental - accepted) / accepted)
        error_uncertainty = sigma_errs[lower_pt_bin:upper_pt_bin] / accepted

        axs1.errorbar(pt_bin_list[lower_pt_bin:upper_pt_bin], error, yerr=error_uncertainty, marker=marker, markersize=marker_size, markerfacecolor='none', color=color, linestyle='none')
        
    axs0.set_ylabel("JER", fontsize=20)
    axs0.set_xscale('log')
    axs0.set_xlim(lower_xlimit, upper_xlimit)
    legend = axs0.legend(title=eta_labels, title_fontsize='15', fontsize=15)
    legend.get_title().set_fontweight('bold')
        
    axs1.set_xlabel(r"$p_T$ [GeV]", fontsize=20)
    axs1.set_ylabel("(point - fit) / fit", fontsize=20)
    axs1.set_xscale('log')
    axs1.set_xlim(lower_xlimit, upper_xlimit)
    axs1.set_ylim(-0.25, 0.25)
                        
    axs1.axhline(y=0, color='k', linestyle='-')
    axs1.axhline(y=0.1, color='k', linestyle='--')
    axs1.axhline(y=-0.1, color='k', linestyle='--')
    axs0.tick_params(axis='both', direction='in', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    axs0.yaxis.set_minor_locator(AutoMinorLocator())
    axs1.tick_params(axis='both', direction='in', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    plt.subplots_adjust(hspace=0.15)
    plt.show()

        
    
def plot_jer_vs_eta(eta_bin_list, sigma_lists, sigma_errs_lists, labels_list, total_filenames_list):
    '''
    Function for obtaining plots of the JER as a function of eta in pT and rho bins.
    '''
    markers = ['o', 's', '^', 'D', 'x', '*', '>']
    marker_size = 5 

    for i, (sigmas, sigma_errs, pt_label, filenames) in enumerate(zip(sigma_lists, sigma_errs_lists, labels_list, total_filenames_list)):
        year = f"{filenames.split('_')[filenames.split('_').index('parameters') + 1]}"
        rho_labels = f"{filenames.split('_')[filenames.split('_').index('rho') + 1].split('-')[0]} < $\u03c1$ < {float(filenames.split('_')[filenames.split('_').index('rho') + 1].split('-')[1].replace('.csv', '')):.2f}"
        marker = markers[i % len(markers)]
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        
        plt.errorbar(eta_bin_list, sigmas, yerr=sigma_errs, fmt=marker, markersize=marker_size, markerfacecolor='none', label=pt_label)
        plt.plot(eta_bin_list, sigmas, color=color)
        
    plt.title(f"{year}", fontsize=20)
    plt.xlabel(r"$\eta$", fontsize=20)
    plt.ylabel("JER", fontsize=20)
    plt.text(0.05, 0.95, rho_labels, transform=plt.gca().transAxes, ha='left', va='top', fontsize=15)
    legend = plt.legend(title=f"$p_T$ Bins", loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, title_fontsize='15', fontsize=15)
    legend.get_title().set_fontweight('bold')
    plt.gca().tick_params(axis='both', direction='in', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.show()

    
    
def plot_jer_vs_rho(rho_bin_list, sigma_lists, sigma_errs_lists, labels_list, total_filenames_list):
    '''
    Function for obtaining plots of the JER as a function of rho in pT and eta bins.
    '''
    markers = ['o', 's', '^', 'D', 'x', '*', '>']
    marker_size = 5 

    for i, (sigmas, sigma_errs, pt_label, filenames) in enumerate(zip(sigma_lists, sigma_errs_lists, labels_list, total_filenames_list)):
        year = f"{filenames[0].split('_')[filenames[0].split('_').index('parameters') + 1]}"
        eta_labels = f"{filenames[0].split('_')[filenames[0].split('_').index('eta') + 1].split('-')[0]} < $\eta$ < {filenames[0].split('_')[filenames[0].split('_').index('eta') + 1].split('-')[1]}"
        marker = markers[i % len(markers)]
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        
        plt.errorbar(rho_bin_list, sigmas, yerr=sigma_errs, fmt=marker, markersize=marker_size, markerfacecolor='none', label=pt_label)
        plt.plot(rho_bin_list, sigmas, color=color)
        
    plt.title(f'{year}', fontsize=20)
    plt.xlabel(r"$\rho$", fontsize=20)
    plt.ylabel("JER", fontsize=20)
    plt.text(0.05, 0.95, eta_labels, transform=plt.gca().transAxes, ha='left', va='top', fontsize=15)
    legend = plt.legend(title=f"$p_T$ Bins", loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, title_fontsize='15', fontsize=15)
    legend.get_title().set_fontweight('bold')
    plt.gca().tick_params(axis='both', direction='in', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.show()
    