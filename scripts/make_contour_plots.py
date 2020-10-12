import copy
import os
import warnings
import glob

import matplotlib.pyplot as plt
import matplotlib

import getdist
import getdist.plots
import getdist.chains
getdist.chains.print_load_details = False

import numpy as np
import scipy.stats

import camb

import anesthetic
import tensiometer
import tensiometer.gaussian_tension
import tensiometer.mcmc_tension

pi = np.pi

KCAP_PATH = "../../../../kcap/"

import sys
sys.path.append(os.path.join(KCAP_PATH, "utils"))

import process_chains
import stat_tools
import hellinger_distance_1D


def get_MAP(MAP_path, verbose=False):
    MAP_max_logpost = -np.inf
    MAPs = {}
    files = {}
    MAP_files = []
    if not isinstance(MAP_path, (list, tuple)):
        MAP_path = [MAP_path]
    for path in MAP_path:
        MAP_files += glob.glob(path)
    if len(MAP_files) == 0:
        raise RuntimeError("No MAP sample files found.")
        
    for file in MAP_files:
        try:
            MAP_chain = process_chains.load_chain(file, run_name="MAP", burn_in=0, 
                                                  ignore_inf=True, strict_mapping=False)
        except RuntimeError:
            continue
            
        if "omegamh2" not in [n.name for n in MAP_chain.getParamNames().names]:
            MAP_chain.addDerived(MAP_chain.getParams().omegam*MAP_chain.getParams().h**2,
                         name="omegamh2", label="\\Omega_{\rm m} h^2")
        if "s12" not in [n.name for n in MAP_chain.getParamNames().names] and "sigma12" in [n.name for n in MAP_chain.getParamNames().names]:
            MAP_chain.addDerived(MAP_chain.getParams().sigma12*(MAP_chain.getParams().omegamh2/0.14)**0.4,
                         name="s12", label="S_{12}")

        MAP_logpost = MAP_chain.getParams().logpost.max()
        MAP_idx = MAP_chain.getParams().logpost.argmax()

        MAPs[MAP_logpost] = {n.name : getattr(MAP_chain.getParams(), n.name)[MAP_idx] for i, n in enumerate(MAP_chain.getParamNames().names)}
        MAPs[MAP_logpost]["loglike"] = MAP_chain.loglikes[MAP_idx]
        
        files[MAP_logpost] = file
        
        if MAP_logpost > MAP_max_logpost:
            MAP = MAPs[MAP_logpost]
            MAP_chi2 = MAP_chain.loglikes[MAP_idx]
            MAP_max_logpost = MAP_logpost
    
    files = {k : files[k] for k in sorted(files, reverse=True)}
    if len(files) == 0:
        raise ValueError(f"Could not load any MAP chains using path template {MAP_path}.")
    print(f"Best MAP ({MAP_max_logpost:.4f}) in {list(files.values())[0]}")
    MAPs = {k : MAPs[k] for k in sorted(MAPs, reverse=True)}
    MAPs_array = np.array([[p for p in m.values()] for m in MAPs.values()])#[:len(MAPs)//3]
    MAPs_logpost = np.array([k for k in MAPs.keys()])#[:len(MAPs)//3]
    
    #print(MAPs)
    #print(MAPs_logpost, MAP_max_logpost, np.exp(MAPs_logpost-MAP_max_logpost))
    #print(MAPs_array)
    MAPs_wmean = np.average(MAPs_array, weights=np.exp(MAPs_logpost-MAP_max_logpost), axis=0)
    
    MAPs_wstd = np.sqrt(np.diag(np.cov(MAPs_array.T, aweights=np.exp(MAPs_logpost-MAP_max_logpost))))

    MAPs_range = np.vstack((MAPs_array.min(axis=0), MAPs_array.max(axis=0))).T
    MAPs_std = MAPs_array.std(axis=0)
    
    if verbose:
        for i, (k, v) in enumerate(MAP.items()):
            print(f"{k:<16}: {v:.3f}, range: ({MAPs_range[i][0]:.3f}, {MAPs_range[i][1]:.3f}), std: {MAPs_std[i]:.3f}, rel. err: {MAPs_std[i]/v:.3f}")
    
    MAP_std = {p : s for p, s in zip(MAP.keys(), MAPs_std)}
    MAP_wstd = {p : s for p, s in zip(MAP.keys(), MAPs_wstd)}
    MAP_wmean = {p : s for p, s in zip(MAP.keys(), MAPs_wmean)}
    MAP_wmedian = {p : stat_tools.weighted_median(MAPs_array[:,i], weights=np.exp(MAPs_logpost-MAP_max_logpost)) for i, p in enumerate(MAP.keys())}
            
    return MAP, MAP_max_logpost, MAP_chi2, MAP_std, MAP_wmean, MAP_wstd, MAP_wmedian, MAPs

def get_stats(s, CI_coverage, weights=None, params=None, MAP=None):
    # Compute chain stats
    stats = {"PJ-HPDI"            : {},
             "chain MAP"          : {},
             "PJ-HPDI n_sample"   : {},
             "M-HPDI"             : {},
             "marg MAP"           : {},
             "M-HPDI constrained" : {},
             "std"                : {},
             "mean"               : {},
             "quantile CI"        : {},
             "median"             : {}, }

    chain_data = s.getParams()
    max_lnpost_idx = (chain_data.logpost).argmax()
    ln_post_sort_idx = np.argsort(chain_data.logpost)[::-1]

    params = params or [n.name for n in s.getParamNames().names]
    
    if weights is None:
        weights = s.weights
    
    for p in ["logpost", "logprior", "loglike", "weight"]:
        if p in params:
            params.pop(params.index(p))
            
    for p in params:        
        try:
            samples = getattr(chain_data, p)
        except AttributeError:
            continue
            
        if np.any(~np.isfinite(samples)):
            warnings.warn(f"NaNs in {p}. Skipping.")
            continue
        if np.isclose(np.var(samples), 0) and not np.isclose(np.mean(samples)**2, 0):
            warnings.warn(f"Variance of {p} close to zero. Skipping.")
            continue

        try:
            PJ_HPDI, chain_MAP, PJ_HPDI_n_sample = stat_tools.find_CI(method="PJ-HPD",
                                                                      samples=samples, weights=weights,
                                                                      coverage=CI_coverage, logpost_sort_idx=ln_post_sort_idx,
                                                                      return_point_estimate=True,
                                                                      return_extras=True,
                                                                      options={"strict" : True, "MAP" : MAP[p] if MAP is not None else None
                                                                              },
                                                                     )
        except RuntimeError as e:
            warnings.warn(f"Failed to get PJ-HPDI for parameter {p}. Trying one-sided interpolation.")
            PJ_HPDI = None
            chain_MAP = samples[max_lnpost_idx]
            PJ_HPDI_n_sample = 0
            
        if PJ_HPDI is None:
            try:
                PJ_HPDI, chain_MAP, PJ_HPDI_n_sample = stat_tools.find_CI(method="PJ-HPD",
                                                                          samples=samples, weights=weights,
                                                                          coverage=CI_coverage, logpost_sort_idx=ln_post_sort_idx,
                                                                          return_point_estimate=True,
                                                                          return_extras=True,
                                                                          options={"strict" : True, 
                                                                                   "MAP" : MAP[p] if MAP is not None else None,
                                                                                   "twosided" : False},
                                                                     )
            except RuntimeError as e:
                warnings.warn(f"Failed again to get PJ-HPDI.")

        if PJ_HPDI_n_sample < 30:
            print(f"Number of PJ-HPD samples for parameter {p} less than 30: {PJ_HPDI_n_sample}")

        try:
            M_HPDI, marg_MAP, no_constraints = stat_tools.find_CI(method="HPD",
                                                                  samples=samples, weights=weights,
                                                                  coverage=CI_coverage,
                                                                  return_point_estimate=True,
                                                                  return_extras=True,
                                                                  options={"prior_edge_threshold" : 0.13}
                                                                  )
        except RuntimeError as e:
            warnings.warn(f"Failed to get M-HPDI for parameter {p}")

        stats["PJ-HPDI"][p] = PJ_HPDI
        stats["chain MAP"][p] = chain_MAP
        stats["PJ-HPDI n_sample"][p] = PJ_HPDI_n_sample

        stats["M-HPDI"][p] = M_HPDI
        stats["marg MAP"][p] = marg_MAP
        stats["M-HPDI constrained"][p] = not no_constraints

        stats["std"][p], stats["mean"][p] = stat_tools.find_CI(method="std",
                                                               samples=samples, weights=weights,
                                                               coverage=CI_coverage,
                                                               return_point_estimate=True,
                                                              )
        stats["quantile CI"][p], stats["median"][p] = stat_tools.find_CI(method="tail CI",
                                                                         samples=samples, weights=weights,
                                                                         coverage=CI_coverage,
                                                                         return_point_estimate=True,
                                                                         )

    stats["chain MAP_logpost"] = chain_data.logpost.max()
    if s.loglikes is not None:
        stats["chain MAP_loglike"] = s.loglikes[max_lnpost_idx]
    
    return stats


def load_equal_weight_chain(chain):
    chain_file = glob.glob(os.path.join(chain.chain_def["root_dir"], "chain/samples_*.txt"))[0]
    equal_weight_chain_file = glob.glob(os.path.join(chain.chain_def["root_dir"], "chain/multinest/*_post_equal_weights.dat"))[0]
    
    with open(chain_file, "r") as f:
        header = f.readlines()[:3]
        
    parameter_names = []
    chain_format = "cosmosis"
    parameter_map = "cosmomc"
    chain_params = [s.strip().lower() for s in header[0][1:].split("\t")]
    for p in chain_params:
        for mapping in process_chains.parameter_dictionary.values():
            if chain_format in mapping and mapping[chain_format] == p:
                parameter_names.append(mapping[parameter_map])
                break
        else:
            raise ValueError(f"Parameter {p} in chain does not have mapping to {parameter_map} format.")
            
    for p in ["loglike", "weight"]:
        if p in parameter_names:
            parameter_names.pop(parameter_names.index(p))
    
    print(len(parameter_names))
    print(parameter_names)
    
    equal_weight_chain = np.loadtxt(equal_weight_chain_file)
    
    samples = getdist.MCSamples(name_tag=chain.name_tag,
                                samples=equal_weight_chain[:,:-1],
                                loglikes=equal_weight_chain[:,-1],
                                names=parameter_names,
                                sampler="nested",
                                ranges=chain.ranges)
    
    return samples


def load_chains(chain_path):
    base_dir = chain_path

    chain_def = [# Systematics chains
                {"root_dir" : os.path.join(base_dir, 
                                          "systematics/multinest_blindC_EE_nE_w_no_baryon"),
                "name"     : "multinest_blindC_EE_nE_w_no_baryon",
                "label"    : "No baryon",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
        
                {"root_dir" : os.path.join(base_dir, 
                                            "systematics/multinest_blindC_EE_nE_w_zero_ho"),
                "name"     : "multinest_blindC_EE_nE_w_zero_ho",
                "label"    : "No higher order GC",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
                
                {"root_dir" : os.path.join(base_dir, 
                                            "systematics/multinest_blindC_EE_nE_w_cut_z_bin_12"),
                "name"     : "multinest_blindC_EE_nE_w_cut_z_bin_12",
                "label"    : "No z-bin 1+2",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
                
                {"root_dir" : os.path.join(base_dir, 
                                            "systematics/multinest_blindC_EE_nE_w_cut_z_bin_1"),
                "name"     : "multinest_blindC_EE_nE_w_cut_z_bin_1",
                "label"    : "No z-bin 1",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
        
                {"root_dir" : os.path.join(base_dir, 
                                            "systematics/multinest_blindC_EE_nE_w_cut_z_bin_2"),
                "name"     : "multinest_blindC_EE_nE_w_cut_z_bin_2",
                "label"    : "No z-bin 2",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
        
                {"root_dir" : os.path.join(base_dir, 
                                            "systematics/multinest_blindC_EE_nE_w_cut_z_bin_3"),
                "name"     : "multinest_blindC_EE_nE_w_cut_z_bin_3",
                "label"    : "No z-bin 3",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
                
                {"root_dir" : os.path.join(base_dir, 
                                            "systematics/multinest_blindC_EE_nE_w_cut_z_bin_4"),
                "name"     : "multinest_blindC_EE_nE_w_cut_z_bin_4",
                "label"    : "No z-bin 4",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
                
                {"root_dir" : os.path.join(base_dir, 
                                            "systematics/multinest_blindC_EE_nE_w_cut_z_bin_5"),
                "name"     : "multinest_blindC_EE_nE_w_cut_z_bin_5",
                "label"    : "No z-bin 5",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "type"     : "systematics",
                "color"    : "midnightblue"},
        
                # Blind C
                {"root_dir" : os.path.join(base_dir, 
                                            "cosmology/multinest_blindC_EE_nE_w"),
                "name"     : "multinest_blindC_EE_nE_w",
                "label"    : "$3\\times2$pt",
                "blind"    : "C",
                "probes"   : ("EE", "nE", "w"),
                "MAP_path" : [os.path.join(base_dir, "MAP/run_1/cosmology/MAP_*_blindC_EE_nE_w_Powell/chain/samples_MAP_*_blindC_EE_nE_w_Powell.txt"),
                                os.path.join(base_dir, "MAP/run_1/cosmology_Nelder_Mead/MAP_*_blindC_EE_nE_w_Nelder-Mead/chain/samples_MAP_*_blindC_EE_nE_w_Nelder-Mead.txt"),
                                os.path.join(base_dir, "MAP/run_2/cosmology_Nelder_Mead/MAP_*_blindC_EE_nE_w_Nelder-Mead/chain/samples_MAP_*_blindC_EE_nE_w_Nelder-Mead.txt"),],
                "type"     : "cosmology",
                "color"    : "red"},
        
                {"root_dir" : os.path.join(base_dir, 
                                            "cosmology/multinest_blindC_EE_nE"),
                "name"     : "multinest_blindC_EE_nE",
                "label"    : "Cosmic shear + GGL",
                "blind"    : "C",
                "probes"   : ("EE", "nE"),
                "MAP_path" : [os.path.join(base_dir, "MAP/run_1/cosmology/MAP_*_blindC_EE_nE_Powell/chain/samples_MAP_*_blindC_EE_nE_Powell.txt"),],
                "type"     : "cosmology",
                "color"    : "indigo"},
        
                {"root_dir" : os.path.join(base_dir, 
                                            "cosmology/multinest_blindC_EE_w"),
                "name"     : "multinest_blindC_EE_w",
                "label"    : "Cosmic shear + galaxy clustering",
                "blind"    : "C",
                "probes"   : ("EE", "w"),
                "MAP_path" : [os.path.join(base_dir, "MAP/run_1/cosmology/MAP_*_blindC_EE_w_Powell/chain/samples_MAP_*_blindC_EE_w_Powell.txt"),
                                os.path.join(base_dir, "MAP/run_1/cosmology_Nelder_Mead/MAP_*_blindC_EE_w_Nelder-Mead/chain/samples_MAP_*_blindC_EE_w_Nelder-Mead.txt"),
                                os.path.join(base_dir, "MAP/run_2/cosmology_Nelder_Mead/MAP_*_blindC_EE_w_Nelder-Mead/chain/samples_MAP_*_blindC_EE_w_Nelder-Mead.txt"),],
                "type"     : "cosmology",
                "color"    : "C1"},
        
                {"root_dir" : os.path.join(base_dir, 
                                            "cosmology/multinest_blindC_EE"),
                "name"     : "multinest_blindC_EE",
                "label"    : "KiDS-1000 cosmic shear",
                "blind"    : "C",
                "probes"   : ("EE",),
                "MAP_path" : [os.path.join(base_dir, "MAP/run_1/cosmology/MAP_*_blindC_EE_Powell/chain/samples_MAP_*_blindC_EE_Powell.txt")],
                "type"     : "cosmology",
                "color"    : "hotpink"},
        
                # Clustering only
                {"root_dir" : os.path.join(base_dir, 
                                            "cosmology/multinest_blindC_w"),
                "name"     : "multinest_w",
                "label"    : "BOSS galaxy clustering",
                "blind"    : "None",
                "probes"   : ("w",),
                "MAP_path" : [os.path.join(base_dir, "MAP/run_1/cosmology/MAP_*_blindC_w_Powell/chain/samples_MAP_*_blindC_w_Powell.txt"),
                                os.path.join(base_dir, "MAP/run_1/cosmology_Nelder_Mead/MAP_*_blindC_w_Nelder-Mead/chain/samples_MAP_*_blindC_w_Nelder-Mead.txt"),
                                os.path.join(base_dir, "MAP/run_2/cosmology_Nelder_Mead/MAP_*_blindC_w_Nelder-Mead/chain/samples_MAP_*_blindC_w_Nelder-Mead.txt"),],
                "type"     : "cosmology",
                "color"    : "C0"},
                
                # External data
                # Planck+3x2pt
                {"root_dir" : os.path.join(base_dir, 
                                            "cosmology/multinest_blindC_EE_nE_w_Planck"),
                "name"     : "multinest_EE_nE_w_Planck",
                "label"    : "$3\\times2$pt + Planck",
                "blind"    : "None",
                "probes"   : ("CMB", "EE", "nE", "w"),
                "MAP_path" : [os.path.join(base_dir, "MAP/run_1/cosmology_with_Planck/MAP_*_blindC_EE_nE_w_Planck_Powell/chain/samples_MAP_*_blindC_EE_nE_w_Planck_Powell.txt")],
                "type"     : "cosmology",
                "color"    : "darkgreen"},
        
                # Planck
                {"root_dir" : os.path.join(base_dir, 
                                            "cosmology/multinest_Planck"),
                "name"     : "multinest_Planck",
                "label"    : "\\textit{Planck} TTTEEE+lowE",
                "blind"    : "None",
                "probes"   : ("CMB",),
                "MAP_path" : [os.path.join(base_dir, "MAP/run_1/cosmology_with_Planck/MAP_*_blindC_Planck_Powell/chain/samples_MAP_*_blindC_Planck_Powell.txt")],
                "type"     : "cosmology",
                "color"    : "darkslategrey"},
                ]
    
    chains = {}

    CI_coverage = 0.68

    for blind in ["C", "None"]:
        print(f"Blind {blind}")
        
        chains[blind] = {}
        for c in chain_def:
            if c["blind"] == blind:
                
                print(f"  Chain {c['name']}")
                
                # Load chain
                chain_file = glob.glob(os.path.join(c["root_dir"], "chain/samples_*.txt"))
                if len(chain_file) != 1:
                    raise ValueError(f"Could not find unique chain file in {c['root_dir']}")
                chain_file = chain_file[0]
                
                value_file = os.path.join(c["root_dir"], "config/values.ini")
                
                s = process_chains.load_chain(chain_file, values=value_file, run_name=c["name"], 
                                              strict_mapping=True)
                
                if "omegamh2" not in [n.name for n in s.getParamNames().names]:
                    s.addDerived(s.getParams().omegam*s.getParams().h**2,
                                name="omegamh2", label="\\Omega_{\rm m} h^2")
                if "s12" not in [n.name for n in s.getParamNames().names]:
                    s.addDerived(s.getParams().sigma12*(s.getParams().omegamh2/0.14)**0.4,
                                name="s12", label="S_{12}")
                
                s.chain_def = c
                chains[blind][c["name"]] = s
                
                stats = {}
                if "MAP_path" in c:
                    stats["MAP"], stats["MAP_logpost"], stats["MAP_loglike"], \
                    stats["MAP_std"], stats["MAP_wmean"], stats["MAP_wstd"], \
                    stats["MAP_wmedian"], stats["MAP_runs"] = get_MAP(c["MAP_path"])
                    if "w" in c["probes"]:
                        stats["MAP highest"] = stats["MAP"]
                        stats["MAP"] = stats["MAP_wmedian"]
                    
                stats = {**stats, **get_stats(s, CI_coverage=CI_coverage, 
                                              MAP=stats["MAP"] if "MAP" in stats else None)}
                                
                chains[blind][c["name"]].chain_stats = stats

    return chains
                
def is_systematics_chain(chain):
    return chain.chain_def["type"] == "systematics"

def is_blind(chain, blind):
    return chain.chain_def["blind"] == blind

def is_probe(chain, probe):
    return chain.chain_def["probes"] == probe

def get_chain_color(chain):
    return chain.chain_def["color"]

def get_chain_label(chain):
    return chain.chain_def["label"]

def select_chains(chains, selection, global_selection=None):
    selected_chains = [None]*len(selection)
    for chain in chains.values():
        for c in chain.values():
            for i, selection_criteria in enumerate(selection):
                matches_all_selection_criteria = all([c.chain_def[k] == v for k,v in selection_criteria.items()])
                if matches_all_selection_criteria:
                    selected_chains[i] = c
    
    [selected_chains.pop(i) for i, s in enumerate(selected_chains) if s is None]
        
    return selected_chains


def plot_systematics_contours(chains, plot_settings, text_width):
    cmap = plt.get_cmap("Set2")

    blind = "C"
    chain_selection = [{"label" : "No baryon",            "blind" : blind},
                       {"label" : "No z-bin 1+2",         "blind" : blind},
                       {"label" : "No z-bin 4",           "blind" : blind},
                       {"label" : "No z-bin 5",           "blind" : blind},
                       {"label" : "No higher order GC",   "blind" : blind},
                       {"probes" : ("EE", "nE", "w"),     "blind" : blind,  "type" : "cosmology"}]
    
    chains_to_plot = select_chains(chains, chain_selection)
       
    chain_colors = [(cmap(i) if is_systematics_chain(c) else get_chain_color(c)) for i, c in enumerate(chains_to_plot)]
    chain_labels = [get_chain_label(c) for c in chains_to_plot]

    params_to_plot = ["omegam", "sigma8", "s8", "ns", "h", "a_baryon", "a_ia", "b1l", "b1h"]
        
    print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

    g = getdist.plots.get_subplot_plotter(width_inch=text_width, scaling=False,
                                          settings=copy.deepcopy(plot_settings))
    g.settings.legend_fontsize = 8
    g.settings.lab_fontsize = 8
    g.settings.axes_fontsize = 8
    
    g.triangle_plot(chains_to_plot,
                    params=params_to_plot,
                    filled_compare=True,
                    contour_colors=chain_colors,
                    legend_labels=["No baryon", "No z-bin 1+2", "No z-bin 4", "No z-bin 5", "No higher order GC", "Fiducial $3\\times2$pt"],
                    diag1d_kwargs={"normalized" : True}, 
                    param_limits={"h" : (0.64, 0.78),
                                "a_ia" : (-0.2,2.5),},
                    legend_ncol=3,
                )

    process_chains.plot_CI(g, chains_to_plot, params_to_plot, 
                        CI={c.chain_def["name"] : c.chain_stats["PJ-HPDI"] for c in chains_to_plot}, 
                        colors={c.chain_def["name"] : color for c, color in zip(chains_to_plot, chain_colors)})

    inset_ax = g.fig.add_axes([0.65, 0.55, 0.35, 0.35])
    g.plot_2d(chains_to_plot, param1="omegam", param2="s8", 
            filled=True,
            colors=chain_colors, 
            #lims=[0.1,0.6, 0.62, 0.88],
            _no_finish=True,
            ax=inset_ax)
    
    #g.fig.dpi = 300
    g.export(f"plots/3x2pt/systematics/blind_{blind}_EE_nE_w_systematics_chains.pdf")
    
def plot_3x2pt_contours_small(chains, plot_settings, column_width):
    blind = "C"
    
    chain_selection = [{"probes" : ("EE",),           "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("w",),            "blind" : "None", "type" : "cosmology"}, 
                       #{"probes" : ("EE", "nE"),      "blind" : blind,  "type" : "cosmology"},
                       #{"probes" : ("EE", "w"),       "blind" : blind,  "type" : "cosmology"}, 
                       {"probes" : ("EE", "nE", "w"),  "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("CMB",),           }, 
                      ]
    
    chains_to_plot = select_chains(chains, chain_selection)
        
    chains_to_plot = chains_to_plot# + [planck_samples]
    
    chain_colors = [get_chain_color(c) for c in chains_to_plot]
    chain_labels = [get_chain_label(c) for c in chains_to_plot]
    
    print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

    params_to_plot = ["omegam", "sigma8", "h"]
    print(f"Plotting parameters {' '.join(params_to_plot)}")

    g = getdist.plots.get_subplot_plotter(width_inch=column_width, scaling=False,
                                          settings=copy.deepcopy(plot_settings))

    g.settings.legend_fontsize = 8
    g.settings.lab_fontsize = 8
    g.settings.axes_fontsize = 8
    
    g.triangle_plot(chains_to_plot,
                    params=params_to_plot,
                    filled_compare=True,
                    contour_colors=chain_colors,
                    legend_labels=["KiDS-1000 cosmic shear", "BOSS galaxy clustering", "$3\\times2$pt",
                                   "\\textit{Planck} TTTEEE+lowE"],
                    diag1d_kwargs={"normalized" : True},
                    param_limits={"omegam" : (0.2, 0.4),
                                  "sigma8" : (0.55, 0.9),
                                  "h"      : (0.64, 0.78),
                                  "omegamh2" : (0.1, 0.2),
                                  "sigma12" : (0.6, 0.9)}
                   )

    process_chains.plot_CI(g, chains_to_plot, params_to_plot, 
                           CI={c.chain_def["name"] : c.chain_stats["PJ-HPDI"] for c in chains_to_plot}, 
                           colors={c.chain_def["name"] : color for c, color in zip(chains_to_plot, chain_colors)})

    #g.fig.dpi = 300
    g.export(f"plots/3x2pt/cosmology/{'_'.join(params_to_plot)}_blind_{blind}.pdf")

def plot_3x2pt_contours_big(chains, plot_settings, text_width):
    blind = "C"
    
    chain_selection = [{"probes" : ("EE",),           "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("w",),            "blind" : "None", "type" : "cosmology"}, 
                       {"probes" : ("EE", "nE"),      "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("EE", "w"),       "blind" : blind,  "type" : "cosmology"}, 
                       {"probes" : ("EE", "nE", "w"), "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("CMB",), },]
    
    chains_to_plot = select_chains(chains, chain_selection)
        
    chains_to_plot = chains_to_plot# + [planck_samples]
    
    chain_colors = [get_chain_color(c) for c in chains_to_plot]
    chain_labels = [get_chain_label(c) for c in chains_to_plot]
    
    print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

    params_to_plot = ["omegam", "sigma8", "s8", "ns", "h", "a_baryon", "a_ia", "b1l", "b1h"]
    print(f"Plotting parameters {' '.join(params_to_plot)}")

    g = getdist.plots.get_subplot_plotter(width_inch=text_width, scaling=False,
                                          settings=copy.deepcopy(plot_settings))
    g.settings.legend_fontsize = 8
    g.settings.lab_fontsize = 8
    g.settings.axes_fontsize = 8
    g.settings.legend_loc = "upper center"
    
    g.triangle_plot(chains_to_plot,
                    params=params_to_plot,
                    filled_compare=True,
                    contour_colors=chain_colors,
                    legend_labels=["KiDS-1000 cosmic shear", "BOSS galaxy clustering", 
                                   "Cosmic shear + GGL", "Cosmic shear + galaxy clustering",
                                   "$3\\times2$pt",
                                   "\\textit{Planck} TTTEEE+lowE"],
                    diag1d_kwargs={"normalized" : True},
                    param_limits={"h"   : (0.64, 0.78),
                                  "a_ia" : (0,2),
                                  "b1l" : (0.5, 4.5),
                                  "b1h" : (0.5, 4.5),
                                 },
                    legend_ncol=2,
                   )

    process_chains.plot_CI(g, chains_to_plot, params_to_plot, 
                           CI={c.chain_def["name"] : c.chain_stats["PJ-HPDI"] for c in chains_to_plot}, 
                           colors={c.chain_def["name"] : color for c, color in zip(chains_to_plot, chain_colors)})

    inset_ax = g.fig.add_axes([0.65, 0.55, 0.35, 0.35])
    g.plot_2d(chains_to_plot, param1="omegam", param2="s8", 
              filled=True,
              colors=chain_colors, 
              lims=[0.1,0.6, 0.62, 0.88],
              _no_finish=True,
              ax=inset_ax)

    
    #g.fig.dpi = 300
    g.export(f"plots/3x2pt/cosmology/{'_'.join(params_to_plot)}_blind_{blind}.pdf")
    

def plot_survey_contours(chains, boss_kv450_samples, des_3x2pt_samples, planck_samples, 
                         plot_settings, column_width):
    chain_selection = [{"probes" : ("EE", ), "blind" : "C",  "type" : "cosmology"},
                       {"probes" : ("EE", "nE", "w"), "blind" : "C",  "type" : "cosmology"},
                       ]

    EE_chain, fiducial_chain = select_chains(chains, chain_selection)
    chains_to_plot = [#kv450_samples,
                      boss_kv450_samples, 
                      #EE_chain,
                      des_3x2pt_samples, 
                      fiducial_chain, 
                      planck_samples]

    chain_colors = [get_chain_color(c) for c in chains_to_plot]
    chain_labels = [get_chain_label(c) for c in chains_to_plot]

    chain_colors[0] = "navy"  #BOSS+KV450
    chain_colors[1] = "orange"  #DES 3x2pt

    print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

    params_to_plot = ["omegam", "sigma8",]
    print(f"Plotting parameters {' '.join(params_to_plot)}")

    g = getdist.plots.get_subplot_plotter(subplot_size=column_width, scaling=False,
                                        settings=copy.deepcopy(plot_settings))
    g.settings.legend_fontsize = 8
    g.settings.lab_fontsize = 8
    g.settings.axes_fontsize = 8

    g.plot_2d(chains_to_plot, param1="omegam", param2="sigma8", 
            filled=True,
            colors=chain_colors, lims=[0.2,0.38, 0.6, 1.1], _no_finish=True)


    g.add_legend(["BOSS+KV450 (Tröster et al. 2020)", "DES Y1 $3\\times2$pt (DES Collaboration 2018)",
                  "KiDS-1000+BOSS+2dFLenS $3\\times2$pt", 
                  "\\textit{Planck} TTTEEE+lowE",], figure=False, #legend_loc="upper left", 
                fontsize=8, frameon=False)
    #g.finish_plot()

    g.export(f"plots/3x2pt/misc/{'_'.join(params_to_plot)}_survey_comparison.pdf")


def plot_S8_shifts(chains, planck_samples, boss_kv450_samples, des_3x2pt_samples, 
                   kv450_samples, des_cs_samples, hsc_pCl_samples, hsc_xipm_samples, 
                   plot_settings, column_width):

    line_height = 0.25
    line_offset = line_height/5

    label_x_position = 0.41

    parameter = "s8"

    CI_to_plot = [("M-HPDI", "marg MAP", ":", "D"),]# ("nominal CI", "nominal PE", ":", "s")]# ("std", "mean"), ("quantile CI", "median")]

    blind = "C"
    chain_selection = [{"probes" : ("EE", "nE", "w"), "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("EE",),           "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("w",),            "blind" : "None", "type" : "cosmology"}, 
                       {"probes" : ("EE", "nE"),      "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("EE", "w"),       "blind" : blind,  "type" : "cosmology"}, 
                    ]

    spacing_idx = []
    chains_to_plot = select_chains(chains, chain_selection)

    # Use Asgari 2020 chains for EE
    #chains_to_plot[-2] = asgari_bp_samples

    spacing_idx.append(len(chains_to_plot))
    chains_to_plot += [planck_samples,]#chains["None"]["multinest_Planck"],]

    spacing_idx.append(len(chains_to_plot))
    chains_to_plot += [boss_kv450_samples,
                    des_3x2pt_samples, 
                    ]


    spacing_idx.append(len(chains_to_plot))
    chains_to_plot += [kv450_samples,
                    des_cs_samples, 
                    hsc_pCl_samples, 
                    hsc_xipm_samples, 
                    ]

    chain_sets = {"cosmology" : (chains_to_plot, spacing_idx)}

    chain_selection = [{"probes" : ("EE", "nE", "w"), "blind" : blind,  "type" : "cosmology"},
                    ]
    spacing_idx = []
    chains_to_plot = select_chains(chains, chain_selection)
    spacing_idx.append(len(chains_to_plot))

    chain_selection = [{"label" : "No baryon",            "blind" : blind},
                       {"label" : "No higher order GC",   "blind" : blind},
                       {"label" : "No z-bin 1+2",         "blind" : blind},
                       {"label" : "No z-bin 1",           "blind" : blind},
                       {"label" : "No z-bin 2",           "blind" : blind},
                       {"label" : "No z-bin 3",           "blind" : blind},
                       {"label" : "No z-bin 4",           "blind" : blind},
                       {"label" : "No z-bin 5",           "blind" : blind},]
        
    chains_to_plot += select_chains(chains, chain_selection)

    chain_sets["systematics"] = (chains_to_plot, spacing_idx)


    for name, (chains_to_plot, spacing_idx) in chain_sets.items():
        
        y_size = line_height*(len(chains_to_plot)+len(spacing_idx)+1)

        chain_colors = [get_chain_color(c) for c in chains_to_plot]
        chain_labels = [get_chain_label(c) for c in chains_to_plot]
        
        fig, ax = plt.subplots(1, 1, figsize=(column_width, y_size), linewidth=1)
        fig.subplots_adjust(bottom=0.15, top=0.99, left=0.01, right=0.99)
        
        ax.grid(False)
        ax.set_xlim(0.40, 0.87)
        ax.set_xticks([0.7, 0.75, 0.8, 0.85])
        
        ax.set_ylim(-y_size, line_height)
        ax.set_yticks([])
        
        ax.set_xlabel(r"$S_8\equiv\sigma_8\sqrt{\Omega_{\rm m}/0.3}$")
        
        ax.plot([], [], ls="-", marker="o", ms=2, lw=1, c="k", alpha=1.0, label="MAP + PJ-HPD CI")
        ax.plot([], [], ls=":", marker="D", ms=2, lw=1, c="k", alpha=1.0, label="M-HPD CI")
        
        if name == "cosmology":
            ax.plot([], [], ls="--", marker="s", ms=2, lw=1, c="k", alpha=1.0, label="nominal")
        ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
        
        for i, c in enumerate(chains_to_plot):
            y = -(i+1)*line_height 
            for s in spacing_idx:
                if i >= s:
                    y -= line_height
            
            color = chain_colors[i]
            label = chain_labels[i]
            
            if c.chain_def["type"] == "external":
                CI = c.chain_stats["nominal CI"][parameter]
                pe = c.chain_stats["nominal PE"][parameter]

                ax.hlines(y=y, 
                        xmin=CI[0], xmax=CI[1],
                        color=color, ls="--", alpha=1.0, lw=1)
                ax.plot(pe, y, marker="s", ms=2, color=color, alpha=1)
            else:
                # PJ-HPDI
                PJ_HPDI = c.chain_stats["PJ-HPDI"][parameter]
                chain_MAP = c.chain_stats["chain MAP"][parameter]
                MAP = c.chain_stats.get("MAP", None)

                if isinstance(PJ_HPDI, tuple):
                    ax.hlines(y=y+line_offset/2, xmin=PJ_HPDI[0], xmax=PJ_HPDI[1],
                            color=color, lw=1)
                if MAP:
                    ax.plot(MAP[parameter], y+line_offset/2, marker="o", ms=2, c=color)
                # Other CI definitions
                for j, (CI_name, point_estimate_name, ls, marker) in enumerate(CI_to_plot):
                    if CI_name not in c.chain_stats:
                        continue

                    CI = c.chain_stats[CI_name][parameter]
                    pe = c.chain_stats[point_estimate_name][parameter]

                    ax.hlines(y=y-(j+0.5)*line_offset, 
                            xmin=CI[0], xmax=CI[1],
                            color=color, ls=ls, alpha=0.8, lw=1)
                    ax.plot(pe, y-(j+0.5)*line_offset, marker=marker, ms=2, color=color, alpha=0.8)

            ax.annotate(label, xy=(label_x_position, y-line_height/8), fontsize=8)
            
            if i == 0:
                ax.axvspan(*PJ_HPDI, color="grey", alpha=0.3, lw=0)
        
        fig.dpi = 300
        fig.savefig(f"plots/3x2pt/{name}/S8_comparison_blind{blind}.pdf")

def plot_clustering_ns(chains, BOSS_fixed, BOSS_fixed_wide_ns, plot_settings, column_width):
    g = getdist.plots.get_subplot_plotter(width_inch=column_width, scaling=False,
                                      settings=copy.deepcopy(plot_settings))
    g.settings.legend_fontsize = 8
    g.settings.lab_fontsize = 8
    g.settings.axes_fontsize = 8

    #params_to_plot = ["omegam", "sigma8", "s8", "h", "ns"]
    params_to_plot = ["omegam", "sigma8", "s8", "h", "ns", 
                    #"omegach2", "omegabh2",
                    #"b1l", 
                    #"b2l", "gamma3l", "a_virl", 
                    #"b1h", 
                    #"b2h", "gamma3h", "a_virh"
                    ]

    g.triangle_plot([#BOSS_chain, BOSS_cosmomc_samples, 
                    BOSS_fixed_wide_ns, BOSS_fixed],
                    params=params_to_plot,
                    filled_compare=True,
                    contour_colors=["C1", "C0"],
                    legend_labels=["BOSS galaxy clustering, wide $n_s$ prior", "BOSS galaxy clustering, fiducial"],
                    diag1d_kwargs={"normalized" : True},
                )

    # g.fig.dpi = 300
    g.export("plots/3x2pt/systematics/GC_ns_prior.pdf")


def calculate_1d_tension(chains, planck_samples):
    with open("stats/tension_1d.txt", "w") as file:
        fid = select_chains(chains, [{"probes" : ("EE", "nE", "w"), "blind" : "C",  "type" : "cosmology"}])[0]

        print("Own Planck chain", file=file)
        planck = select_chains(chains, [{"probes" : ("CMB",)}])[0]
        for p in ["s8", "sigma8", "sigma12", "s12"]:
            fid_mean = fid.chain_stats["mean"][p]
            fid_std = (fid.chain_stats["std"][p][1]-fid.chain_stats["std"][p][0])/2
            
            fid_marg_MAP = fid.chain_stats["marg MAP"][p]
            fid_marg_u_l = (fid.chain_stats["M-HPDI"][p][1]-fid.chain_stats["marg MAP"][p], 
                            fid.chain_stats["M-HPDI"][p][0]-fid.chain_stats["marg MAP"][p])
            
            planck_mean = planck.chain_stats["mean"][p]
            planck_std = (planck.chain_stats["std"][p][1]-planck.chain_stats["std"][p][0])/2
            
            planck_marg_MAP = planck.chain_stats["marg MAP"][p]
            planck_marg_u_l = (planck.chain_stats["M-HPDI"][p][1]-planck.chain_stats["marg MAP"][p], 
                            planck.chain_stats["M-HPDI"][p][0]-planck.chain_stats["marg MAP"][p])
            
            d = (fid_mean-planck_mean)/np.sqrt(fid_std**2 + planck_std**2)
            PTE = scipy.stats.norm.cdf(d)
            print(f"{p:<8} {fid_mean:.3f}±{fid_std:.3f} ({fid_marg_MAP:.3f}^+{fid_marg_u_l[0]:.3f}_{fid_marg_u_l[1]:.3f})"
                f"  vs {planck_mean:.3f}±{planck_std:.3f} ({planck_marg_MAP:.3f}^+{planck_marg_u_l[0]:.3f}_{planck_marg_u_l[1]:.3f}): {PTE:.4f} {d:.2f}\\sigma", file=file)

        print("", file=file)
        print("Nominal Planck chain", file=file)
        planck = planck_samples
        for p in ["s8", "sigma8"]:
            fid_mean = fid.chain_stats["mean"][p]
            fid_std = (fid.chain_stats["std"][p][1]-fid.chain_stats["std"][p][0])/2
            
            planck_mean = planck.chain_stats["mean"][p]
            planck_std = (planck.chain_stats["std"][p][1]-planck.chain_stats["std"][p][0])/2
            
            d = (fid_mean-planck_mean)/np.sqrt(fid_std**2 + planck_std**2)
            print(f"{p:<8} {fid_mean:.3f}±{fid_std:.3f}  vs {planck_mean:.3f}±{planck_std:.3f}: {d:.2f}\\sigma", file=file)


        # Hellinger distance
        hellinger_planck, hellinger_planck_sigma = hellinger_distance_1D.hellinger_tension(sample1=chains["C"]["multinest_blindC_EE_nE_w"].getParams().s8,
                                                                sample2=planck_samples.getParams().s8,
                                                                weight1=chains["C"]["multinest_blindC_EE_nE_w"].weights,
                                                                weight2=planck_samples.weights)

        hellinger, hellinger_sigma = hellinger_distance_1D.hellinger_tension(sample1=chains["C"]["multinest_blindC_EE_nE_w"].getParams().s8,
                                                                sample2=chains["None"]["multinest_Planck"].getParams().s8,
                                                                weight1=chains["C"]["multinest_blindC_EE_nE_w"].weights,
                                                                weight2=chains["None"]["multinest_Planck"].weights)

        print("", file=file)
        print("Hellinger distance (own Planck chain)", file=file)
        print(f"d_H = {hellinger:.3f}, {hellinger_sigma:.2f}σ", file=file)
        print("Hellinger distance (official Planck chain)", file=file)
        print(f"d_H = {hellinger_planck:.3f}, {hellinger_planck_sigma:.2f}σ", file=file)

        # p_S statistic
        ew_3x2pt = load_equal_weight_chain(chains["C"]["multinest_blindC_EE_nE_w"])
        ew_Planck = load_equal_weight_chain(chains["None"]["multinest_Planck"])

        ew_3x2pt.sampler = "uncorrelated"
        ew_Planck.sampler = "uncorrelated"

        print("Samples in equal weight 3x2pt chain: ", ew_3x2pt.samples.shape[0])
        print("Samples in equal weight Planck chain: ", ew_Planck.samples.shape[0])

        ew_diff_chain = tensiometer.mcmc_tension.parameter_diff_chain(ew_3x2pt, ew_Planck, boost=20)
        print("Samples in diff chain: ", ew_diff_chain.samples.shape[0])

        prob = tensiometer.mcmc_tension.exact_parameter_shift(ew_diff_chain, 
                                                      param_names=["delta_s8"],
                                                      method="nearest_elimination"
                                                      )
        prob_sigma = [np.sqrt(2)*scipy.special.erfcinv(1-_s) for _s in prob]

        print("", file=file)
        print("p_S statistic (S8)", file=file)
        print(f"PTE: {1-prob[0]:.4f} ({1-prob[2]:.4f}-{1-prob[1]:.4f}), {prob_sigma[0]:.2f}σ ({prob_sigma[1]:.2f}σ-{prob_sigma[2]:.2f}σ)", file=file)

def calculate_nd_tension(chains, chain_path):
    with open("stats/chain_stats_tension_nd.txt", "w") as file:
        chain_selection = [{"probes" : ("EE", ), "blind" : "C",  "type" : "cosmology"},
                        {"probes" : ("EE", "nE"), "blind" : "C",  "type" : "cosmology"},
                        {"probes" : ("EE", "w"), "blind" : "C",  "type" : "cosmology"},
                        {"probes" : ("w",), "blind" : "None",  "type" : "cosmology"},
                        {"probes" : ("EE", "nE", "w"), "blind" : "C",  "type" : "cosmology"},
                        {"probes" : ("CMB", "EE", "nE", "w"),},
                        {"probes" : ("CMB",)},
                        ]

        chains_to_analyse = select_chains(chains, chain_selection)

        # chains_to_analyse += [asgari_cosebis_samples, asgari_cosebis_Planck_samples, asgari_Planck_samples]

        stats_direct = {}
        for name, n_varied, c in zip(["CS", "CS+GGL", "CS+GC", "GC", "3x2pt", "3x2pt+Planck", "Planck",
                                    #"COSEBIS", "COSEBIS+Planck", "Planck (MA)"
                                    ], 
                                    (12, 18, 20, 13, 20, 22, 7,
                                    #12, 14, 7
                                    ), 
                                    chains_to_analyse):
            logL = c.loglikes
            
            d = 2 * (np.average(logL**2, weights=c.weights) - np.average(logL, weights=c.weights)**2)
            mean_logL = np.average(logL, weights=c.weights)
            
            # Raveri & Hu 2018
            u, l = list(c.ranges.upper.values())[:n_varied], list(c.ranges.lower.values())[:n_varied]
            
            # Flat priors
            prior_cov = np.diag([1/12*(u_-l_)**2 for u_,l_ in zip(u,l)])
            
            # Add covariances for Gaussian priors
            param_names = [n.name for n in c.getParamNames().names]
            #print(param_names)
            if "p_z1" in param_names:
                dz_idx = slice(param_names.index("p_z1"), param_names.index("p_z5")+1)
                dz_cov = np.loadtxt(os.path.join(c.chain_def["root_dir"], "data/KiDS/SOM_cov_multiplied.asc"))
                prior_cov[dz_idx,dz_idx] = dz_cov
            if "calPlanck" in param_names:
                prior_cov[:5,:5] *= 10 # Account for the artifically shrunk prior volume on the cosmology parameters
                calPlanck_idx = param_names.index("calPlanck")
                prior_cov[calPlanck_idx, calPlanck_idx] = 0.0025**2
            
            # MA chains
            # if "nofz_shifts--uncorr_bias_1" in param_names:
            #     dz_idx = slice(param_names.index("nofz_shifts--uncorr_bias_1"), param_names.index("nofz_shifts--uncorr_bias_5")+1)
            #     dz_cov = np.loadtxt("../runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE/data/KiDS/SOM_cov_multiplied.asc")
            #     prior_cov[dz_idx,dz_idx] = dz_cov
                
            posterior_cov = c.getCov(n_varied)
            # From tensionmetrics
            e, _ = np.linalg.eig(np.linalg.inv(prior_cov) @ posterior_cov)
            e[e > 1.] = 1.
            e[e < 0.] = 0.
            n_tot = len(e)
            n_eff = n_tot - np.sum(e)

            MAP_loglike = c.chain_stats["MAP_loglike"] if "MAP" in c.chain_stats else c.chain_stats["chain MAP_loglike"]
            
            n = 2*MAP_loglike - 2*mean_logL
            
            stats_direct[name] = {"d" : d, "n_eff" : n_eff, "n_post" : n,
                                "meanL" : mean_logL, "logZ" : c.log_Z, "D" : mean_logL - c.log_Z,
                                "MAP_loglike" : MAP_loglike}
            
            print(f"{name:<20}: d = {d:.2f}, n_eff = {n_eff:.2f}, n_post = {n:.2f}, <logL> = {mean_logL:.2f}, D = {mean_logL - c.log_Z:.2f}, logZ = {c.log_Z:.2f}", file=file)

        
        stats = {}
        for name, root in [("Planck", os.path.join(chain_path, "cosmology/multinest_Planck/chain/multinest/multinest_multinest_Planck_")),
                           ("3x2pt+Planck", os.path.join(chain_path, "cosmology/multinest_blindC_EE_nE_w_Planck/chain/multinest/multinest_multinest_blindC_EE_nE_w_Planck_")),
                           ("3x2pt", os.path.join(chain_path, "cosmology/multinest_blindC_EE_nE_w/chain/multinest/multinest_multinest_blindC_EE_nE_w_")),
                           #("Planck (MA)", "../runs/3x2pt/ma/planck/chain/multinest_"),
                           #("COSEBIS+Planck", "../runs/3x2pt/ma/cosebis/chain/multinest_C_"),
                           #("COSEBIS", "../runs/3x2pt/ma/main_chains_iterative_covariance/cosebis/chain/multinest_C_"),
                           ]:
            nested = anesthetic.NestedSamples(root=root)
            ns_output = nested.ns_output(nsamples=1000)
            stats[name] = ns_output
            print(name)
            print(f'log Z:  {ns_output["logZ"].mean():.2f}±{ns_output["logZ"].std():.2f}', file=file)
            print(f'D:      {ns_output["D"].mean():.2f}±{ns_output["D"].std():.2f}', file=file)
            print(f'd:      {ns_output["d"].mean():.2f}±{ns_output["d"].std():.2f}', file=file)


        # Compute tension metrics
        logR = stats_direct["3x2pt+Planck"]["logZ"] - stats_direct["3x2pt"]["logZ"] - stats_direct["Planck"]["logZ"]
        R = np.exp(logR)
        logI = stats_direct["3x2pt"]["D"] + stats_direct["Planck"]["D"] - stats_direct["3x2pt+Planck"]["D"]
        logS = logR - logI

        d = stats_direct["3x2pt"]["d"] + stats_direct["Planck"]["d"] - stats_direct["3x2pt+Planck"]["d"]
        PTE = scipy.stats.chi2(df=d).sf(d-2*logS)
        PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(PTE)

        Q_DMAP = 2*stats_direct["3x2pt"]["MAP_loglike"] + 2*stats_direct["Planck"]["MAP_loglike"] - 2*stats_direct["3x2pt+Planck"]["MAP_loglike"]
        n_eff = stats_direct["3x2pt"]["n_eff"] + stats_direct["Planck"]["n_eff"] - stats_direct["3x2pt+Planck"]["n_eff"]
        Q_DMAP_PTE = scipy.stats.chi2(df=n_eff).sf(Q_DMAP)
        Q_DMAP_PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(Q_DMAP_PTE)


        Q_UDM_Planck_cutoff, _, _, _ = tensiometer.gaussian_tension.Q_UDM_get_cutoff(
                                                    chain_1=chains["None"]["multinest_Planck"],
                                                    chain_2=chains["C"]["multinest_blindC_EE_nE_w"],
                                                    chain_12=chains["None"]["multinest_EE_nE_w_Planck"],
                                                    param_names=["omegach2", "omegabh2", "h", "ns", "s8proxy"])

        Q_UDM_Planck, Q_UDM_Planck_dof = tensiometer.gaussian_tension.Q_UDM(
                                                chain_1=chains["None"]["multinest_Planck"],
                                                chain_12=chains["None"]["multinest_EE_nE_w_Planck"],
                                                lower_cutoff=Q_UDM_Planck_cutoff,
                                                param_names=["omegach2", "omegabh2", "h", "ns", "s8proxy"])

        Q_UDM_Planck_PTE = scipy.stats.chi2(df=Q_UDM_Planck_dof).sf(Q_UDM_Planck)
        Q_UDM_Planck_PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(Q_UDM_Planck_PTE)
            
        print("", file=file)
        print("N-d tension statistics", file=file)
        print("Stats 3x2pt vs Planck (chains)", file=file)
        print(f"log R: {logR:.2f}, R: {R:.2f}", file=file)
        print(f"log I: {logI:.2f}, log S: {logS:.2f}", file=file)
        print(f"d: {d:.2f}", file=file)
        print(f"PTE: {PTE:.3f}, {PTE_sigma:.2f}σ", file=file)

        print("", file=file)
        print(f"Q_DMAP: {Q_DMAP:.2f}, n_eff: {n_eff:.2f}", file=file)
        print(f"PTE: {Q_DMAP_PTE:.3f}, {Q_DMAP_PTE_sigma:.2f}σ", file=file)

        print("", file=file)
        print(f"Q_UDM (Planck vs joint): {Q_UDM_Planck:.2f}, n_eff: {Q_UDM_Planck_dof:.2f}", file=file)
        print(f"PTE: {Q_UDM_Planck_PTE:.3f}, {Q_UDM_Planck_PTE_sigma:.2f}σ", file=file)

        logR = stats["3x2pt+Planck"]["logZ"] - stats["3x2pt"]["logZ"] - stats["Planck"]["logZ"]
        R = np.exp(logR)
        logI = stats["3x2pt"]["D"] + stats["Planck"]["D"] - stats["3x2pt+Planck"]["D"]
        logS = logR - logI

        d = stats["3x2pt"]["d"] + stats["Planck"]["d"] - stats["3x2pt+Planck"]["d"]
        PTE = scipy.stats.chi2(df=d).sf(d-2*logS)
        PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(PTE)

        print("", file=file)
        print("Stats 3x2pt vs Planck (anesthetic)", file=file)
        print(f"log R: {logR.mean():.2f}±{logR.std():.2f}, R: {R.mean():.2f}±{R.std():.2f}", file=file)
        print(f"log I: {logI.mean():.2f}±{logI.std():.2f}, log S: {logS.mean():.2f}±{logS.std():.2f}", file=file)
        print(f"d: {d.mean():.2f}±{d.std():.2f}", file=file)
        print(f"PTE: {PTE.mean():.2f}±{PTE.std():.2f}, sigmas: {PTE_sigma.mean():.2f}±{PTE_sigma.std():.2f}", file=file)

    return stats_direct, stats

def create_goodness_of_fit_table(chains, stats_direct):
    w_ndof = 168
    EE_ndof = 120
    nE_ndof = 22

    data_ndof = {("EE", "nE", "w") : EE_ndof + nE_ndof + w_ndof,
                ("EE", "w")       : EE_ndof + w_ndof,
                ("EE", "nE",)     : EE_ndof + nE_ndof,
                ("EE", )          : EE_ndof,
                ("w",)            : w_ndof,}

    param_neff = {("EE", "nE", "w") : stats_direct["3x2pt"]["n_eff"],
                ("EE", "w")       : stats_direct["CS+GC"]["n_eff"],
                ("EE", "nE",)     : stats_direct["CS+GGL"]["n_eff"],
                ("EE", )          : stats_direct["CS"]["n_eff"],
                ("w",)            : stats_direct["GC"]["n_eff"],}

    param_neff_J2020 = {("EE", "nE", "w") : None,
                        ("EE", "w")       : None,
                        ("EE", "nE",)     : 8.7,
                        ("EE", )          : 4.5,
                        ("w",)            : None,}

    goodness_of_fit_top = r"""
\begin{tabular}{lcccccc}
    \toprule
    Probe             & $\chi^2_{\rm MAP}$  & Data DoF  & Model DoF                   & PTE  & Model DoF          & PTE    \\
                    &                     &           &\citep{joachimi/etal:inprep} &      & \citep{Raveri2019} & \\
    \midrule
"""
        
    goodness_of_fit_bottom = r"""
    \bottomrule
\end{tabular}
"""
        
    blind = "C"
    chain_selection = [{"probes" : ("EE",),           "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("w",),            "blind" : "None", "type" : "cosmology"}, 
                       {"probes" : ("EE", "nE"),      "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("EE", "w"),       "blind" : blind,  "type" : "cosmology"}, 
                       {"probes" : ("EE", "nE", "w"), "blind" : blind,  "type" : "cosmology"}]
    
    chains_to_analyse = select_chains(chains, chain_selection)
    chain_labels = [get_chain_label(c) for c in chains_to_analyse]
                                    
    # Use Asgari 2020 values for EE
    #chains_to_analyse[0] = asgari_bp_samples
    
    goodness_of_fit_content = ""
    for label, c in zip(chain_labels, chains_to_analyse):
        try:
            chi2 = -2*c.chain_stats["MAP_loglike"]
            chi2_prefix = ""
        except KeyError:
            chi2 = -2*c.chain_stats["chain MAP_loglike"]
            chi2_prefix = "< "
        
        ndof = data_ndof[c.chain_def["probes"]]
        neff = param_neff[c.chain_def["probes"]]
        
        neff_J2020 = param_neff_J2020[c.chain_def["probes"]]
        neff_J2020_str = f"{neff_J2020:.1f}" if neff_J2020 is not None else "--"
        
        p = scipy.stats.chi2(df=ndof-neff).sf(chi2)
        
        p_J2020_str = f"{scipy.stats.chi2(df=ndof-neff_J2020).sf(chi2):<5.3f}" if neff_J2020 is not None else "--"
        
        goodness_of_fit_content += f"\t{label:<16} & ${chi2_prefix}{chi2:>5.1f}$ & ${ndof:>d}$  &{neff_J2020_str} & {p_J2020_str} &{neff:.1f} & {p:<5.3f} \\\\" + "\n"
        
    with open("stats/goodness_of_fit.tex", "w") as file:
        print(goodness_of_fit_top + goodness_of_fit_content + goodness_of_fit_bottom, file=file)

def create_parameter_constraint_table(chains):
    short_labels = {("EE", "nE", "w") : r"$3\times2$pt",
                    ("w",) : r"GC",
                    ("EE", "w") : r"CS+GC",
                    ("EE", "nE") : r"CS+GGL",}

    parameter_constraints_bottom = r"""
    \bottomrule
\end{tabular}
"""

    cosmology_params_to_analyse = ['s8', 'omegach2', 'omegabh2', 'h', 'ns']
    nuisance_params_to_analyse = ['a_baryon', 'a_ia', 
                                'shift_z1', 'shift_z2', 'shift_z3', 'shift_z4', 'shift_z5', 
                                'b1l', 'b2l', 'gamma3l', 'a_virl', 'b1h', 'b2h', 'gamma3h', 'a_virh']
    derived_params_to_analyse = ['omegam', 'sigma8', 'sigma12', 's12', 'as', 'theta']

    blind = "C"
    
    chain_selection = [{"probes" : ("EE", "nE", "w"), "blind" : blind,  "type" : "cosmology"},
                       #{"probes" : ("EE",),           "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("w",),            "blind" : "None", "type" : "cosmology"}, 
                       {"probes" : ("EE", "nE"),      "blind" : blind,  "type" : "cosmology"},
                       {"probes" : ("EE", "w"),       "blind" : blind,  "type" : "cosmology"}, 
                       ]
    
    chains_to_analyse = select_chains(chains, chain_selection)
    chain_labels = [short_labels[c.chain_def["probes"]] for c in chains_to_analyse]
    
    l1 = "Parameter    "
    l2 = "             "
    for label in chain_labels:
        l1 += f"& {label} & {label}"
        l2 += "& (joint) & (marginal)"
        
    l1 += " \\\\ \n"
    l2 += " \\\\ \n"
    
    parameter_constraints_top = r"""
\begin{tabular}""" + "{" + "l"*(2*len(chains_to_analyse)+1) + "}" + r"""
    \toprule
    """ + l1 + l2 + r"""
    \midrule
"""
            
    parameter_constraints_lines = []
    for param_group in [cosmology_params_to_analyse, nuisance_params_to_analyse, derived_params_to_analyse]:
        param_labels = {p : [n.label for n in chains_to_analyse[0].getParamNames().names if n.name == p][0] 
                                for p in param_group}

        for p in param_group:
            parameter_constraints_str = f"${param_labels[p]:<8}$"

            for label, c in zip(chain_labels, chains_to_analyse):
                e = 0
                if "MAP" in c.chain_stats and p in c.chain_stats["MAP"]:
                    MAP = copy.copy(c.chain_stats["MAP"][p])
                    if np.log10(abs(MAP)) < -2 or np.log10(abs(MAP)) > 2:
                        e = int(np.log10(abs(MAP)))-1
                        MAP /= 10**e
                    MAP_str = f"{MAP:.3g}"
                elif p in c.chain_stats["chain MAP"]:
                    MAP = c.chain_stats["chain MAP"][p]
                    if np.log10(abs(MAP)) < -2 or np.log10(abs(MAP)) > 2:
                        e = int(np.log10(abs(MAP)))-1
                        MAP /= 10**e
                    MAP_str = f"\\approx {MAP:.3g}"
                else:
                    MAP_str = ""

                if p in c.chain_stats["PJ-HPDI n_sample"] and c.chain_stats["PJ-HPDI n_sample"][p] > 10:
                    l, u = copy.copy(c.chain_stats["PJ-HPDI"][p])
                    l /= 10**e
                    u /= 10**e
                    u_str = "{" + f"+{u-MAP:.2g}" + "}"
                    l_str = "{" + f"{l-MAP:.2g}" + "}"
                    PJ_HPDI_str = f"{MAP_str}^{u_str}_{l_str}"
                else:
                    PJ_HPDI_str = f"{MAP_str}"
                if e != 0:
                    PJ_HPDI_str += r"\times 10^{" + str(e) + "}"
                PJ_HPDI_str = "$" + PJ_HPDI_str + "$"

                if p in c.chain_stats["M-HPDI constrained"]:
                    if c.chain_stats["M-HPDI constrained"][p]:
                        MAP = copy.copy(c.chain_stats["marg MAP"][p])
                        if np.log10(abs(MAP)) < -2 or np.log10(abs(MAP)) > 2:
                            e = int(np.log10(abs(MAP)))-1
                            MAP /= 10**e
                        else:
                            e = 0
                        MAP_str = f"{MAP:.3g}"
                        l, u = copy.copy(c.chain_stats["M-HPDI"][p])
                        l /= 10**e
                        u /= 10**e
                        u_str = "{" + f"+{u-MAP:.2g}" + "}"
                        l_str = "{" + f"{l-MAP:.2g}" + "}"
                        M_HPDI_str = f"{MAP_str}^{u_str}_{l_str}"
                        if e != 0:
                            M_HPDI_str += r"\times 10^{" + str(e) + "}"
                        M_HPDI_str = "$" + M_HPDI_str + "$"
                    else:
                        M_HPDI_str = "--"
                else:
                    M_HPDI_str = ""


                parameter_constraints_str += f"& {PJ_HPDI_str} & {M_HPDI_str}"

            parameter_constraints_str += "\\\\ [0.3 em]"
            parameter_constraints_lines.append(parameter_constraints_str)
        parameter_constraints_lines.append(r"\midrule")
        
    parameter_constraints_lines.pop(-1)
    parameter_constraints_content = "\n".join(parameter_constraints_lines)
    
    with open("stats/parameter_constraints.tex", "w") as file:
        print(parameter_constraints_top + parameter_constraints_content + parameter_constraints_bottom, file=file)
    
    print("Band power MAP:")
    print(chains["C"]["multinest_blindC_EE"].chain_stats["MAP"])
    print("MAP chi2: ", -2*chains["C"]["multinest_blindC_EE"].chain_stats["MAP_loglike"])
    print("PJ-HPD CI:")
    print(chains["C"]["multinest_blindC_EE"].chain_stats["PJ-HPDI"])
    
        
if __name__ == "__main__":

    KiDS1000_chain_path = "../../../../kcap/KiDS-1000_chains/"
    external_chain_path = "../../../../external_datasets/"

    text_width = 523.5307/72
    column_width = 256.0748/72

    plot_settings = getdist.plots.GetDistPlotSettings()
    plot_settings.figure_legend_frame = False
    plot_settings.legend_frame = False
    plot_settings.figure_legend_loc = "upper right"
    plot_settings.alpha_filled_add=0.8
    plot_settings.alpha_factor_contour_lines=0.8
    plot_settings.fontsize = 10
    # plot_settings.axes_fontsize = 8
    #plot_settings.lab_fontsize = 8
    plot_settings.legend_fontsize = 10

    plot_settings.x_label_rotation = 45.0

    matplotlib.rc("text", usetex=True)
    matplotlib.rc("text.latex", preamble=
r"""
\usepackage{txfonts}
\newcommand{\mathdefault}[1][]{}""")

    matplotlib.rc("font", family="Times")


    chains = load_chains(KiDS1000_chain_path)

    plot_systematics_contours(chains, plot_settings, text_width)
    plot_3x2pt_contours_small(chains, plot_settings, column_width)
    plot_3x2pt_contours_big(chains, plot_settings, text_width)

    CI_coverage = 0.68

    c = 2.99792458e5
    planck_samples = getdist.loadMCSamples(os.path.join(external_chain_path, "Planck/COM_CosmoParams_base-plikHM_R3.01/base/plikHM_TTTEEE_lowl_lowE/base_plikHM_TTTEEE_lowl_lowE"),)
    planck_samples.addDerived(planck_samples.getParams().H0/100, name="h", label="h")
    planck_samples.addDerived(planck_samples.getParams().sigma8*np.sqrt(planck_samples.getParams().omegam/0.3), name="s8", label="S_8")
    planck_samples.addDerived(planck_samples.getParams().rdrag/(planck_samples.getParams().DM038**2 * c*0.38 / planck_samples.getParams().Hubble038)**(1/3), name="rsDv1", label="r_{\\rm{d}}/D_V(z1)")
    planck_samples.addDerived(planck_samples.getParams().DM038 * planck_samples.getParams().Hubble038/c, name="FAP1", label="F_{\\rm AP}(0.38)")
    planck_samples.addDerived(planck_samples.getParams().fsigma8z038, name="fsigma8z1", label="f\\sigma_8(0.38)")
    planck_samples.addDerived(planck_samples.getParams().rdrag/(planck_samples.getParams().DM061**2 * c*0.61 / planck_samples.getParams().Hubble061)**(1/3), name="rsDv2", label="r_{\\rm{d}}/D_V(z2)")
    planck_samples.addDerived(planck_samples.getParams().DM061 * planck_samples.getParams().Hubble061/c, name="FAP2", label="F_{\\rm AP}(0.61)")
    planck_samples.addDerived(planck_samples.getParams().fsigma8z061, name="fsigma8z2", label="f\\sigma_8(0.61)")
    planck_samples.addDerived(1/planck_samples.getParams().rsDv1, name="Dvrs1", label="D_V(0.38)/r_{\\rm{d}}")
    planck_samples.addDerived(1/planck_samples.getParams().rsDv2, name="Dvrs2", label="D_V(0.61)/r_{\\rm{d}}")
    planck_samples.addDerived(planck_samples.getParams().A*1e-9, name="as", label="A_s")

    planck_samples.addDerived(-0.5*(planck_samples.getParams().chi2_CMB + planck_samples.getParams().chi2_prior), name="logpost", label="\\log P")
    planck_samples.addDerived(-0.5*planck_samples.getParams().chi2_prior, name="logprior", label="\\log \\pi")
    planck_samples.name_tag = "plikHM_TTTEEE_lowl_lowE"

    planck_samples.chain_stats = get_stats(planck_samples, CI_coverage=CI_coverage,
                                           weights=planck_samples.weights/planck_samples.weights.sum(),
                                           params=["omegach2", "omegabh2", "omegam", "sigma8", "s8", "as", "ns", "h"])

    planck_samples.chain_stats["nominal CI"] = {"s8" : (0.818, 0.850)}
    planck_samples.chain_stats["nominal PE"] = {"s8" : 0.834}

    planck_samples.chain_def = {"name" : "plikHM_TTTEEE_lowl_lowE",
                                "color" : "darkslategrey",
                                "type" : "external",
                                "prones" : ("CMB",),
                                "label" : "\\textit{Planck} TTTEEE+lowE"}

    hsc_xipm_samples = getdist.loadMCSamples(os.path.join(external_chain_path, "HSC/xipm/HSC_hamana2020_fiducial/hsc_hamana2020_fiducial"),
                                            )
    hsc_xipm_samples.addDerived(hsc_xipm_samples.getParams().par_oc*hsc_xipm_samples.getParams().par_h0**2, name="omegach2")
    hsc_xipm_samples.addDerived(hsc_xipm_samples.getParams().par_ob*hsc_xipm_samples.getParams().par_h0**2, name="omegabh2")
    hsc_xipm_samples.addDerived(hsc_xipm_samples.getParams().par_om, name="omegam")
    hsc_xipm_samples.addDerived(hsc_xipm_samples.getParams().par_sig8, name="sigma8")
    hsc_xipm_samples.addDerived(hsc_xipm_samples.getParams().par_s8, name="s8")
    hsc_xipm_samples.addDerived(-hsc_xipm_samples.loglikes/2, name="logpost")

    hsc_xipm_samples.name_tag = "HSC $\\xi_\\pm$ (Hamana et al. 2020)"

    hsc_xipm_samples.chain_stats = get_stats(hsc_xipm_samples, CI_coverage=CI_coverage,
                                             params=["omegam", "sigma8", "s8"])
    hsc_xipm_samples.chain_def = {"label" : hsc_xipm_samples.name_tag, 
                                "color" : "C2", 
                                "type" : "external", 
                                "probes" : ("EE",)}
    hsc_xipm_samples.chain_stats["nominal CI"] = {"s8" : (0.775, 0.836)}
    hsc_xipm_samples.chain_stats["nominal PE"] = {"s8" : 0.804}

    chain = np.loadtxt(os.path.join(external_chain_path, "HSC/pCl/HSC_Y1_LCDM_post_mnu0.06eV.txt"), usecols=[0,1,16,17])
    hsc_pCl_samples = getdist.MCSamples(name_tag="HSC pseudo-$C_\\ell$ (Hikage et al. 2019)",
                                        samples=chain[:,[2,3]],
                                        weights=chain[:,0],
                                        loglikes=chain[:,1],
                                        names=["omegam", "sigma8"],
                                        sampler="nested")

    hsc_pCl_samples.addDerived(hsc_pCl_samples.getParams().sigma8*np.sqrt(hsc_pCl_samples.getParams().omegam/0.3), name="s8")
    hsc_pCl_samples.addDerived(hsc_pCl_samples.loglikes, name="logpost")

    hsc_pCl_samples.chain_stats = get_stats(hsc_pCl_samples, 
                                            CI_coverage=CI_coverage,
                                            weights=hsc_pCl_samples.weights/hsc_pCl_samples.weights.sum(),
                                            params=["omegam", "sigma8", "s8"])
    hsc_pCl_samples.chain_def = {"label" : hsc_pCl_samples.name_tag, 
                                "color" : "C2", 
                                "type" : "external", 
                                "probes" : ("EE",)}
    hsc_pCl_samples.chain_stats["nominal CI"] = {"s8" : (0.747, 0.81)}
    hsc_pCl_samples.chain_stats["nominal PE"] = {"s8" : 0.78}

    des_3x2pt_samples = process_chains.load_chain(os.path.join(external_chain_path, "DES/d_l3.txt"), 
                                                run_name="DES Y1 $3\\times2$pt (DES Collaboration 2018)")
    des_3x2pt_samples.addDerived(des_3x2pt_samples.getParams().sigma8*np.sqrt(des_3x2pt_samples.getParams().omegam/0.3), name="s8")

    des_3x2pt_samples.chain_stats = get_stats(des_3x2pt_samples, 
                                              CI_coverage=CI_coverage,
                                              params=["omegam", "sigma8", "s8"])
    des_3x2pt_samples.chain_def = {"label" : des_3x2pt_samples.name_tag, 
                                   "color" : "teal", 
                                   "type" : "external", 
                                   "probes" : ("3x2pt",)}
    des_3x2pt_samples.chain_stats["nominal CI"] = {"s8" : (0.758, 0.804)}
    des_3x2pt_samples.chain_stats["nominal PE"] = {"s8" : 0.783}


    des_cs_samples = process_chains.load_chain(os.path.join(external_chain_path, "DES/s_l3.txt"), 
                                            run_name="DES Y1 cosmic shear (Troxel et al. 2018)")
    des_cs_samples.addDerived(des_cs_samples.getParams().sigma8*np.sqrt(des_cs_samples.getParams().omegam/0.3), name="s8")

    des_cs_samples.chain_stats = get_stats(des_cs_samples, 
                                           CI_coverage=CI_coverage,
                                           params=["omegam", "sigma8", "s8"])
    des_cs_samples.chain_def = {"label" : des_cs_samples.name_tag, 
                                "color" : "C2", 
                                "type" : "external", 
                                "probes" : ("xipm",)}
    des_cs_samples.chain_stats["nominal CI"] = {"s8" : (0.755, 0.809)}
    des_cs_samples.chain_stats["nominal PE"] = {"s8" : 0.782}

    kv450_samples = process_chains.load_chain(os.path.join(external_chain_path, "KV450/KV450_fiducial/KV450_fiducial.txt"), 
                                            run_name="KV450 (Hildebrandt et al. 2020)", chain_format="montepython")
    kv450_samples.addDerived(-kv450_samples.loglikes, name="logpost")

    kv450_samples.chain_stats = get_stats(kv450_samples, 
                                          CI_coverage=CI_coverage,
                                          params=["omegam", "sigma8", "s8"])
    kv450_samples.chain_def = {"label" : kv450_samples.name_tag, 
                                "color" : "C2", 
                                "type" : "external", 
                                "probes" : ("3x2pt",)}
    kv450_samples.chain_stats["nominal CI"] = {"s8" : (0.701, 0.777)}
    kv450_samples.chain_stats["nominal PE"] = {"s8" : 0.737}

    boss_kv450_samples = process_chains.load_chain(os.path.join(external_chain_path, "BOSS+KV450/samples_fiducial_sample_theta_multinest.txt"), 
                                                run_name="BOSS+KV450 (Tröster et al. 2020)")
    boss_kv450_samples.chain_stats = get_stats(boss_kv450_samples, 
                                               CI_coverage=CI_coverage,
                                               params=["omegam", "sigma8", "s8", "ns"])
    boss_kv450_samples.chain_def = {"label" : boss_kv450_samples.name_tag, 
                                    "color" : "mediumseagreen", 
                                    "type" : "external", 
                                    "probes" : ("3x2pt",)}
    boss_kv450_samples.chain_stats["nominal CI"] = {"s8" : (0.702, 0.754)}
    boss_kv450_samples.chain_stats["nominal PE"] = {"s8" : 0.728}


    BOSS_fixed = process_chains.load_chain(os.path.join(KiDS1000_chain_path, "systematics/multinest_sample_fiducial_w/chain/samples_multinest_sample_fiducial_w.txt"),
                                              values=os.path.join(KiDS1000_chain_path, "systematics/multinest_sample_fiducial_w/config/values.ini"),
                                              burn_in=0,
                                              run_name="new, fiducial")
    BOSS_fixed_wide_ns = process_chains.load_chain(os.path.join(KiDS1000_chain_path, "systematics/multinest_sample_S8_wide_ns_w/chain/samples_multinest_sample_S8_wide_ns_w.txt"),
                                                values=os.path.join(KiDS1000_chain_path, "systematics/multinest_sample_S8_wide_ns_w/config/values.ini"),
                                                burn_in=0,
                                                run_name="new, wide ns")

    
    plot_clustering_ns(chains, BOSS_fixed, BOSS_fixed_wide_ns, plot_settings, column_width)

    plot_survey_contours(chains, boss_kv450_samples, des_3x2pt_samples, planck_samples, 
                         plot_settings, column_width)

    plot_S8_shifts(chains, planck_samples, boss_kv450_samples, des_3x2pt_samples, 
                   kv450_samples, des_cs_samples, hsc_pCl_samples, hsc_xipm_samples, 
                   plot_settings, column_width)
    
    # calculate_1d_tension(chains, planck_samples)
    # stats_chain, stats_nested = calculate_nd_tension(chains, KiDS1000_chain_path)

    # create_goodness_of_fit_table(chains, stats_chain)
    # create_parameter_constraint_table(chains)