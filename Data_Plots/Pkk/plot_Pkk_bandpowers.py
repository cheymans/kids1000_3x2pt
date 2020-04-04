import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
from astropy.io import ascii
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import chi2
import matplotlib.ticker as ticker


# Some font setting
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 19}

plt.rc('font', **font)

#set up the figure grid
tick_spacing = 2
fig,axes= plt.subplots(6,7,figsize=(10, 7.5),gridspec_kw={'hspace': 0, 'wspace': 0})

# Read in user input to set the patch, blind, zmin,zmax, nbootstrap
if len(sys.argv) <2: 
    print("Usage: %s LFVER BLIND e.g 2Dbins_v2_goldclasses A" % sys.argv[0])
    sys.exit(1)
else:
    LFVER=sys.argv[1] # catalogue version identifier
    BLIND=sys.argv[2] # blind
        
# number of tomographic bins, and band power modes to plot
ntomobin=5
ntomocomb=15
nmodes=8

# before we read in the per tomo bin combination data, we need to read in the full covariance from the mocks
#These are 3x2pt covs
Bcovdat='Pkk_cov/thps_cov_kids1000_mar30_bandpower_B_apod_0_matrix.dat'
Bcov=np.loadtxt(Bcovdat)
#Emode covariance to compare the amplitude of the Bmode to the expected Emode signal
Ecovdat='Pkk_cov/thps_cov_kids1000_mar30_bandpower_E_apod_0_matrix.dat'
Ecov=np.loadtxt(Ecovdat)

# The first 8x3 rows are w (2 bins, auto, auto cross)
# the next 8x5x2 rows are gammat (8 nodes, 5 sources, 2 lenses)
# then it there are the 8x15 cosmic shear band powers
startpt=nmodes*(3+10)
# and set up a smaller array for each tomobin combination
covizjz=np.zeros((nmodes,nmodes))

# to make different sub plots we count the grid square that we want to plot in
#initialising the counter
gridpos=-1

#information about the file names
filetop='Pkk_data/xi2bandpow_output_K1000_ALL_BLIND_'+str(BLIND)+'_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c'
filetail='nbins_8_Ell_100.0_1500.0_zbins'

# theory curves
#read in the expectation value for the Emode cosmic shear signal
#MD='/home/cech/KiDSLenS/Cat_to_Obs_K1000_P1/'

# read in B mode data per tomo bin combination
for iz in range(1,ntomobin+1):
    for jz in range(iz,ntomobin+1):

        # read in the data
        tomochar='%s_%s'%(iz,jz)
        EBnfile='%s_%s_%s_%s.dat'%(filetop,LFVER,filetail,tomochar)
        indata = np.loadtxt(EBnfile)
        ell=indata[:,0]
        PBB=indata[:,3]  # this is l^2 P_BB/ 2pi
        PEE=indata[:,1]  # this is l^2 P_EE/ 2pi
        
        
        #the expected bandpower_shear_e are l^2 Cl_E/2pi
        #BPtheory=np.loadtxt('%s/ForBG/outputs/test_output_S8_fid_test/bandpower_shear_e/bin_%d_%d.txt'%(MD,jz,iz))
        #ellmin=np.loadtxt('%s/ForBG/outputs/test_output_S8_fid_test/bandpower_shear_e/l_min_vec.txt'%(MD))
        #ellmax=np.loadtxt('%s/ForBG/outputs/test_output_S8_fid_test/bandpower_shear_e/l_max_vec.txt'%(MD))
        #elltheory = (ellmax+ellmin)*0.5
        #ax.plot(elltheory,BPtheory/elltheory*1e7,color='blue',label='$\\P_E/100$')

        #which grid do we want to plot this in?
        grid_x_E=(iz-1)
        grid_y_E=(5-jz)
        ax=axes[grid_y_E,grid_x_E]
        ax.plot(ell,PEE*1e5,color='blue')
        ax.label_outer()
        
        # Adding in the l-scale for the E mode
        if grid_y_E==0:
            ax.xaxis.tick_top()
            ax.set_xlabel('$\ell$')
            ax.xaxis.set_label_position('top')
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.label.set_visible(True)
        
        #breaking up the large Emode covariance matrix to plot the Emode significance for this
        # tomographic bin combination                
        #ipos=startpt+gridpos*nmodes
        #cov_izjz=Ecov[ipos:ipos+nmodes,ipos:ipos+nmodes]
        #diagerr=np.sqrt(np.diagonal(cov_izjz))

        #BP_high = (BPtheory + diagerr)/elltheory*1e7
        #BP_low = (BPtheory - diagerr)/elltheory*1e7
        #ax.fill_between(elltheory, BP_low, BP_high, color='lightgrey',label='$P_E$')
        
        #breaking up the large Bmode covariance matrix to find the significance
        #of the B mode for this tomographic bin combination
        #ipos=startpt+gridpos*nmodes
        #cov_izjz=Bcov[ipos:ipos+nmodes,ipos:ipos+nmodes]

        # now plot the results (present l PBB/2pi rather than l^2 PBB/2pi which is given in the data file)
        # inclue with annotations of the bin combination and p-value 
        #ax.errorbar(ell, PBB/ell*1e7, yerr=diagerr/ell*1e7, fmt='o', color='magenta',label=tomochar,markerfacecolor='none')
        
        #ax.set_ylim(-2.8,5.9)
        ax.annotate(tomochar, xy=(0.3,0.9),xycoords='axes fraction',
            size=14, ha='right', va='top')
        #ax.annotate(pvalchar, xy=(0.95,0.9),xycoords='axes fraction',
        #    size=14, ha='right', va='top')
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.set_xscale('log')
        ax.set_xlim(101,1700.0)
        
        ax.set_yscale('log')
        ax.set_ylim(0.1,160.0)


        grid_x_B=(jz+1)
        grid_y_B=(6-iz)
        
        ax=axes[grid_y_B,grid_x_B]
        ax.label_outer()

        ax.plot(ell,PBB/ell*1e7,color='blue')
        
        if grid_y_B==5:
            ax.set_xlabel('$\ell$')
        
        # Adding in the y-scale for the B mode
        #if grid_x_B==6:
        #    ax.yaxis.tick_right()
        #    ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                

        ax.set_ylim(-2.8,5.9)
        ax.annotate(tomochar, xy=(0.3,0.9),xycoords='axes fraction',
                    size=14, ha='right', va='top')
        #ax.annotate(pvalchar, xy=(0.95,0.9),xycoords='axes fraction',
        #    size=14, ha='right', va='top')
        ax.axhline(y=0, color='black', ls=':')
        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.set_xlim(101,1700.0)


        # I also want to know the significance over all bin combinations
        # to do this I need to construct a large data vector

#add labels
axes[2,0].set_ylabel('$\ell^2 P_E / 2\pi \,\, [10^{-5}]$')
    
axes[3,6].yaxis.label.set_visible(True)
axes[3,6].set_ylabel('$\ell P_B / 2\pi \,\, [10^{-7}]$')
axes[3,6].yaxis.set_label_position('right')

#Blank out the empty cells
for i in range(6):
    blankgrid=6-i
    axes[i,blankgrid].set_visible(False)
    axes[i,blankgrid-1].set_visible(False)


outfile='Pkk_K1000_%s.png'%(LFVER)
plt.savefig(outfile)
plt.show()


