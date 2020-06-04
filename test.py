import os, argparse
import numpy as np
import matplotlib.pyplot as plt

from model import PENG_model
from old_code import overconsumption


from astropy.cosmology import Planck15 as cosmo, z_at_value

z_init  = 10
z_final = 0

cluster_mass = 13.5  #log10(Mhalo)
n_clusters   = 1000

oc_flag      = True      #
oc_eta       = 2

logMh_range  = np.arange(9,14,0.1)
Mh_range     = np.power(10, logMh_range)
z_range      = np.arange(0,10,0.1)

if __name__ == "__main__":
    model        = PENG_model()
    model.oc_eta = 1
    
    tt = np.array([cosmo.lookback_time(z).value - model.t_delay_2(Mh_range, z) for z in z_range])
    
    fig,ax = plt.subplots()
    ax.contourf(z_range, logMh_range, tt.T, np.arange(0,14,2), extend='both')

plt.show()