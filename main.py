import os, argparse

import numpy as np

from model import PENG_model
from plot_utils import plot_results

z_init  = 10
z_final = 0

cluster_mass = 13.5  #log10(Mhalo)
n_clusters   = 1000

oc_flag      = True  # flag for overconsumption model. 
oc_eta       = 0      # mass-loading factor

plot_flag    = True
savefigs     = True

if __name__ == "__main__":
    model   = PENG_model()
    
    model.gen_galaxies(1e6)
    model.gen_field_analytic(z_init, z_final)
    
    print('done analytical model')
    
    model.setup_evolve(z_init, z_final)
    model.gen_cluster(cluster_mass, z_init, z_final, n_clusters, oc_flag, oc_eta)
    
    while model.t >= model.t_final and model.condition:
        model.evolve_field()
        model.evolve_cluster() # applies quenching conditions and the mass increase of the galaxies

        model.grow_cluster()
        
        model.update_step() # advances t, z to next step
    
        print('~~~~~~~~~~~~~~~~~~~~~~~')
    
    model.parse_masked_mass_field()
    model.parse_masked_mass_cluster()
    
    plot_results(plot_flag, savefigs, model, z_init, z_final)
    
    total_stel_mass             = np.sum(np.power(10,model.final_mass_cluster_SF)) + np.sum(np.power(10, model.final_mass_cluster_Q))
    total_stel_mass_per_cluster = total_stel_mass/n_clusters
    
    print('Total stellar mass of cluster: ', np.log10(total_stel_mass_per_cluster))
    