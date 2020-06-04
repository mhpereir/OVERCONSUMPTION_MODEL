import os, argparse

from model import PENG_model
from plot_utils import plot_results
from old_code import overconsumption

z_init  = 10
z_final = 0

cluster_mass = 13.5  #log10(Mhalo)
n_clusters   = 1000

oc_flag      = True  # flag for overconsumption model. 
oc_eta       = 1     # mass-loading factor

plot_flag    = True

if __name__ == "__main__":
    model   = PENG_model()
    
    model.gen_galaxies(1e6)
    model.gen_field_analytic(z_init, z_final)
    
    print('done analytical model')
    
    model.setup_evolve(z_init, z_final)
    model.gen_cluster(cluster_mass, z_init, z_final, n_clusters, oc_flag, oc_eta)
    
    while model.t >= model.t_final and model.condition:
        model.evolve_field()
        model.grow_cluster()
        model.evolve_cluster()
    
    model.parse_masked_mass_field()
    model.parse_masked_mass_cluster()
    
    plot_results(plot_flag, model, z_init, z_final)
    
    