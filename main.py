import json
import numpy as np

from time import time
from multiprocessing import Pool

from model import PENG_model
from plot_utils import plot_results

if __name__ == "__main__":
    
    with open("params.json") as paramfile:
        params = json.load(paramfile)   
    
    z_init_field   = params['model_params']['z_init_field']
    z_init_cluster = params['model_params']['z_init_cluster']
    
    n_cores = params['model_setup']['n_cores']
    p       = Pool(n_cores, maxtasksperchild=50)
    
    model_c   = PENG_model(params, z_init_cluster)  # initializes the model class
    
    model_c.gen_galaxies()                            # generates the SF population at z_init
    model_c.gen_field_analytic(p)                            # generates the PENG model predictions for the field, analytically
                                                             #      used in plotting & determining cluster galaxy growth
    
    print('\t Done analytical model')
    
    model_c.setup_evolve()                                         # prepares some vars & RK45
    model_c.gen_cluster() # first galaxies assigned to cluster at z_init
    
    start_time_1 = time()
    while model_c.t >= model_c.t_final and model_c.condition:
        '''
        Generates the cluster SMFs'
        '''
        start_time_2 = time()
        model_c.evolve_field(p)
        model_c.evolve_cluster(p) # applies quenching conditions and the mass increase of the galaxies
        
        model_c.grow_cluster()
        model_c.update_ssfr_params()
        
        
        model_c.update_step() # advances t, z to next step
        
        print('\t time per step: {:.2f}s'.format(time() - start_time_2))
        print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Total ellapsed time for Cluster: {:.2f}s'.format(time() - start_time_1))
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    model_f   = PENG_model(params, z_init_field)
    
    if z_init_field ==  z_init_cluster:
        model_f = model_c
    else:
        model_f.gen_galaxies()
        #model_f.sf_masses = np.copy(model_c.sf_masses)  #save ourselves some time and re-use the initial SF population
        model_f.gen_field_analytic(p)
        
        model_f.setup_evolve()
        
        start_time_1 = time()
        while model_f.t >= model_f.t_final and model_f.condition:
            '''
            Generates the field, which can start at a different redshift from the cluster
            '''
            start_time_2 = time()
            model_f.evolve_field()
            
            model_f.update_ssfr_params()
            
            model_f.update_step() # advances t, z to next step
            
            #print('\t time per step: {:.2f}s'.format(time() - start_time_2))
            #print('~~~~~~~~~~~~~~~~~~~~~~~')
        print('Total ellapsed time for Field: {:.2f}s'.format(time() - start_time_1))
    
    model_f.parse_masked_mass_field()
    model_c.parse_masked_mass_cluster()
    
    plot_results(params, model_c, model_f)
    
    total_stel_mass             = np.sum(np.power(10,model_c.final_mass_cluster_SF)) + np.sum(np.power(10, model_c.final_mass_cluster_Q))
    total_stel_mass_per_cluster = total_stel_mass/params['model_setup']['n_cluster']
    
    print('Total stellar mass of cluster: ', np.log10(total_stel_mass_per_cluster))
    
    p.close()
    p.join()
    
