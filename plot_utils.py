import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import Planck15 as cosmo, z_at_value

class plot_results:
    def __init__(self, plotting_flag, savefigs, model, z_init, z_final): 
        
        self.savefigs = savefigs
        
        if plotting_flag:
            
            self.init_pop(model)
            self.final_pop(model, z_init, z_final)
            
            self.QFs_QE(model)
            
            self.delay_times(model, z_init, z_final)
            
            plt.show()

    def init_pop(self, model):
        '''
        Initial star forming field
        '''
        x_range_init = np.arange(model.logM_min,model.logM_max+0.05,0.1)
            
        fig,ax = plt.subplots()
        y_init, x_init,_ = ax.hist(model.sf_masses, log=True, bins=x_range_init)
        ax.semilogy(x_range_init, model.schechter_SMF_func(x_range_init)*y_init[abs(x_init[:-1] - 4) < 0.001]/model.schechter_SMF_func(4) )
        
    def final_pop(self, model, z_init, z_final):
        '''
        Parameters
        ----------
        model : class
            Contains the evolved cluster and field data.
        z_init : float
            redshift at which cluster & field are initiated.
        z_final : float
            redshift of observation.

        Returns
        -------
        figures: 
        '''
        x_range = np.arange(8,12,0.1)
        
        fig,ax = plt.subplots()
        y_sf,x_sf,_ = ax.hist(model.final_mass_field_SF, bins=np.arange(6,14,0.1), log=True, alpha=0.5, color='b')
        y_q,x_q,_   = ax.hist(model.final_mass_field_Q, bins=np.arange(6,14,0.1), log=True, alpha=0.5, color='r')
        ax.semilogy(x_range, model.phi_sf_interp(x_range,z_final) * y_sf[abs(x_sf[:-1] - 10) < 0.001]/model.phi_sf_interp(10,z_final) )
        ax.semilogy(x_range, model.phi_q_interp(x_range,z_final)  * y_sf[abs(x_q[:-1] - 10) < 0.001]/model.phi_sf_interp(10,z_final) )
        ax.set_xlim([8.5,12])
        
        fig,ax = plt.subplots()
        y_sf,x_sf,_ = ax.hist(model.final_mass_cluster_SF, bins=np.arange(6,14,0.1), log=True, alpha=0.5, color='b')
        y_q,x_q,_   = ax.hist(model.final_mass_cluster_Q, bins=np.arange(6,14,0.1), log=True, alpha=0.5, color='r')
        ax.semilogy(x_range, model.phi_sf_interp(x_range,z_final) * y_sf[abs(x_sf[:-1] - 10) < 0.001]/model.phi_sf_interp(10,z_final) )
        ax.semilogy(x_range, model.phi_q_interp(x_range,z_final)  * y_sf[abs(x_q[:-1] - 10) < 0.001]/model.phi_sf_interp(10,z_final) )
        ax.set_xlim([8.5,12])
        
    def QFs_QE(self, model):
        hist_sf_field, bins   = np.histogram(model.final_mass_field_SF, bins=np.arange(6,14,0.1))
        hist_q_field, bins    = np.histogram(model.final_mass_field_Q, bins=np.arange(6,14,0.1))
        
        hist_sf_cluster, bins = np.histogram(model.final_mass_cluster_SF, bins=np.arange(6,14,0.1))
        hist_q_cluster, bins  = np.histogram(model.final_mass_cluster_Q, bins=np.arange(6,14,0.1))
        
        bins_midp = (bins[:-1] + bins[1:])/2
        QF_field  = hist_q_field / (hist_q_field + hist_sf_field)
        QF_cluster  = hist_q_cluster / (hist_q_cluster + hist_sf_cluster)
        
        QE = (QF_cluster - QF_field)/(1 - QF_field)
        
        fig,ax = plt.subplots()
        ax.plot(bins_midp, QF_field)
        ax.plot(bins_midp, QF_cluster)
        ax.set_xlim([9,11.5])
        
        fig,ax = plt.subplots()
        ax.plot(bins_midp, QE)
        ax.set_xlim([9,11.5])
        
    def delay_times(self, model, z_init, z_final):
        z_range     = np.arange(z_final, z_init, 0.1)
        
        logMh_range = np.arange(7,15,0.1)
        Mh_range    = np.power(10, logMh_range)
        
        xx   = []
        xx_2 = []
        yy   = []
        zz   = []
        
        for z in z_range:
            xx.append(np.zeros(len(Mh_range)) + z)
            xx_2.append(np.zeros(len(Mh_range)) + cosmo.lookback_time(z).value)
            yy.append(np.log10(model.M_star(Mh_range, z)))
            zz.append(cosmo.lookback_time(z).value - model.t_delay_2(Mh_range, z))
        
        
        xx   = np.array(xx).flatten()
        xx_2 = np.array(xx_2).flatten()
        yy   = np.array(yy).flatten()
        zz   = np.array(zz).flatten()
        
        fig,ax = plt.subplots()
        #contourf_ = ax.tricontourf(xx,yy,zz, np.arange(0,13,0.1), extend='both')
        
        ax.scatter(model.infall_z_Q, model.final_mass_cluster_Q, s=0.1, c='r')
        ax.scatter(model.infall_z_SF, model.final_mass_cluster_SF, s=0.1, c='b')
        ax.set_xlabel('Redshift [z]')
        ax.set_ylabel('Stellar Mass [log($M_*$/$M_\odot$)]')
        ax.set_ylim([8,12])
        #cbar      = fig.colorbar(contourf_, label='Quenched due to OC')
        
        
        
        
        
        fig,ax = plt.subplots()
        #contourf_ = ax.tricontourf(xx_2,yy,zz, np.arange(0,13,0.1), extend='both')
        
        ax.scatter(cosmo.lookback_time(model.infall_z_Q).value, model.final_mass_cluster_Q, s=0.1, c='r')
        ax.scatter(cosmo.lookback_time(model.infall_z_SF).value, model.final_mass_cluster_SF, s=0.1, c='b')
        ax.set_xlabel('Lookback Time [Gyr]')
        ax.set_ylabel('Stellar Mass [log($M_*$/$M_\odot$)]')
        ax.set_ylim([8,12])
        #cbar      = fig.colorbar(contourf_, label='Quenched due to OC')
        
        
        
        
        fig,ax = plt.subplots()
        #contourf_ = ax.tricontourf(xx_2,yy,zz, np.arange(0,13,0.1), extend='both')
        
        temp_z      = np.ma.copy(model.infall_z)
        temp_z.mask = np.ma.nomask
        temp_m      = np.ma.copy(model.infall_Ms)
        temp_m.mask = np.ma.nomask
        temp_m      = np.ma.log10(temp_m)
        
        ax.scatter(cosmo.lookback_time(temp_z).value, temp_m, s=0.1, c='r')
        ax.set_xlabel('Lookback Time [Gyr]')
        ax.set_ylabel('Stellar Mass [log($M_*$/$M_\odot$)]')
        ax.set_ylim([8,12])
        #cbar      = fig.colorbar(contourf_, label='Quenched due to OC')
        
        
        