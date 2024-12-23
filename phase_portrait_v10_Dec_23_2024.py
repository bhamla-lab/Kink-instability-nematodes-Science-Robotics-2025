
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:36:16 2023
Code for simulating foot latch extreme cases

@author: itiwari7
"""




import csv
import math
import numpy as np
from IPython.display import Video, clear_output
from elastica._linalg import _batch_matvec, _batch_cross, _batch_dot
from elastica._calculus import difference_kernel_for_block_structure
from elastica._calculus import quadrature_kernel_for_block_structure
from elastica.memory_block.memory_block_rod import make_block_memory_metadata
# import wrappers
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, CallBacks
# import rod class and forces to be applied
from elastica.rod.cosserat_rod import CosseratRod
# import timestepping functions
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica import*
# Import Damping Class
from elastica.dissipation import AnalyticalLinearDamper, LaplaceDissipationFilter
from elastica.timestepper import integrate
from elastica.rod.cosserat_rod import (_compute_bending_twist_strains,
                                       _compute_shear_stretch_strains,
                                       _compute_damping_forces)
from elastica.external_forces import GravityForces, MuscleTorques, NoForces,inplace_addition
from elastica.interaction import InteractionPlane,AnisotropicFrictionalPlane,SlenderBodyTheory
# import call back functions
from elastica.callback_functions import CallBackBaseClass
from collections import defaultdict
from elastica._rotations import _inv_rotate
import scipy.io as sio
# importing the library
import os
 
tfactor=1.0*10**0       #microsecond
lfactor=1.0*10**6       #micron
mfactor=1.0*10**6       #milligram



#%%



class straighten_slowly(NoForces):
    def __init__(self, ramp_up_time, zero_kappa, zero_sigma, target_kappa, target_sigma):
          self. ramp_up_time = ramp_up_time
          self. initial_kappa = zero_kappa
          self. initial_sigma = zero_sigma
          self. target_kappa = target_kappa
          self.target_sigma = target_sigma
    
    def apply_forces(self, system, time: np.float = 0.0):
          factor = min(time/self.ramp_up_time, 1.0)
          system.rest_kappa[:] = self.initial_kappa - factor * (self.initial_kappa - self.target_kappa)
          system.rest_sigma[:] = self.initial_sigma - factor * (self.initial_sigma - self.target_sigma)
    


class capillary_force(NoForces):
    def __init__(self,capillary,base_radius):
        "nothing here"
    
    def apply_forces(self, system, time: np.float = 0.0):
          pos=system.position_collection
          ind1=pos[1]<2*base_radius
          ind2=pos[1]<0*lfactor
          system.external_forces[1][ind1]-=capillary
          system.external_forces[1][ind2]+=capillary  #worm body below the floor experiences no negative force
         
    

def compute_elementwise_bending_energy(self):
        """
        Compute total bending energy of the rod at the instance.
        """

        kappa_diff = self.kappa - self.rest_kappa
        bending_internal_torques = _batch_matvec(self.bend_matrix, kappa_diff)

        return (
            0.5
            * (
                _batch_dot(kappa_diff, bending_internal_torques)
                * self.rest_voronoi_lengths
            )
        )


        # Add call backs
class ContinuumRodCallBack(CallBackBaseClass):
            """
            Call back function for continuum snake
            """
        
            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params
        
            def make_callback(self, system, time, current_step: int):
        
                if current_step % self.every == 0:
        
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(system.position_collection.copy())
                    self.callback_params["velocity"].append(system.velocity_collection.copy())
                    self.callback_params["avg_velocity"].append(
                        system.compute_velocity_center_of_mass()
                    )
        
                    self.callback_params["center_of_mass"].append(
                        system.compute_position_center_of_mass()
                    )
                    self.callback_params["bending_energy"].append(
                        system.compute_bending_energy()
                    )
                    self.callback_params["translational_energy"].append(
                        system.compute_translational_energy()
                    )
                    self.callback_params["rotational_energy"].append(
                        system.compute_rotational_energy()
                    )
                    self.callback_params["vcom"].append(
                        system.compute_velocity_center_of_mass()
                    )
                    self.callback_params["curvature"].append(system.kappa.copy())
        
                    return
                
                
        
def plot_video(plot_params: dict, video_name="video.mp4", margin=0.2, fps=15):
    from matplotlib import pyplot as plt
    import matplotlib.animation as manimation
    from mpl_toolkits import mplot3d

    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["position"])
    com_over_time = np.array(plot_params["center_of_mass"])
    xx=com_over_time[:,0];
    yy=com_over_time[:,1];
    total_time = 10#int(np.around(t[..., -1], 1))
    total_frames = fps * total_time
    step = round(len(t) / total_frames)
    if (step==0):
        step=1
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps,bitrate=500, metadata=metadata)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((np.ptp(positions_over_time[:][0]), np.ptp(positions_over_time[:][1]), np.ptp(positions_over_time[:][2])))  # aspect ratio is 1:1:1 in data space
    #ax.set_xlim(0 - margin, 3 + margin)
    #ax.set_ylim(-1.5 - margin, 1.5 + margin)
    #ax.set_zlim(0, 1)
    ax.view_init(elev=90, azim=-90)
    #ax.set_ylabel("Y Position (micron)", fontsize=12)
    #ax.set_xlabel("X Position (micron)", fontsize=12)
    #ax.set_zlabel("Z Position (micron)", fontsize=12)
    ax.set_ylim(0,1000)
    ax.set_xlim(0,1000)
    
    ax.set_box_aspect((1000, 1000, np.ptp(positions_over_time[:][2])))  # aspect ratio is 1:1:1 in data space
    
    rod_lines_3d = ax.plot(
        positions_over_time[0][0],
        positions_over_time[0][1],
        positions_over_time[0][2],
        linewidth=4,color=[0,0,1]
    )[0]
    #traj_lines_3d = ax.plot(
     #   xx[0],
      #  yy[0],
      #  positions_over_time[0][2],
      #  linewidth=4,color=[0,0,0]
    #)[0]
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    #print(plt.style.available)
    with writer.saving(fig, video_name, dpi=100):
        with plt.style.context("seaborn-white"):
            cnt=0.0
            for time in range(1, len(t), step):
                rod_lines_3d.set_xdata(positions_over_time[time][0])
                rod_lines_3d.set_ydata(positions_over_time[time][1])
               # traj_lines_3d.set_xdata(xx[0:time])
               # traj_lines_3d.set_ydata(xx[0:time])
                rod_lines_3d.set_3d_properties(positions_over_time[time][2])
                ax.set_xlabel('Time=%.6f' %(cnt*(t[1]-t[0])/tfactor), fontsize=12)
                cnt=cnt+step
                writer.grab_frame()
    plt.close(fig)
            
            

def plot_video_zoomout(plot_params: dict, video_name="video.mp4", margin=0.2, fps=15):
    from matplotlib import pyplot as plt
    import matplotlib.animation as manimation
    from mpl_toolkits import mplot3d

    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["position"])
    total_time = 10#int(np.around(t[..., -1], 1))
    total_frames = fps * total_time
    step = round(len(t) / total_frames)
    if (step==0):
        step=1
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    #ax.set_xlim(0 - margin, 3 + margin)
    #ax.set_ylim(-1.5 - margin, 1.5 + margin)
    #ax.set_zlim(0, 1)
    ax.view_init(elev=90, azim=-90)
    #ax.set_ylabel("Y Position (micron)", fontsize=12)
    #ax.set_xlabel("X Position (micron)", fontsize=12)
    #ax.set_zlabel("Z Position (micron)", fontsize=12)
    ax.set_ylim(0,8000)
    ax.set_xlim(0,2000)
    ax.set_box_aspect((2000, 8000, np.ptp(positions_over_time[:][2])))  # aspect ratio is 1:1:1 in data space
    
    rod_lines_3d = ax.plot(
        positions_over_time[0][0],
        positions_over_time[0][1],
        positions_over_time[0][2],
        linewidth=4,color=[0,0,1]
    )[0]
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    #print(plt.style.available)
    with writer.saving(fig, video_name, dpi=100):
        with plt.style.context("seaborn-white"):
            cnt=0.0
            for time in range(1, len(t), step):
                rod_lines_3d.set_xdata(positions_over_time[time][0])
                rod_lines_3d.set_ydata(positions_over_time[time][1])
                rod_lines_3d.set_3d_properties(positions_over_time[time][2])
                ax.set_title('Time=%.6f' %(cnt*(t[1]-t[0])/tfactor), fontsize=12)
                cnt=cnt+step
                writer.grab_frame()
    plt.close(fig)


def data_extractor(plot_params: dict,shearable_rod):
    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["center_of_mass"])
    curves_over_time=np.array(plot_params["position"])
    xx=positions_over_time[:,0];
    yy=positions_over_time[:,1];
    all_positions_over_time = np.array(plot_params["position"])
    ii=0
    flag=0
    for row in all_positions_over_time:                       #Ensuring that the worm at least jumps one body length, otherwise reject
        if(min(row[1])>20.0*base_radius):
            contact_lost=ii
            flag=0
            break
        else:
            flag=1
        ii=ii+1
        

    bending_energy = np.array(plot_params["bending_energy"])
    translational_energy = np.array(plot_params["translational_energy"])
    rotational_energy = np.array(plot_params["rotational_energy"])
    vcom = np.array(plot_params["vcom"])
    mass = np.sum(shearable_rod.mass)
    kecom = 0.5*mass*vcom*vcom
    kecomx=np.zeros(bending_energy.shape)
    kecomy=np.zeros(bending_energy.shape)
    kecomz=np.zeros(bending_energy.shape)
    ii=0
    for row in kecom:
        kecomx[ii]=row[0]
        kecomy[ii]=row[1]
        kecomz[ii]=row[1]
        ii=ii+1
    if(flag==0):    
        take_off_angle=np.arctan2(yy[contact_lost]-yy[0], xx[contact_lost]-xx[0]) * 180 / np.pi
        print(take_off_angle)
        print('Worm took off!')
        return kecomy[contact_lost], bending_energy[contact_lost], rotational_energy[contact_lost], take_off_angle,curves_over_time
    else:
        print('Worm stuck!')
        return np.NaN, np.NaN, np.NaN, np.NaN,curves_over_time    #print(max(kecomy))
    #print(kecomy[contact_lost])



def save_position_data(plot_params: dict, video_name="video.mp4", margin=0.2, fps=15):
            from matplotlib import pyplot as plt
            import matplotlib.animation as manimation
            from mpl_toolkits import mplot3d
        
            t = np.array(plot_params["time"])
            positions_over_time = np.array(plot_params["position"])
            with open ('Example.csv','w',newline = '') as csvfile:
                my_writer = csv.writer(csvfile, delimiter = ' ')
                my_writer.writerows(positions_over_time[:][0][:]) 



class RodSimulator(BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    pass







xgrid=21
ygrid=18
ke=np.zeros([xgrid, ygrid])
be=np.zeros([xgrid, ygrid])
re=np.zeros([xgrid, ygrid])
t_angle=np.zeros([xgrid, ygrid])

final_time = 0.001*tfactor
dt = 5e-9*tfactor    
total_steps = int(final_time / dt)

alpha_vals = np.linspace(-50.0,70.0,xgrid)
theta_vals = np.linspace(-20.0,80.0,ygrid)
alpha=alpha_vals[0]
theta=theta_vals[0]

filename='pose_alpha_'+'%3.2f' %(alpha)+'_theta_'+'%3.2f' %(theta)+'.csv'
with open(filename, mode ='r')as file:
           
    # reading the CSV file
    csvFile = csv.reader(file)
    i=0
    # displaying the contents of the CSV file
    for lines in csvFile:
        i+=1
        
n_nodes_total=i
curves=np.zeros([3,n_nodes_total])


c1=0
while c1<xgrid:
    c2=0
    while c2<ygrid:    
        alpha=alpha_vals[c1]#10#-10#alpha_vals[c1]
        theta=theta_vals[c2]#-20#60#theta_vals[c2]
        print([alpha, theta])
        filename='pose_alpha_'+'%3.2f' %(alpha)+'_theta_'+'%3.2f' %(theta)+'.csv'
        filename_video='pose_alpha_'+'%3.2f' %(alpha)+'_theta_'+'%3.2f' %(theta)+'.mp4'
        filename_curves='pose_alpha_'+'%3.2f' %(alpha)+'_theta_'+'%3.2f' %(theta)+'_curves.mat'
        with open(filename, mode ='r')as file:
          # reading the CSV file
          csvFile = csv.reader(file)
          i=0
          # displaying the contents of the CSV file
          for lines in csvFile:
              i+=1
        
        n_nodes_total=i
        positions=np.zeros([3,n_nodes_total])
        # opening the CSV file
        with open(filename, mode ='r')as file:
          # reading the CSV file
          csvFile = csv.reader(file)
          i=0
          # displaying the contents of the CSV file
          for lines in csvFile:
              #if(i>0):
              positions[:,i]=lines
              print(lines)
              i+=1
        
        aa=[]
        aa.append(positions[0][0:n_nodes_total:2])
        aa.append(positions[1][0:n_nodes_total:2])
        aa.append(positions[2][0:n_nodes_total:2])
        positions=np.array(aa)
        positions[1,:]=positions[1,:]-min(positions[1,:])
        n_nodes=len(positions[1])
        positions=positions*(10**-6)*lfactor   #because source shape data in the file is already in microns
        
        n_elem=n_nodes-1
        directors=np.zeros([3,3,n_elem])
        d3=np.zeros([1,3,n_elem])
        tempx=np.diff(positions[0,:])
        tempy=np.diff(positions[1,:])
        tempmod=np.sqrt(tempx**2+tempy**2)
        rod_len=np.sum(tempmod)
        tempx=tempx/tempmod
        tempy=tempy/tempmod
        directors[2,0,:]=tempx
        directors[2,1,:]=tempy
        directors[1,2,:]=-np.ones(np.shape(tempx))
        directors[0,0,:]=directors[1,1,:]*directors[2,2,:]-directors[1,2,:]*directors[2,1,:]
        directors[0,1,:]=-(directors[1,0,:]*directors[2,2,:]-directors[1,2,:]*directors[2,0,:])
                
                
                
        Rod_sim = RodSimulator()
        
        
        # Define rod parameters
        n_elem = n_nodes-1
        start = np.array(positions[:,0])
        direction = np.array([0.1, 0.0, np.sqrt(1-0.1**2)])
        normal = np.array([0.0, 1.0, 0.0])
        base_length = rod_len      
        base_radius = 0.5*(rod_len/20)     #division by 20 because S. Carpocapsae has aspect
                                        #ratio of ~20, multiply by 0.5 to make diameter
                                        #into radius
        base_area = np.pi * base_radius ** 2
        density = 1000.0*mfactor/(lfactor**3)   
        nu_worm =2*10**3*mfactor/(lfactor*tfactor)               
        E = 1*10**7*mfactor/(lfactor*(tfactor**2))   
        poisson_ratio = 0.5
        shear_modulus = E / (poisson_ratio + 1.0)
        
        
        shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            0.0,
            E,
            shear_modulus=shear_modulus,
            position=positions, 
            directors=directors
        )
        
        
        be_vals=compute_elementwise_bending_energy(shearable_rod)
                
        
        og_bend_matrix=shearable_rod.bend_matrix
        og_shear_matrix = shearable_rod.shear_matrix
        target_kappa=shearable_rod.rest_kappa
        target_sigma=shearable_rod.rest_sigma
        
        initial_kappa = shearable_rod.kappa
        initial_sigma = shearable_rod.sigma
        shearable_rod.rest_kappa = shearable_rod.kappa
        shearable_rod.rest_sigma = shearable_rod.sigma
        
        
        #add rod
        Rod_sim.append(shearable_rod)
        
        
        # Add slender body forces
        dynamic_viscosity = 1.82*10**-5*mfactor/(lfactor*tfactor)    
        Rod_sim.add_forcing_to(shearable_rod).using(
            SlenderBodyTheory, dynamic_viscosity=dynamic_viscosity
        )
                
        
        # Add gravitational forces
        gravitational_acc = -9.806*lfactor/(tfactor**2)  
        
        Rod_sim.add_forcing_to(shearable_rod).using(
            GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
        )
        print("Gravity now acting on shearable rod")
        
        

        # Define friction force parameters
        origin_plane = np.array([0.0, -1.0*base_radius, 0.0])
        normal_plane = np.array([0.0, 1.0, 0.0])
        slip_velocity_tol = 1e-8
        froude = 0.1
        mu = 50
        kinetic_mu_array = np.array(
            [1.0 * mu, 1.0 * mu, 1.0 * mu]
        )  # [forward, backward, sideways]
        static_mu_array = 2 * kinetic_mu_array
        
        # Add friction forces to the substrate
        Rod_sim.add_forcing_to(shearable_rod).using(
            AnisotropicFrictionalPlane,
            k=0.1,
            nu=50,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
        print("Friction forces added to the sustrate")
        
        

        
        Rod_sim.dampen(shearable_rod).using(AnalyticalLinearDamper,damping_constant=nu_worm,time_step = dt)
        
        ramp_up_time=2*10**-5*tfactor
        Rod_sim.add_forcing_to(shearable_rod).using(straighten_slowly,
                    ramp_up_time, 
                    initial_kappa,
                    initial_sigma,
                    target_kappa,
                    target_sigma
                    )
        print("Rod will straighten slowly.")
        
        
        
        capillary=1*10**-7*((mfactor*lfactor)/tfactor**2)
        Rod_sim.add_forcing_to(shearable_rod).using(capillary_force,capillary,base_radius)
        print("Rod foot will be stuck by capillary forces.")
        
        
        pp_list = defaultdict(list)
        Rod_sim.collect_diagnostics(shearable_rod).using(
            ContinuumRodCallBack, step_skip=1, callback_params=pp_list
        )
        print("Callback function added to the simulator")
        
                
        Rod_sim.finalize()
        print("Total steps", total_steps)
        timestepper = PositionVerlet()
        integrate(timestepper, Rod_sim, final_time, total_steps)
        plot_video(pp_list, video_name=filename_video, margin=0.2, fps=30)
        
        Video(filename_video)
        
        
        plot_video_zoomout(pp_list, video_name='zoomout'+filename_video, margin=0.2, fps=30)
        
        Video('zoomout'+filename_video)

        ke[c1][c2], be[c1][c2], re[c1][c2], t_angle[c1][c2],curves = data_extractor(pp_list,shearable_rod)
        tempdict={'curves':curves}
        sio.savemat(filename_curves,tempdict)
        del Rod_sim, pp_list, tempdict,shearable_rod
        c2=c2+1
    c1=c1+1

savedict = {
    'ke':ke,
    'be':be,
    're':re,
    't_angle':t_angle,
    'tscale':tfactor,
    'mscale':mfactor,
    'lscale':lfactor,
    'dt':dt,
    'final_time':final_time,
    'aspect_ratio':20.0,
    'friction_coeff':mu,
    'n_nodes':n_nodes,
    'nu_worm':nu_worm,
    'air_viscosity':dynamic_viscosity,
    'young':E,
    'poisson_ratio':poisson_ratio,
    'density':density,
    'capillary': capillary ,   
    'alpha_vals':alpha_vals,
    'theta_vals':theta_vals
}
        
sio.savemat('foot_scan_diagnostic_data.mat', savedict)

        

        
    
    

       