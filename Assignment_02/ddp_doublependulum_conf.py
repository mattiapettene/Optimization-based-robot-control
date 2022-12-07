# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
from math import pi, sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

q0 = np.array([ pi , 0.])        # initial configuration
qT = np.array([ 0. , 0])         # goal configuration
dt = 0.005                       # DDP time step
N = 400                          # horizon size

dt_sim = 5e-3                    # time step used for the final simulation
ndt = 1                          # number of integration steps for each control loop

simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'euler'        # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(2)   # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 10.0

PUSH = 1                         # flag to activate four pushes on the second joint (instantaneous variation of the second joint velocity) at N/8, N/4, N/2 and 3*N/4 simulation steps
push_vec = np.array([0,3])       # instantaneous velocy increment of the second joint

SELECTION_MATRIX = 1             # flag to use the selection matrix method
ACTUATION_PENALTY = 0            # flag to use the actuation penalty method

TORQUE_LIMITS = 0                # flag to bound controls

PLOT_TORQUES = 1                 # flag to plot controls
PLOT_JOINT_POS = 1               # flag to plot joint angles
PLOT_JOINT_VEL = 1               # flag to plot joint velocities

use_viewer = True
simulate_real_time = 1           # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 1                      # print some info every PRINT_T seconds
DISPLAY_T = 0.02                 # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [1.2517311573028564, 0.6763767004013062, 0.28195011615753174, 0.3313407003879547, 0.557260274887085, 0.651939868927002, 0.3932540714740753]
