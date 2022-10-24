# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:31:22 2022

@author: Gianluigi Grandesso
"""

import time
import numpy as np
import hw1_conf as conf
from plot_utils import create_empty_figure
from tsid_biped import TsidBiped
import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

print("".center(conf.LINE_WIDTH,'#'))
print(" Test Walking ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_COM = 1
PLOT_COP = 1
PLOT_FOOT_TRAJ = 0
PLOT_TORQUES = 0
PLOT_JOINT_VEL = 0

data = np.load(conf.DATA_FILE_TSID)

tsid_bip = TsidBiped(conf, conf.viewer)

N = data['com'].shape[1]
N_pre  = int(conf.T_pre/conf.dt)        
N_post = int(conf.T_post/conf.dt)

com_pos = np.empty((3, N+N_post))*nan
com_vel = np.empty((3, N+N_post))*nan
com_acc = np.empty((3, N+N_post))*nan
x_LF   = np.empty((3, N+N_post))*nan
dx_LF  = np.empty((3, N+N_post))*nan
ddx_LF = np.empty((3, N+N_post))*nan
ddx_LF_des = np.empty((3, N+N_post))*nan
x_RF   = np.empty((3, N+N_post))*nan
dx_RF  = np.empty((3, N+N_post))*nan
ddx_RF = np.empty((3, N+N_post))*nan
ddx_RF_des = np.empty((3, N+N_post))*nan
f_RF = np.zeros((6, N+N_post))
f_LF = np.zeros((6, N+N_post))
cop_RF = np.zeros((2, N+N_post))
cop_LF = np.zeros((2, N+N_post))
tau    = np.zeros((tsid_bip.robot.na, N+N_post))
q_log  = np.zeros((tsid_bip.robot.nq, N+N_post))
v_log  = np.zeros((tsid_bip.robot.nv, N+N_post))

contact_phase = data['contact_phase']
com_pos_ref = np.asarray(data['com'])
squat_pos_ref = np.array([com_pos_ref[0,:],com_pos_ref[1,:],conf.squat_height*np.ones(N)])
com_vel_ref = np.asarray(data['dcom'])
squat_vel_ref = np.array([com_vel_ref[0,:],com_vel_ref[1,:],np.zeros(N)])
com_acc_ref = np.asarray(data['ddcom'])
squat_acc_ref = np.array([com_acc_ref[0,:],com_acc_ref[1,:],np.zeros(N)])
x_RF_ref    = np.asarray(data['x_RF'])
dx_RF_ref   = np.asarray(data['dx_RF'])
ddx_RF_ref  = np.asarray(data['ddx_RF'])
x_LF_ref    = np.asarray(data['x_LF'])
dx_LF_ref   = np.asarray(data['dx_LF'])
ddx_LF_ref  = np.asarray(data['ddx_LF'])
cop_ref     = np.asarray(data['cop'])
com_acc_des = np.empty((3, N+N_post))*nan # acc_des = acc_ref - Kp*pos_err - Kd*vel_err

x_rf   = tsid_bip.get_placement_RF().translation
offset = x_rf - x_RF_ref[:,0]
for i in range(N):
    com_pos_ref[:,i] += offset + np.array([0.,0.,0.0])
    x_RF_ref[:,i] += offset
    x_LF_ref[:,i] += offset

t = -conf.T_pre
q, v = tsid_bip.q, tsid_bip.v

for i in range(-N_pre, N+N_post):
    time_start = time.time()
    
    if i==0:
        print("Starting to walk (remove contact left foot)")
        tsid_bip.remove_contact_LF()
    elif i>0 and i<N-1:
        if contact_phase[i] != contact_phase[i-1]:
            print("Time %.3f Changing contact phase from %s to %s"%(t, contact_phase[i-1], contact_phase[i]))
            if contact_phase[i] == 'left':
                tsid_bip.add_contact_LF()
                tsid_bip.remove_contact_RF()
            else:
                tsid_bip.add_contact_RF()
                tsid_bip.remove_contact_LF()
    
    if i<0:
        tsid_bip.set_com_ref(com_pos_ref[:,0], 0*com_vel_ref[:,0], 0*com_acc_ref[:,0])
        if conf.SQUAT:
            tsid_bip.set_squat_ref(squat_pos_ref[:,0], 0*squat_vel_ref[:,0], 0*squat_acc_ref[:,0])
    elif i<N:
        tsid_bip.set_com_ref(com_pos_ref[:,i], com_vel_ref[:,i], com_acc_ref[:,i])
        tsid_bip.set_LF_3d_ref(x_LF_ref[:,i], dx_LF_ref[:,i], ddx_LF_ref[:,i])
        tsid_bip.set_RF_3d_ref(x_RF_ref[:,i], dx_RF_ref[:,i], ddx_RF_ref[:,i])
        if conf.SQUAT:
            tsid_bip.set_squat_ref(squat_pos_ref[:,i], squat_vel_ref[:,i], squat_acc_ref[:,i])

    HQPData = tsid_bip.formulation.computeProblemData(t, q, v)

    sol = tsid_bip.solver.solve(HQPData)
    if(sol.status!=0):
        print("QP problem could not be solved! Error code:", sol.status)
        break
    if norm(v,2)>40.0:
        print("Time %.3f Velocities are too high, stop everything!"%(t), norm(v))
        break
    
    if i>0:
        q_log[:,i] = q
        v_log[:,i] = v
        tau[:,i] = tsid_bip.formulation.getActuatorForces(sol)
    dv = tsid_bip.formulation.getAccelerations(sol)
    
    if i>=0:
        com_pos[:,i] = tsid_bip.robot.com(tsid_bip.formulation.data())
        com_vel[:,i] = tsid_bip.robot.com_vel(tsid_bip.formulation.data())
        com_acc[:,i] = tsid_bip.comTask.getAcceleration(dv)
        com_acc_des[:,i] = tsid_bip.comTask.getDesiredAcceleration
        x_LF[:,i], dx_LF[:,i], ddx_LF[:,i] = tsid_bip.get_LF_3d_pos_vel_acc(dv)
        if not tsid_bip.contact_LF_active:
            ddx_LF_des[:,i] = tsid_bip.leftFootTask.getDesiredAcceleration[:3]
        x_RF[:,i], dx_RF[:,i], ddx_RF[:,i] = tsid_bip.get_RF_3d_pos_vel_acc(dv)
        if not tsid_bip.contact_RF_active:
            ddx_RF_des[:,i] = tsid_bip.rightFootTask.getDesiredAcceleration[:3]
        
        if tsid_bip.formulation.checkContact(tsid_bip.contactRF.name, sol):
            T_RF = tsid_bip.contactRF.getForceGeneratorMatrix
            f_RF[:,i] = T_RF @ tsid_bip.formulation.getContactForce(tsid_bip.contactRF.name, sol)
            if(f_RF[2,i]>1e-3): 
                cop_RF[0,i] = f_RF[4,i] / f_RF[2,i]
                cop_RF[1,i] = -f_RF[3,i] / f_RF[2,i]
        if tsid_bip.formulation.checkContact(tsid_bip.contactLF.name, sol):
            T_LF = tsid_bip.contactRF.getForceGeneratorMatrix
            f_LF[:,i] = T_LF @ tsid_bip.formulation.getContactForce(tsid_bip.contactLF.name, sol)
            if(f_LF[2,i]>1e-3): 
                cop_LF[0,i] = f_LF[4,i] / f_LF[2,i]
                cop_LF[1,i] = -f_LF[3,i] / f_LF[2,i]

    if i%conf.PRINT_N == 0:
        print("Time %.3f"%(t))
        if tsid_bip.formulation.checkContact(tsid_bip.contactRF.name, sol) and i>=0:
            print("\tnormal force %s: %.1f"%(tsid_bip.contactRF.name.ljust(20,'.'), f_RF[2,i]))

        if tsid_bip.formulation.checkContact(tsid_bip.contactLF.name, sol) and i>=0:
            print("\tnormal force %s: %.1f"%(tsid_bip.contactLF.name.ljust(20,'.'), f_LF[2,i]))

        print("\ttracking err %s: %.3f"%(tsid_bip.comTask.name.ljust(20,'.'), norm(tsid_bip.comTask.position_error, 2)))
        print("\t||v||: %.3f\t ||dv||: %.3f"%(norm(v, 2), norm(dv)))

    q, v = tsid_bip.integrate_dv(q, v, dv, conf.dt)
    t += conf.dt

    if conf.PUSH and i==int(N/2):
        data_bip = tsid_bip.formulation.data()
        if tsid_bip.contact_LF_active:
            J_LF = tsid_bip.contactLF.computeMotionTask(0.0, q, v, data_bip).matrix
        else:
            J_LF = np.zeros((0, tsid_bip.model.nv))
        if tsid_bip.contact_RF_active:
            J_RF = tsid_bip.contactRF.computeMotionTask(0.0, q, v, data_bip).matrix
        else:
            J_RF = np.zeros((0, tsid_bip.model.nv))
        J = np.vstack((J_LF, J_RF))
        J_com = tsid_bip.comTask.compute(t, q, v, data_bip).matrix
        A = np.vstack((J_com, J))
        b = np.concatenate((np.array(conf.push_robot_com_vel), np.zeros(J.shape[0])))
        v += np.linalg.lstsq(A, b, rcond=-1)[0]

    if i%conf.DISPLAY_N == 0: tsid_bip.display(q)

    time_spent = time.time() - time_start
    if(time_spent < conf.dt): time.sleep(conf.dt-time_spent)
    
# PLOT STUFF
time = np.arange(0.0, (N+N_post)*conf.dt, conf.dt)

if PLOT_COM:
    (f, ax) = create_empty_figure(3,1)
    for i in range(3):
        ax[i].plot(time, com_pos[i,:], label='CoM '+str(i))
        ax[i].plot(time[:N], com_pos_ref[i,:], 'r:', label='CoM Ref '+str(i))
        if conf.SQUAT and i!=2:
            ax[i].plot(time[:N], squat_pos_ref[i,:], 'g:', label='CoM Squat Ref '+str(i))
        if conf.SQUAT and i==2:
            ax[i].plot(time[:N], np.ones(N)*conf.squat_height, 'g:', label='CoM Squat Ref 2')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('CoM [m]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
    
    (f, ax) = create_empty_figure(3,1)
    for i in range(3):
        ax[i].plot(time, com_vel[i,:], label='CoM Vel '+str(i))
        ax[i].plot(time[:N], com_vel_ref[i,:], 'r:', label='CoM Vel Ref '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('CoM Vel [m/s]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
        
    (f, ax) = create_empty_figure(3,1)
    for i in range(3):
        ax[i].plot(time, com_acc[i,:], label='CoM Acc '+str(i))
        ax[i].plot(time[:N], com_acc_ref[i,:], 'r:', label='CoM Acc Ref '+str(i))
        ax[i].plot(time, com_acc_des[i,:], 'g--', label='CoM Acc Des '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('CoM Acc [m/s^2]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)

if PLOT_COP:
    (f, ax) = create_empty_figure(2,1)
    for i in range(2):
        ax[i].plot(time, cop_LF[i,:], label='CoP LF '+str(i))
        ax[i].plot(time, cop_RF[i,:], label='CoP RF '+str(i))
        if i==0:   
            ax[i].plot([time[0], time[-1]], [-conf.lxn, -conf.lxn], ':', label='CoP Lim '+str(i))
            ax[i].plot([time[0], time[-1]], [conf.lxp, conf.lxp], ':', label='CoP Lim '+str(i))
        elif i==1: 
            ax[i].plot([time[0], time[-1]], [-conf.lyn, -conf.lyn], ':', label='CoP Lim '+str(i))
            ax[i].plot([time[0], time[-1]], [conf.lyp, conf.lyp], ':', label='CoP Lim '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('CoP [m]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
    
if PLOT_FOOT_TRAJ:
    for i in range(3):
        plt.figure()
        plt.plot(time, x_RF[i,:], label='x RF '+str(i))
        plt.plot(time[:N], x_RF_ref[i,:], ':', label='x RF ref '+str(i))
        plt.plot(time, x_LF[i,:], label='x LF '+str(i))
        plt.plot(time[:N], x_LF_ref[i,:], ':', label='x LF ref '+str(i))
        plt.legend()
        
    #for i in range(3):
    #    plt.figure()
    #    plt.plot(time, dx_RF[i,:], label='dx RF '+str(i))
    #    plt.plot(time[:N], dx_RF_ref[i,:], ':', label='dx RF ref '+str(i))
    #    plt.plot(time, dx_LF[i,:], label='dx LF '+str(i))
    #    plt.plot(time[:N], dx_LF_ref[i,:], ':', label='dx LF ref '+str(i))
    #    plt.legend()
    #    
    #for i in range(3):
    #    plt.figure()
    #    plt.plot(time, ddx_RF[i,:], label='ddx RF '+str(i))
    #    plt.plot(time[:N], ddx_RF_ref[i,:], ':', label='ddx RF ref '+str(i))
    #    plt.plot(time, ddx_RF_des[i,:], '--', label='ddx RF des '+str(i))
    #    plt.plot(time, ddx_LF[i,:], label='ddx LF '+str(i))
    #    plt.plot(time[:N], ddx_LF_ref[i,:], ':', label='ddx LF ref '+str(i))
    #    plt.plot(time, ddx_LF_des[i,:], '--', label='ddx LF des '+str(i))
    #    plt.legend()
    
if PLOT_TORQUES:        
    plt.figure()
    for i in range(tsid_bip.robot.na):
        tau_normalized = 2*(tau[i,:]-tsid_bip.tau_min[i]) / (tsid_bip.tau_max[i]-tsid_bip.tau_min[i]) - 1
        # plot torques only for joints that reached 50% of max torque
        if np.max(np.abs(tau_normalized))>0.5:
            plt.plot(time, tau_normalized, alpha=0.5, label=tsid_bip.model.names[i+2])
    plt.plot([time[0], time[-1]], 2*[-1.0], ':')
    plt.plot([time[0], time[-1]], 2*[1.0], ':')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Normalized Torque')
    leg = plt.legend()
    leg.get_frame().set_alpha(0.5)
    
if PLOT_JOINT_VEL:
    plt.figure()
    for i in range(tsid_bip.robot.na):
        v_normalized = 2*(v_log[6+i,:]-tsid_bip.v_min[i]) / (tsid_bip.v_max[i]-tsid_bip.v_min[i]) - 1
        # plot v only for joints that reached 50% of max v
        if np.max(np.abs(v_normalized))>0.5:
            plt.plot(time, v_normalized, alpha=0.5, label=tsid_bip.model.names[i+2])
    plt.plot([time[0], time[-1]], 2*[-1.0], ':')
    plt.plot([time[0], time[-1]], 2*[1.0], ':')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Normalized Joint Vel')
    leg = plt.legend()
#    leg.get_frame().set_alpha(0.5)
    
plt.show()
