# -*- coding: utf-8 -*-
"""

@author: adelpret
"""

import numpy as np
from ddp import DDPSolver
import pinocchio as pin

class DDPSolverLinearDyn(DDPSolver):
    ''' The linear system dynamics are defined by:
            x_{t+1} = A x_t + B u_t
        The task is defined by a quadratic cost: sum_{i=0}^N 0.5 x' H_{xx,i} x + h_{x,i} x + h_{s,i}
        plus a control regularization: sum_{i=0}^{N-1} lmbda ||u_i||.
    '''
    
    def __init__(self, name, ddp_params, H_xx, h_x, h_s, lmbda, underact, dt, CONTROL_BOUNDS, DEBUG=False,  w_bounds=0, max_torque=100, beta=0.1, eps = 1e-3):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.H_xx = H_xx
        self.h_x = h_x
        self.h_s = h_s
        self.lmbda = lmbda
        self.underact = underact
        self.CONTROL_BOUNDS = CONTROL_BOUNDS
        self.w_bounds = w_bounds
        self.max_torque = max_torque
        self.beta = beta
        self.eps = eps
        self.dt = dt
        self.nx = h_x.shape[1]
        self.nu = self.nx
        
    def cost(self, X, U):
        ''' total cost (running+final) for state trajectory X and control trajectory U '''
        N = U.shape[0]
        cost = self.cost_final(X[-1,:])
        for i in range(N):
            cost += self.cost_running(i, X[i,:], U[i,:])
        return cost
        
    def cost_running(self, i, x, u):                                    
        ''' Running cost at time step i for state x and control u '''
        cost = 0.5*np.dot(x, np.dot(self.H_xx[i,:,:], x)) \
                + np.dot(self.h_x[i,:].T, x) + self.h_s[i] \
                + 0.5*self.lmbda*np.dot(u.T, u) \
                # + ... add here the running cost term for taking into account the underactuation
        if self.CONTROL_BOUNDS:
            # ... implement here the running cost term for taking into the control limits
        return cost
        
    def cost_final(self, x):
        ''' Final cost for state x '''
        cost = 0.5*np.dot(x, np.dot(self.H_xx[-1,:,:], x)) \
                + np.dot(self.h_x[-1,:].T, x) + self.h_s[-1] 
        return cost
        
    def cost_running_x(self, i, x, u):
        ''' Gradient of the running cost w.r.t. x '''
        c_x = self.h_x[i,:] + np.dot(self.H_xx[i,:,:], x)
        return c_x
        
    def cost_final_x(self, x):
        ''' Gradient of the final cost w.r.t. x '''
        c_x = self.h_x[-1,:] + np.dot(self.H_xx[-1,:,:], x)
        return c_x
        
    def cost_running_u(self, i, x, u):
        ''' Gradient of the running cost w.r.t. u '''
        c_u = self.lmbda * u \
        # + ... add here the derivative w.r.t u of the running cost term for taking into account the underactuation                            
        if self.CONTROL_BOUNDS:
            # ... implement here the derivative w.r.t u of the running cost term for taking into the control limits
        return c_u
        
    def cost_running_xx(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x '''
        return self.H_xx[i,:,:]
        
    def cost_final_xx(self, x):
        ''' Hessian of the final cost w.r.t. x '''
        return self.H_xx[-1,:,:]
        
    def cost_running_uu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. u '''
        c_uu = self.lmbda * np.eye(self.nu) \
        # + ... add here the second derivative w.r.t u of the running cost term for taking into account the underactuation
        if self.CONTROL_BOUNDS:
            # ... implement here the second derivative w.r.t u of the running cost term for taking into the control limits
        return c_uu
        
    def cost_running_xu(self, i, x, u):
        ''' Hessian of the running cost w.r.t. x and then w.r.t. u '''
        return np.zeros((self.nx, self.nu))

class DDPSolverDoublePendulum(DDPSolverLinearDyn):
    ''' 
        Derived class of DDPSolverLinearDyn implementing the multi-body dynamics of a double pendulum.
        The task is defined by a quadratic cost: sum_{i=0}^N 0.5 x' H_{xx,i} x + h_{x,i} x + h_{s,i}
        plus a control regularization: sum_{i=0}^{N-1} lmbda ||u_i||.
    '''
    
    def __init__(self, name, robot, ddp_params, H_xx, h_x, h_s, lmbda, underact, dt, CONTROL_BOUNDS, DEBUG=False, simu=None, w_bounds=0, max_torque=100, beta=0.1, eps = 1e-3):
        DDPSolver.__init__(self, name, ddp_params, DEBUG)
        self.robot = robot
        self.H_xx = H_xx
        self.h_x = h_x
        self.h_s = h_s
        self.lmbda = lmbda
        self.underact = underact
        self.CONTROL_BOUNDS = CONTROL_BOUNDS
        self.w_bounds = w_bounds
        self.max_torque = max_torque
        self.beta = beta
        self.eps = eps
        self.nx = h_x.shape[1]
        self.nu = robot.na
        self.dt = dt
        self.simu = simu
        
        nv = self.robot.nv # number of joints
        self.Fx = np.zeros((self.nx, self.nx))
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fu = np.zeros((self.nx, self.nu))
        self.dx = np.zeros(2*nv)
        
    ''' System dynamics '''
    def f(self, x, u):
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
        ddq = pin.aba(model, data, q, v, u)     
        self.dx[nv:] = ddq
        v_mean = v + 0.5*self.dt*ddq
        self.dx[:nv] = v_mean
        return x + self.dt * self.dx

    def f_x_fin_diff(self, x, u, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed with finite differences'''
        f0 = self.f(x, u)
        Fx = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            xp = np.copy(x)
            xp[i] += delta
            fp = self.f(xp, u)
            Fx[:,i] = (fp-f0)/delta
        return Fx
        
    def f_u_fin_diff(self, x, u, delta=1e-8):
        ''' Partial derivatives of system dynamics w.r.t. u computed with finite differences'''
        f0 = self.f(x, u)
        Fu = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            up = np.copy(u)
            up[i] += delta
            fp = self.f(x, up)
            Fu[:,i] = (fp-f0)/delta
                
        return Fu
        
    def f_x(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. x '''
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
                
        # first compute Jacobians for continuous time dynamics
        pin.computeABADerivatives(model, data, q, v, u)
        self.Fx[:nv, :nv] = 0.0
        self.Fx[:nv, nv:] = np.identity(nv)
        self.Fx[nv:, :nv] = data.ddq_dq
        self.Fx[nv:, nv:] = data.ddq_dv

        if conf.SELECTION_MATRIX==1 and conf.ACTUATION_PENALTY==0:
            S = np.array([[1.0,0.0],[0.0,0.0]])                        # Selection matrix for taking into account underactuation
            self.Fu[nv:, :] = data.Minv.dot(S.T)                       # Partial derivatives of system dynamics w.r.t. u    
        elif conf.SELECTION_MATRIX==0 and conf.ACTUATION_PENALTY==1:
            self.Fu[nv:, :] = data.Minv
        else:
            raise RuntimeError("No method has been chosen to consider the underactuated case")    

        # Convert them to discrete time
        self.Fx = np.identity(2*nv) + dt * self.Fx
        self.Fu *= dt
        
        return self.Fx
    
    def f_u(self, x, u):
        ''' Partial derivatives of system dynamics w.r.t. u '''
        return self.Fu
        
    def callback(self, X, U):
        pass
        for i in range(0, N):
            time_start = time.time()
            self.simu.display(X[i,:self.robot.nq])
            time_spent = time.time() - time_start
            if(time_spent < self.dt):
                time.sleep(self.dt-time_spent)
                
    def start_simu(self, x0, X_bar, U_bar, KK, dt_sim, PUSH, TORQUE_LIMITS):
        n = x0.shape[0]
        m = U_bar.shape[1]
        N = U_bar.shape[0]
        X_sim = np.zeros((N+1, n))
        U_sim = np.zeros((N, m))
        X_sim[0,:] = x0
        print("\n"+" SIMULATION RESULTS ".center(conf.LINE_WIDTH, '*'))
        print("Start simulation")
        for i in range(N):
            if TORQUE_LIMITS:
                U_sim[i,:] = np.clip(U_bar[i,:] - KK[i,:,:] @ (X_sim[i,:]-X_bar[i,:]),-self.max_torque, self.max_torque)
            else:
                U_sim[i,:] = U_bar[i,:] - KK[i,:,:] @ (X_sim[i,:]-X_bar[i,:])
            # Needed when using the penalty method for taking into account underactuation
            U_sim[i,1] = 0.0                                                                     
            if PUSH and (i == int(N/8) or i == int(N/4) or i == int(N/2) or i == int(3*N/2)):
                X_sim[i,int(n/2):] += conf.push_vec
            X_sim[i+1,:] = self.f(X_sim[i,:], U_sim[i,:])
            time_start = time.time()
            self.simu.display(X_sim[i,:self.robot.nq])
            time_spent = time.time() - time_start
            if(time_spent < dt_sim):
                time.sleep(dt_sim-time_spent)
        print("Simulation finished")    
        cost = self.cost(X_sim, U_sim)                                          
        print("Cost Sim.", cost)
        print("Effort Sim.", np.linalg.norm(U_sim))
        return X_sim, U_sim


        
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import orc.utils.plot_utils as plut
    import time
    from example_robot_data.robots_loader import load
    from orc.utils.robot_wrapper import RobotWrapper
    from orc.utils.robot_simulator import RobotSimulator
    import ddp_doublependulum_conf as conf
    np.set_printoptions(precision=3, suppress=True)
    
    ''' Test DDP with a double pendulum
    '''
    print("".center(conf.LINE_WIDTH,'#'))
    print(" DDP - Double Pendulum ".center(conf.LINE_WIDTH, '#'))
    print("".center(conf.LINE_WIDTH,'#'), '\n')

    N = conf.N               # horizon size
    dt = conf.dt             # control time step
    mu = 10                  # initial regularization
    ddp_params = {}
    ddp_params['alpha_factor'] = 0.5
    ddp_params['mu_factor'] = 10.
    ddp_params['mu_max'] = 1e0
    ddp_params['min_alpha_to_increase_mu'] = 0.1
    ddp_params['min_cost_impr'] = 1e-1
    ddp_params['max_line_search_iter'] = 10
    ddp_params['exp_improvement_threshold'] = 1e-3
    ddp_params['max_iter'] = 200
    DEBUG = False

    SELECTION_MATRIX = conf.SELECTION_MATRIX       
    ACTUATION_PENALTY = conf.ACTUATION_PENALTY       

    r = load("double_pendulum")
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    nq, nv = robot.nq, robot.nv
    
    n = nq+nv                                                   # state size
    m = robot.na                                                # control size
    U_bar = np.zeros((N,m));                                    # initial guess for control inputs
    x0 = np.concatenate((conf.q0, np.zeros(robot.nv)))          # initial state
    x_tasks = np.concatenate((conf.qT, np.zeros(robot.nv)))     # goal state
    N_task = N;                                                 # time step to reach goal state
    
    tau_g = robot.nle(conf.q0, np.zeros(robot.nv))
    for i in range(N):
        U_bar[i,:] = tau_g
    
    ''' TASK FUNCTION  '''
    lmbda = 2e1                                                     # control regularization
    if conf.TORQUE_LIMITS:
        w_bounds = 1e2                                              # weight of the penalty on the control limits (barrier function)
        max_torque = 0.75                                           # control limit
        beta = 0.1                                                  # parameter of the barrier function determining the sharpness of the boundaries
        eps = 1e-3                                                  # parameter of the barrier function to relax the boundaries
    if conf.SELECTION_MATRIX==1:
        underact = 0                                                # underactuation penalty weight
    elif conf.ACTUATION_PENALTY==1:
        underact = 1e6                                                    
    H_xx = np.zeros((N+1, n, n))
    h_x  = np.zeros((N+1, n))
    h_s  = np.zeros(N+1)
    W = np.diagflat(np.concatenate([2*np.ones(nq), np.zeros(nv)]))
    for i in range(N_task):
        H_xx[i,:,:]  = W
        h_x[i,:]     = -W @ x_tasks
        h_s[i]       = 0.5*x_tasks.T @ W @ x_tasks
    
    print("Displaying desired goal configuration")
    simu = RobotSimulator(conf, robot)
    simu.display(conf.qT)
    time.sleep(1.)
    
    if conf.TORQUE_LIMITS:
        solver = DDPSolverDoublePendulum("doublependulum", robot, ddp_params, H_xx, h_x, h_s, lmbda, underact, dt, conf.TORQUE_LIMITS, DEBUG, simu,  w_bounds, max_torque, beta, eps)
    else:
        solver = DDPSolverDoublePendulum("doublependulum", robot, ddp_params, H_xx, h_x, h_s, lmbda, underact, dt, conf.TORQUE_LIMITS, DEBUG, simu)
    
    (X,U,KK) = solver.solve(x0, U_bar, mu)
    
    if conf.SELECTION_MATRIX==1:    
        print("\nMETHOD = SELECTION MATRIX")
    elif conf.ACTUATION_PENALTY==1:
        print("\nMETHOD = ACTUATION_PENALTY")
    
    solver.print_statistics(x0, U, KK, X, conf.LINE_WIDTH)
    
    print("Show reference motion")
    for i in range(0, N):
        time_start = time.time()
        simu.display(X[i,:nq])
        time_spent = time.time() - time_start
        if(time_spent < dt):
            time.sleep(dt-time_spent)
    print("Reference motion finished")
    time.sleep(1)
    
    print("Show real simulation")
    X_sim, U_sim = solver.start_simu(x0, X, U, KK, conf.dt_sim, conf.PUSH, conf.TORQUE_LIMITS)
    time_vec = np.linspace(0.0,conf.N*conf.dt,N+1)

    if conf.PLOT_TORQUES:        
        plt.figure()
        plt.plot(time_vec[:-1], U_sim[:,0], "b")
        plt.plot(time_vec[:-1], U[:,0], "b--", alpha=0.8, linewidth=1.5)
        if conf.TORQUE_LIMITS:
            plt.plot(time_vec[:-1], max_torque*np.ones(len(time_vec[:-1])), "k--", alpha=0.8, linewidth=1.5)
            plt.plot(time_vec[:-1], -max_torque*np.ones(len(time_vec[:-1])), "k--", alpha=0.8, linewidth=1.5)
        plt.gca().set_xlabel('Time [s]')
        plt.gca().set_ylabel('[Nm]')
        leg = plt.legend(["1st joint torque", "1st joint ref. torque"],loc='upper right')
    
    if conf.PLOT_JOINT_POS:        
        plt.figure()
        plt.plot(time_vec, X_sim[:,0],'b')
        plt.plot(time_vec, X[:,0],'b--', alpha=0.8, linewidth=1.5)
        plt.plot(time_vec, X_sim[:,1],'r')
        plt.plot(time_vec, X[:,1],'r--', alpha=0.8, linewidth=1.5)
        plt.gca().set_xlabel('Time [s]')
        plt.gca().set_ylabel('[rad]')
        plt.legend(["1st joint position","1st joint ref. position","2nd joint position","2nd joint ref position"],loc='upper right')
        
    if conf.PLOT_JOINT_VEL:        
        plt.figure()
        plt.plot(time_vec, X_sim[:,2],'b')
        plt.plot(time_vec, X[:,2],'b--', alpha=0.8, linewidth=1.5)
        plt.plot(time_vec, X_sim[:,3],'r')
        plt.plot(time_vec, X[:,3],'r--', alpha=0.8, linewidth=1.5)
        plt.gca().set_xlabel('Time [s]')
        plt.gca().set_ylabel('[rad/s]')
        plt.legend(["1st joint velocity","1st joint ref. velocity","2nd joint velocity","2nd joint ref velocity"],loc='upper right')

    plt.show()

