from pendulum import Pendulum
import numpy as np
from numpy import pi
import time
    

class DPendulum:
    ''' Discrete Pendulum environment. Joint torque are discretized
        with the specified steps. Joint velocity and torque are saturated. 
        Guassian noise can be added in the dynamics. 
        Cost is -1 if the goal state has been reached, zero otherwise.
    '''
    def __init__(self, njoint, nu, vMax=5, uMax=5, dt=0.2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(njoint, noise_stddev)
        self.njoint = njoint # Number of joints
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.vMax = vMax    # Max velocity (v in [-vmax,vmax])
        self.nu = nu        # Number of discretization steps for joint torque
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.dt = dt        # time step
        self.DU = 2*uMax/nu # discretization resolution for joint torque

    
    # Joint torque continuous->discrete
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))
    
    # Discrete to continuous
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU
    
    def reset(self,x=None):
        self.x = self.pendulum.reset(x)
        return self.x

    def step(self,iu):
        u   = self.d2cu(iu)
        self.x, cost = self.pendulum.step(u)
        return self.x, cost

    def render(self):
        self.pendulum.render()
        time.sleep(self.pendulum.DT)
    
    def plot_V_table(self, V, x):
        ''' Plot the given Value table V '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(x[0], x[1], V, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('V table')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_policy(self, pi, x):
        ''' Plot the given policy table pi '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(x[0], x[1], pi, cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.title('Policy')
        plt.xlabel("q")
        plt.ylabel("dq")
        plt.show()
        
    def plot_Q_table(self, Q):
        ''' Plot the given Q table '''
        import matplotlib.pyplot as plt
        X,U = np.meshgrid(range(Q.shape[0]),range(Q.shape[1]))
        plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        plt.title('Q table')
        plt.xlabel("x")
        plt.ylabel("u")
        plt.show()
