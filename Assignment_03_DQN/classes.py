import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
import time
import random
from collections import deque
import matplotlib.pyplot as plt

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class AGENT:

  def __init__(self, NX, NU, ENV, DISCOUNT, Q_VALUE_LEARNING_RATE):

    """
        nx: number of states
        nu: number of control input
        discount: discount factor 
        q_value_learning_rate: alpha coefficient of Q learning algorithm
    """
    self.nx = NX
    self.nu = NU
    self.ndu = ENV.nu
    self.discount = DISCOUNT
    self.q_value_learning_rate = Q_VALUE_LEARNING_RATE
    self.Q = self.get_critic()
    self.Q_target = self.get_critic()
    self.critic_optimizer = tf.keras.optimizers.Adam(self.q_value_learning_rate)
    self.Q_target.set_weights(self.Q.get_weights())

  def np2tf(self, y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
  def tf2np(self, y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()

  def get_critic(self):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(self.nx+self.nu,))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs)

    return model

  def get_action(self, exploration_prob, env, x, egreedy):
    ''' epsilon-greedy policy strategy '''

    # Given exploration_prob take random control input
    if(uniform() < exploration_prob and egreedy == True):
      u = randint(0, env.nu)
    # otherwise take greedy control
    else:
      x = np.array([x]).T
      xu = np.reshape([np.append([x]*np.ones(env.nu),[np.arange(env.nu)])],(env.pendulum.nx+1,env.nu))
      if (egreedy == False):
        u = np.argmax((self.Q_target(xu.T)))

      else:
        u = np.argmax((self.Q(xu.T)))

    return u

  def update(self, xu_batch, cost_batch, xu_next_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape: 
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = self.Q_target(xu_next_batch, training=True) 
        # Compute 1-step targets for the critic loss
        y = cost_batch + self.discount*target_values                            
        # Compute batch of Values associated to the sampled batch of states
        Q_value = self.Q(xu_batch, training=True)                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, self.Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    self.critic_optimizer.apply_gradients(zip(Q_grad, self.Q.trainable_variables))

  def update_Q_target(self):
    # Update the current Q_target weight with the weight of Q at that moment
    self.Q_target.set_weights(self.Q.get_weights())

    w = self.Q.get_weights()
    # for i in range(len(w)):
    #     print("Shape Q weights layer", i, w[i].shape)
        
    # for i in range(len(w)):
    #     print("Norm Q weights layer", i, np.linalg.norm(w[i]))
        
    # print("\nDouble the weights")
    # for i in range(len(w)):
    #   w[i] *= 2
    # self.Q.set_weights(w)

    # w = self.Q.get_weights()
    # for i in range(len(w)):
    #   print("Norm Q weights layer", i, np.linalg.norm(w[i]))

    # w = self.Q_target.get_weights()
    # for i in range(len(w)):
    #   print("Norm Q weights layer", i, np.linalg.norm(w[i]))

  def Q2V_pi(self, env, nstates, njoint):
    ''' Compute value table and policy pi '''

    vmax = env.vMax
    nx = env.pendulum.nx

    # Initialize pi, V and x
    pi = np.zeros(shape=(nstates, nstates))
    V = np.zeros(shape=(nstates, nstates))
    x = np.zeros(shape=(nx, nstates))

    # Discretization of position and velocity
    DQ = 2*np.pi/(nstates-1) 
    DV = 2*vmax/(nstates-1)
    

    for i in range(nstates):
      for j in range(nstates):
        
        x[0,:] = np.arange(-np.pi, np.pi+DQ, DQ)
        x[1,:] = np.arange(-vmax, vmax+DV, DV)
        xu = np.reshape([x[0,i]*np.ones(self.ndu),x[1,j]*np.ones(self.ndu),np.arange(self.ndu)],(nx+1,self.ndu))
        
        V[i,j] = np.max(self.Q(xu.T))
        pi[i,j] = env.d2cu(np.argmax(self.Q(xu.T)))

    return V, pi, x 


class BUFFER:

  ''' Store and sample data '''

  def __init__(self, CAPACITY_BUFFER, BATCH_SIZE):
    self.replay_buffer = deque(maxlen=CAPACITY_BUFFER)
    self.capacity_buffer = CAPACITY_BUFFER
    self.batch_size = BATCH_SIZE

  def store_data(self, state, control, cost, next_state):
    # memorize the last experience in replay buffer
    experience = [state,control,cost,next_state]
    self.replay_buffer.append(experience)

  def sample_batch(self, env, exploration_prob, agent):
    # sample from replay_buffer a batch of batch_size experiences
    batch = random.choices(self.replay_buffer,k=self.batch_size)
    x_batch, u_batch, cost_batch, next_state_batch = list(zip(*batch))
    u_next_batch = np.empty(self.batch_size)

    x_batch = np.concatenate([x_batch],axis=1).T
    u_batch = np.asarray(u_batch)
    cost_batch = np.asarray(cost_batch)

    for i in range(self.batch_size):
      u_next_batch[i] = agent.get_action(exploration_prob,env,next_state_batch[i],False)
    
    next_state_batch = np.concatenate([next_state_batch],axis=1).T
    xu_batch = np.reshape(np.append(x_batch,u_batch),(env.pendulum.nx+1,self.batch_size))
    xu_next_batch = np.reshape(np.append(next_state_batch,u_next_batch),(env.pendulum.nx+1,self.batch_size))
    cost_batch = np.reshape(cost_batch, (self.batch_size))

    return xu_batch,xu_next_batch,cost_batch   


class TRAINING:

  def __init__(self) -> None:
    pass

  def get_greedy_policy(self, env, agent, x0, maxEpisodeLength, discount):
    ''' Simulate the system using greedy strategy inside get_action and return experiences '''

    # Iniziatialize vectors
    X_sim = np.zeros([maxEpisodeLength, env.pendulum.nx])
    U_sim = np.zeros(maxEpisodeLength)
    C_sim = np.zeros(maxEpisodeLength)

    x0 = None
    x0 = x = env.reset(x0)

    J = 0.0   # Cost-to-go initialization

    gamma_seq = 1

    for i in range(maxEpisodeLength):
      u = agent.get_action(0, env, x, True)

      if(env.njoint == 2):
        x,c = env.step([u, env.c2du(0.0)])
      else: 
        x,c = env.step([u])
      
      J = gamma_seq*c
      gamma_seq *= discount

      env.render()

      X_sim[i,:] = np.concatenate(np.array([x]).T)
      U_sim[i] = env.d2cu(u)
      C_sim[i] = c

    print("Cost to go from state", x0, "is", J)

    return X_sim, U_sim, C_sim


  def dqn_learning(self, buffer, agent, env,\
                 discount, nEpisodes, maxEpisodeLength, min_buffer, c_step,\
                 exploration_prob, exploration_decreasing_decay, min_exploration_prob, \
                 plot, nprint):

    # Initialization cost-to-go
    costtogo = []
    counter = 0
    tot_time = 0

    # For each episode
    for i in range(nEpisodes):
      env.reset()

      J = 0
      gamma_seq = 1    # gamma value which will be update at each step in TD

      # Getting time for each episode
      start = time.time()

      # Start the simulation for maxEpisodesLenght steps
      for m in range(maxEpisodeLength):

        x = env.x # state of the envinonment

        u = agent.get_action(exploration_prob, env, x, True) # e-greedy approach to get best action

        if (env.njoint==2):
          next_state, cost = env.step([u,env.c2du(0.0)]) # second joint not actuated
        else:
          next_state, cost = env.step([u]) # only one joint

        buffer.store_data(x, u, cost, next_state)

        ## use stored data to update the weight for the Q_target function
        if len(buffer.replay_buffer) > min_buffer:
          # collect state and action randomly from replay_buffer 
          xu_batch,xu_next_batch,cost_batch = buffer.sample_batch(env, exploration_prob, agent)

          # np -> tf 
          xu_batch = agent.np2tf(xu_batch)
          xu_next_batch = agent.np2tf(xu_next_batch)
          cost_batch = agent.np2tf(cost_batch)

          # update weights 
          agent.update(xu_batch,cost_batch,xu_next_batch)

          # update target network every c-step
          counter += 1
          if counter % c_step == 0:
            agent.update_Q_target()
        
        # Update cost to go funcion J of the TD approach in the m-step simulation
        J += gamma_seq * cost
        gamma_seq *= discount # update for the next iteration
      
      costtogo.append(J)

      # reducing the exploration probability 
      exploration_prob = max(np.exp(-exploration_decreasing_decay*i), min_exploration_prob)
      
      duration = round((time.time()-start),3)

      print("Episode", i, "completed in", duration, "s - exp_prob =", round(100*exploration_prob,2), "| cost_to_go (J) =", round(J,2))

      tot_time += duration        

      # Plot and simulate 
      if(i % nprint==0 and i>=nprint): 
        X_sim, U_sim, C_sim = self.get_greedy_policy(env, agent, None, maxEpisodeLength, discount)

        if(plot):
          x_axis = np.linspace(0.0, maxEpisodeLength*env.pendulum.DT, maxEpisodeLength)
          plt.show(block=False)
          PLOT.plot_figures(x_axis, X_sim, U_sim, C_sim, env)
          plt.show()

    # Total training time 
    print("Total training time = ", round(tot_time,3), "s")

    with open("saving/time.txt", "w") as f:
      # Write a string to the file
      f.write("Total training time = " + str(round(tot_time,3)))

  
    return costtogo


class PLOT:

  def plot_figures(x_axis, X_sim, U_sim, C_sim, env):
    plt.figure()
    plt.plot(x_axis, U_sim[:], "b")
    if env.uMax:
        plt.plot(x_axis, env.uMax*np.ones(len(x_axis)), "k--", alpha=0.8, linewidth=1.5)
        plt.plot(x_axis, -env.uMax*np.ones(len(x_axis)), "k--", alpha=0.8, linewidth=1.5)
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[Nm]')
    plt.title ("Torque input")

    plt.figure()
    plt.plot(x_axis, C_sim[:], "b")
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Cost')
    plt.title ("Cost")

    plt.figure()
    if env.njoint == 1:
        plt.plot(x_axis, X_sim[:,0],'b')
    else:
        plt.plot(x_axis, X_sim[:,0],'b')
        plt.plot(x_axis, X_sim[:,1],'r')
        plt.legend(["1st joint position","2nd joint position"],loc='upper right')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad]')
    plt.title ("Joint position")
    
    plt.figure()
    if env.njoint == 1:
        plt.plot(x_axis, X_sim[:,1],'b')
    else:
        plt.plot(x_axis, X_sim[:,2],'b')
        plt.plot(x_axis, X_sim[:,3],'r')

        plt.legend(["1st joint velocity","2nd joint velocity"],loc='upper right')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad/s]')
    plt.title ("Joint velocity")




  



    
