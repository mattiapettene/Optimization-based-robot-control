import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from numpy.random import randint, uniform
import matplotlib.pyplot as plt
import time
from numpy.random import rand
import tensorflow as tf

from dpendulum import DPendulum
from classes import AGENT, BUFFER, TRAINING, PLOT

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)
#RANDOM_SEED=1000

### --- Hyper paramaters
NEPISODES                    = 100       # Number of training episodes
NPRINT                       = 25        # print something every NPRINT episodes
MAX_EPISODE_LENGTH           = 100       # Max episode length
QVALUE_LEARNING_RATE         = 1e-3      # alpha coefficient of Q learning algorithm
DISCOUNT                     = 0.99      # Discount factor 
PLOT_LABEL                   = True      # Plot stuff if True
EXPLORATION_PROB             = 1         # initial exploration probability of eps-greedy policy
EXPLORATION_DECREASING_DECAY = 0.05      # exploration decay for exponential decreasing
MIN_EXPLORATION_PROB         = 0.001     # minimum of exploration probability
CAPACITY_BUFFER              = 2000       # capacity buffer
BATCH_SIZE                   = 32        # batch size 
MIN_BUFFER                   = 100       # Start sampling from buffer when have length > MIN_BUFFER
C_STEP                       = 250       # Every c step update w

# Select the number of joints
nj = int(input("Select the number of joints [1/2]: "))
if (nj != 1 and nj != 2):
   exit("Number of joints not valid")

# Select if to train a new model or import an existing one
train = input("Would you like to train a new model instead of import the existing one? [yes/no] ")
if (train != 'yes' and train != 'no'):
   exit("Input not valid")

# ----- Control/State
njoint                       = nj         # number of joint
nx                           = 2*njoint  # number of states
nu                           = 1         # number of control input
ndcontrol                    = 14        # number of discretization steps for the joint torque u
ndstates                     = 14        # number of discretization steps for the joint state (for plot)


### --- Initialize environment, agent, buffer, training
env = DPendulum(njoint, ndcontrol)
agent = AGENT(nx, nu, env, DISCOUNT, QVALUE_LEARNING_RATE)
agent.Q.summary()
buffer = BUFFER(CAPACITY_BUFFER, BATCH_SIZE)
training = TRAINING()


if train == 'yes':
  # Train a new model
  costtogo = training.dqn_learning(buffer, agent, env, DISCOUNT, NEPISODES, MAX_EPISODE_LENGTH, MIN_BUFFER, C_STEP, EXPLORATION_PROB, EXPLORATION_DECREASING_DECAY, MIN_EXPLORATION_PROB, PLOT_LABEL, NPRINT)
  plt.show()

  # Save trained model
  if (njoint == 1):
    print("\nSave NN model and weights to file (in HDF5)")
    agent.Q.save("saving/model1joint")
    agent.Q.save_weights("saving/model1joint_weights.h5")
  else:
    print("\nSave NN model and weights to file (in HDF5)")
    agent.Q.save("saving/model2joint")
    agent.Q.save_weights("saving/model2joint_weights.h5")

  # Plot figures
  plt.figure()
  plt.plot(np.cumsum(costtogo)/range(1,NEPISODES+1))
  plt.title("Average Cost-to-go")
  
else:
  # Load an existing model
  if(njoint == 1):
    agent.Q = tf.keras.models.load_model("saving/model1joint")
    agent.Q_target.load_weights("saving/model1joint_weights.h5")
  else:
    agent.Q = tf.keras.models.load_model("saving/model2joint")
    agent.Q_target.load_weights("saving/model2joint_weights.h5")
  assert(agent.Q)

# Compute cost function V and policy pi
if(njoint==1):
  V, pi, xgrid = agent.Q2V_pi(env, ndstates, njoint)

  env.plot_V_table(V, xgrid)
  env.plot_policy(pi, xgrid)
  print("Average|min|max Value function (V)", np.mean(V), "|", np.min(V), "|", np.max(V))

X_sim, U_sim, C_sim = training.get_greedy_policy(env, agent, None, MAX_EPISODE_LENGTH, DISCOUNT)


# Plots
if(PLOT_LABEL):
  x_axis = np.linspace(0.0, MAX_EPISODE_LENGTH*env.pendulum.DT, MAX_EPISODE_LENGTH)
  PLOT.plot_figures(x_axis, X_sim, U_sim, C_sim, env)
  plt.show()

