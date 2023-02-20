# Optimization-based robot control projects

## First assignment (TSID)
The objectives of the first assignment are making practice using Task Space Inverse Dynamics (TSID) to make a humanoid robot walk and understanding the role of the task weights in order to tune them.
The pdf [report](https://github.com/mattiapettene/orc-project/blob/main/Report_assignment_01.pdf) contains the entire project procedure and all the considerations related to it, while in this [folder](https://github.com/mattiapettene/orc-project/tree/main/Assignment_01_TSID) there is all the Python code developed to simulate the humanoid robot walk.

## Second assignment (DDP)
The goals of the second assignment are using Differential Dynamic Programming (DDP) for generating trajectories and feedback control gains, then comparing two different methods for considering underactuation in DDP applied to the case of a double pendulum without motor on the second joint. All the considerations are reported in [this](https://github.com/mattiapettene/orc-project/blob/main/Report_assignment_02.pdf) report, while all the code related to this assignment is in this [folder](https://github.com/mattiapettene/orc-project/tree/main/Assignment_02_DDP).

## Third assignment (DQN)
In this project is required to implement the algorithm Deep Q-Network (DQN) and use it to find the optimal control policy for a single and a double pendulum swing-up problem. The double pendulum has only one actuated joint. The theoretical part and all the plots are reported in this [report](https://github.com/mattiapettene/orc-project/blob/main/Report_assignment_03.pdf), while the code is in this [folder](https://github.com/mattiapettene/orc-project/tree/main/Assignment_03_DQN).


&nbsp;
&nbsp;

## Environment setup

Open the terminal and execute the following commands:

```
sudo apt install python3-numpy python3-scipy python3-matplotlib curl

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -

sudo apt-get update
```

If you are using Ubuntu 20.04 run:

```
sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```

If you are using Ubuntu 22.04 run:

```
sudo apt install robotpkg-py310-pinocchio robotpkg-py310-example-robot-data robotpkg-urdfdom robotpkg-py310-qt5-gepetto-viewer-corba robotpkg-py310-quadprog robotpkg-py310-tsid
```

Configure the environment variables by adding the following lines to your file ~/.bashrc:

```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
export LOCOSIM_DIR=$HOME/orc/<folder_containing_locosim>/locosim
```

where <folder_containing_orc> is the folder containing the "orc" folder, which in turns contains all the python code of this class.

#### Test

You can check whether the installation went fine by trying to run this python script:

```
python3 test_software.py
```
You should see a new window appearing, displaying a robot moving somewhat randomly.
