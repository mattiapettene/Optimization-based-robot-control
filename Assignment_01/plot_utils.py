import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_xy(time, walking_time, foot_length, foot_width, Z_ref, Z_x, Z_y, X, Y):
    plt.plot(Z_x, Z_y, 'r', label = r'computed CoP')
    plt.plot( X[0:walking_time,0], Y[0:walking_time,0], 'lime',
            label = r'CoM')
    currentAxis    = plt.gca()
    for i in range(walking_time):
        current_foot = patches.Rectangle((Z_ref[i,0]-foot_length/2,
                                          Z_ref[i,1]-foot_width/2),                                            foot_length, foot_width,   \
                                          linewidth = 0.8,
                                          linestyle = '-.',
                                          edgecolor = 'b',
                                          facecolor = 'none')
        currentAxis.add_patch(current_foot)
    currentAxis.set_xlim([-0.5,5.0])
    currentAxis.set_ylim([-0.5,0.8])
    plt.xlabel(r'x (m)')
    plt.ylabel(r'y (m)')
    plt.legend()
    plt.show()

def movePlotSpines(ax, spinesPos):
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', spinesPos[0]))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', spinesPos[1]))

def create_empty_figure(nRows=1, nCols=1, figsize=(7, 7), spinesPos=None, sharex=True):
    f, ax = plt.subplots(nRows, nCols, figsize=figsize, sharex=sharex)
    mngr = plt.get_current_fig_manager()

    if spinesPos is not None:
        if nRows * nCols > 1:
            for axis in ax.reshape(nRows * nCols):
                movePlotSpines(axis, spinesPos)
        else:
            movePlotSpines(ax, spinesPos)
    return f, ax

# Meshcat utils

def meshcat_material(r, g, b, a):
    import meshcat

    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material

def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))

# Gepetto/meshcat abstraction

def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        import meshcat
        viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addBox(name, sizex, sizey, sizez, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)

def addViewerSphere(viz, name, size, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        import meshcat
        viz.viewer[name].set_object(meshcat.geometry.Sphere(size),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addSphere(name, size, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)

def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.applyConfiguration(name, xyzquat)
        viz.viewer.gui.refresh()
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)

